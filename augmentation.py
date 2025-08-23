"""
Augmentation Dataset Cross-Domain -> In-Domain
Notes: NOT USE - Performance bị ảnh hưởng bởi Pos_Tagging + Pretrained Model Embedding
"""

from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import torch
import torch.nn as nn
from icecream import ic
import faiss
from collections import defaultdict
import spacy
import pandas as pd
import itertools
from vncorenlp import VnCoreNLP
from tqdm import tqdm
import re

NLP = spacy.load("en_core_web_sm")
mapping_keywords = {
    'bn': 'bệnh nhân',
    'th': 'trường hợp'
}

def replace_keywords(text, mapping):
    pattern = r'\b(' + '|'.join(re.escape(k) for k in mapping.keys()) + r')\b'
    def repl(match):
        key = match.group(0)
        return mapping_keywords.get(key, key)
    return re.sub(pattern, repl, text)

class Dictionary_base_Augmentation(nn.Module):
    def __init__(self, model_name: str, batch_size: int, threshold: float, align_layer: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.rdrsegmenter = VnCoreNLP("./VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx500m')
        self.batch_size = batch_size
        self.threshold = threshold
        self.align_layer = align_layer

    def get_batch_embeddings(self, batch: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(batch, max_length=512, padding=True, truncation=True,return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        batch_embeddings = outputs.pooler_output.cpu().numpy() # Access pooled output (from [CLS])
        return batch_embeddings


    def extract_en_noun_phrases(self, sentence: str) -> list[str]:
        doc_sen = NLP(sentence)
        phrases = []
        for chunk in doc_sen.noun_chunks:
            if not any(token.pos_ == 'NUM' for token in chunk):
                phrases.append(chunk.text.strip())
        return phrases

    def extract_vi_noun_phrases(self, sentence: str) -> list[str]:
        sentence = replace_keywords(sentence, mapping_keywords)
        pos_tagged = self.rdrsegmenter.pos_tag(sentence)[0]
        noun_tags = ['N', 'Np', 'Nc', 'Nu', 'Ny', 'Nb', 'A', 'P']

        phrases = []
        curr_phrases = []
        for token, tag in pos_tagged:
            if tag in noun_tags:
                curr_phrases.append(token)
            else:
                if curr_phrases:
                    phrase = " ".join(curr_phrases).strip()
                    phrase = phrase.strip(",.?!;:()[]")
                    if phrase:
                        phrases.append(phrase)
                    curr_phrases = []
        if curr_phrases:
            phrase = " ".join(curr_phrases).strip()
            phrase = phrase.strip(",.?!;:()[]")
            if phrase:
                phrases.append(phrase)
        return phrases

    def align(self, src_phrases: list[str], tgt_phrases: list[str]):
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in src_phrases], [self.tokenizer.tokenize(word) for word in tgt_phrases]
        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)),
            return_tensors='pt',
            model_max_length=self.tokenizer.model_max_length,
            truncation=True,
        )['input_ids'].to(self.device)
        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)),
            return_tensors='pt',
            model_max_length=self.tokenizer.model_max_length,
            truncation=True
        )['input_ids'].to(self.device)

        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]

        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            softmax_srctgt = nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = nn.Softmax(dim=-2)(dot_prod)
            softmax_inter = (softmax_srctgt > self.threshold) * (softmax_tgtsrc > self.threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        mapping_align_words = defaultdict(list)
        for i, j in align_subwords:
            src_phrase = src_phrases[sub2word_map_src[i]]
            tgt_phrase = tgt_phrases[sub2word_map_tgt[j]].strip()
            if tgt_phrase not in mapping_align_words[src_phrase]:
                mapping_align_words[src_phrase].append(tgt_phrase.replace("_", " "))
        return mapping_align_words

    def forward(self, en_ind_corpus: list, vi_ind_corpus: list, ind_dic=None) -> dict:
        final_mapping = {}
        for S, T in tqdm(zip(en_ind_corpus, vi_ind_corpus), desc='Mapping Align Words', total=len(en_ind_corpus)):
            phrases_Sk = self.extract_en_noun_phrases(S)
            phrases_Tk = self.extract_vi_noun_phrases(T)
            mapping_align_words = self.align(phrases_Sk, phrases_Tk)
            for en_phrase, vi_phrases in mapping_align_words.items():
                if en_phrase not in final_mapping:
                    final_mapping[en_phrase] = set()
                final_mapping[en_phrase].update(vi_phrases)

            to_write = {k: list(v) for k, v in final_mapping.items()}
            with open("appendix_dic.json", 'w', encoding='utf-8') as f:
                json.dump(to_write, f, ensure_ascii=False, indent=4)

        return {k: list(v) for k, v in final_mapping.items()}

if __name__ == '__main__':
    en_ind_corpus = open("/content/drive/MyDrive/[VLSP] MT in Medical Domain/data/Final/train.en.txt", 'r', encoding='utf-8')
    vi_ind_corpus = open("/content/drive/MyDrive/[VLSP] MT in Medical Domain/data/Final/train.vi.txt", 'r', encoding='utf-8')
    en_list, vi_list = [], []
    for en, vi in zip(en_ind_corpus, vi_ind_corpus):
        en_list.append(en.strip())
        vi_list.append(vi.strip())
    aug_dic_method = Dictionary_base_Augmentation("bert-base-multilingual-cased", batch_size=256, threshold=1e-4, align_layer=8)
    final_mapping = aug_dic_method(en_list, vi_list)