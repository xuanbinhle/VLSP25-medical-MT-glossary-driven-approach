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
    def __init__(self, model_name: str, batch_size: int, threshold: float, align_layer: int, topN: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.rdrsegmenter = VnCoreNLP("./VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx500m')
        self.batch_size = batch_size
        self.threshold = threshold
        self.align_layer = align_layer
        self.topN = topN

    def get_batch_embeddings(self, batch: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(batch, max_length=512, padding=True, truncation=True,return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        batch_embeddings = outputs.pooler_output.cpu().numpy() # Access pooled output (from [CLS])
        return batch_embeddings

    def get_topN_src_sentences(self, src_ood_parallel_corpus_embeddings, src_ind_dic_embeddings) -> list[list[int]]:
        index = faiss.IndexFlatIP(src_ood_parallel_corpus_embeddings.shape[-1])
        index.add(src_ood_parallel_corpus_embeddings)
        faiss.normalize_L2(src_ood_parallel_corpus_embeddings) # Normalize L2
        faiss.normalize_L2(src_ind_dic_embeddings) # Normalize L2
        _, indices = index.search(src_ind_dic_embeddings, k=self.topN)
        return indices.tolist()

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
        noun_tags = ['N', 'Np', 'Nc', 'Nu', 'Ny', 'Nb', 'A']

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
                mapping_align_words[src_phrase].append(tgt_phrase)
        return mapping_align_words

    def substitute(self, ood_sentence: str, phrase_max_ood: str, phrase_ind: str) -> str:
        return ood_sentence.replace(phrase_max_ood, phrase_ind).strip()

    def forward(self, en_ind_corpus: list, vi_ind_corpus: list, ind_dic=None) -> dict:
        # src_ind_parallel_corpus, tgt_ind_parallel_corpus = [], []
        # for S, T in ind_parallel_corpus:
        #     src_ind_parallel_corpus.append(S)
        #     tgt_ind_parallel_corpus.append(T)

        # src_ind_dic, tgt_ind_dic = [], []
        # for src, tgt in ind_dic:
        #     src_ind_dic.append(src)
        #     tgt_ind_dic.append(tgt)

        # src_ood_parallel_corpus_embeddings = self.get_batch_embeddings(src_ind_parallel_corpus)
        # src_ind_dic_embeddings = self.get_batch_embeddings(src_ind_dic)
        # src_ood_topN_sen_corpus_indices = self.get_topN_src_sentences(src_ood_parallel_corpus_embeddings, src_ind_dic_embeddings)
        # assert len(src_ood_topN_sen_corpus_indices) == src_ind_dic_embeddings.shape[0], "Not Suitable Size"
        
        # self.f_en_out = open(f"{self.file_out}.en.txt", 'w', encoding='utf-8')
        # self.f_vi_out = open(f"{self.file_out}.vi.txt", 'w', encoding='utf-8')
        # for i in tqdm(range(0, len(src_ind_parallel_corpus), self.batch_size), desc='Calculating Similarity'):
        #     src_batch = src_ind_parallel_corpus[i:i+self.batch_size]
        #     tgt_batch = tgt_ind_parallel_corpus[i:i+self.batch_size]
            
        #     src_embedding = self.get_batch_embeddings(src_batch)
        #     tgt_embedding = self.get_batch_embeddings(tgt_batch)
        #     faiss.normalize_L2(src_embedding)
        #     faiss.normalize_L2(tgt_embedding)
        #     for id_sen in src_ood_topN_sen_corpus_indices[i]:
        #         phrases_Sk = self.extract_en_noun_phrases(src_odd_parallel_corpus[id_sen])
        #         phrases_Tk = self.extract_vi_noun_phrases(tgt_odd_parallel_corpus[id_sen])
        #         mapping_align_words = self.align(phrases_Sk, phrases_Tk)

        #         phrases_embeddings = self.get_batch_embeddings(phrases_Sk)
        #         index_phrase = faiss.IndexFlatIP(phrases_embeddings.shape[-1])
        #         faiss.normalize_L2(phrases_embeddings)
        #         index_phrase.add(phrases_embeddings)
        #         _, max_sim_index = index_phrase.search(src_embedding, k=1)
        #         phrase_max_Sk = phrases_Sk[max_sim_index[0].item()]
        #         if phrase_max_Sk not in mapping_align_words:
        #             continue
        #         phrase_max_Tk = mapping_align_words[phrase_max_Sk]
        #         if phrase_max_Tk not in mapping_align_words:
        #             continue

        #         aug_src_sen = self.substitute(src_odd_parallel_corpus[id_sen], phrase_max_Sk, src_ind_dic[i])
        #         aug_tgt_sen = self.substitute(tgt_odd_parallel_corpus[id_sen], phrase_max_Tk, tgt_ind_dic[i])
        #         ic(aug_src_sen, aug_tgt_sen)
        #         raise
        #         self.f_en_out.write(f"{aug_src_sen}\n")
        #         self.f_vi_out.write(f"{aug_tgt_sen}\n")
        #         self.f_en_out.flush()
        #         self.f_vi_out.flush()
        
        final_mapping = {}
        for S, T in tqdm(zip(en_ind_corpus, vi_ind_corpus), desc='Mapping Align Words'):
            phrases_Sk = self.extract_en_noun_phrases(S)
            phrases_Tk = self.extract_vi_noun_phrases(T)
            mapping_align_words = self.align(phrases_Sk, phrases_Tk)
            ic(mapping_align_words)
            # for en, vi in mapping_align_words.items():
            #     final_mapping[en] 

if __name__ == '__main__':
    en_ind_corpus = open("./data/Final/train.en.txt", 'r', encoding='utf-8')
    vi_ind_corpus = open("./data/Final/train.vi.txt", 'r', encoding='utf-8')
    aug_dic_method = Dictionary_base_Augmentation("./bert-base-multilingual-cased", batch_size=256, threshold=1e-3, align_layer=8, topN=10)
    final_mapping = aug_dic_method(en_ind_corpus, vi_ind_corpus)
    with open("appendix_dic.json", 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, ensure_ascii=False, indent=4)