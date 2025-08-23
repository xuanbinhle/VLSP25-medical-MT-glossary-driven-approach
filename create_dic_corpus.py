"""
Xây dựng Dictionary of Corpus (Noun_Phrase En -> Vi) sử dụng pretrained EnViT5 
Notes: USE - Không bị phụ thuộc Pos Tagging Tiếng Việt.
"""


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import spacy
import pandas as pd
import itertools
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

class Dictionary_Augmentation(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def extract_en_noun_phrases(self, sentence: str) -> list[str]:
        doc_sen = NLP(sentence)
        phrases = []
        for chunk in doc_sen.noun_chunks:
            if not any(token.pos_ == 'NUM' for token in chunk):
                phrases.append(chunk.text.strip())
        return phrases

    def forward(self, en_ind_corpus: list) -> dict:
        final_mapping = {}
        for idx, S in tqdm(enumerate(en_ind_corpus), desc='Generate Dictionary Terms', total=len(en_ind_corpus)):
            phrases_Sk = self.extract_en_noun_phrases(S)
            if not phrases_Sk:
                continue
            new_phrases_Sk = [f"en: {phrase}" for phrase in phrases_Sk]
            inputs = self.tokenizer(new_phrases_Sk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            encoded = self.model.generate(**inputs, max_length=512)
            outputs = self.tokenizer.batch_decode(encoded, skip_special_tokens=True)
            cleaned_outputs = [vi[4:] for vi in outputs]

            for en_phrase, vi_phrase in zip(phrases_Sk, cleaned_outputs):
                if en_phrase not in final_mapping:
                    final_mapping[en_phrase] = set()
                final_mapping[en_phrase].add(vi_phrase)

            if (idx + 1) % 500 == 0:
                print(f"Finish Translate Terms with {idx+1} sentences!!!")
                to_write = {k: list(v) for k, v in final_mapping.items()}
                with open("appendix_dic_end.json", 'w', encoding='utf-8') as f:
                    json.dump(to_write, f, ensure_ascii=False, indent=4)

        return {k: list(v) for k, v in final_mapping.items()}

if __name__ == '__main__':
    en_ind_corpus = open("./Final/train.en.txt", 'r', encoding='utf-8')
    en_list = []
    for en in en_ind_corpus:
        en_list.append(en.strip())
    aug_dic_method = Dictionary_Augmentation("VietAI/envit5-translation")
    final_mapping = aug_dic_method(en_list)