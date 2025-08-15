from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
from tqdm import tqdm

MODEL_EN2VI = "vinai/vinai-translate-en2vi-v2"
BATCH_SIZE = 32
MILESTONE = 1273 * BATCH_SIZE

class BackTranslation:
    def __init__(self, model_en2vi: str):
        self.tokenizer_en2vi = AutoTokenizer.from_pretrained(model_en2vi, src_lang="en_XX")
        self.device = torch.device('cuda')
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model_en2vi).to(self.device)

    def translate_en2vi(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer_en2vi(texts, padding=True, return_tensors="pt").to(self.device)
        outputs = self.model_en2vi.generate(**inputs, decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"], num_beams=5, early_stopping=True)
        return self.tokenizer_en2vi.batch_decode(outputs, skip_special_tokens=True)

if __name__ == '__main__':
    en_ood = open("/content/[VLSP] MT in Medical Domain/data/Final/train.en.txt", 'r', encoding='utf-8')
    vi_ood = open("/content/[VLSP] MT in Medical Domain/data/Final/train.vi.txt", 'r', encoding='utf-8')
    augment_backtranslation_en = open("/content/drive/MyDrive/backtranslation.en.txt", 'w', encoding='utf-8')
    augment_backtranslation_vi = open("/content/drive/MyDrive/backtranslation.vi.txt", 'w', encoding='utf-8')
    seen_pairs = set()
    en_list, vi_list = [], []
    for en, vi in zip(en_ood, vi_ood):
        seen_pairs.add((en.strip(), vi.strip()))
        en_list.append(en.strip())
        vi_list.append(vi.strip())

    method_backtranslation = BackTranslation(MODEL_EN2VI)
    for i in tqdm(range(MILESTONE, len(en_list), BATCH_SIZE), desc='BackTranslation'):
        en_sentences = [en for en in en_list[i:i+BATCH_SIZE]]
        vi_translated = method_backtranslation.translate_en2vi(en_sentences)
        for org_en, trans_vi in zip(en_sentences, vi_translated):
            pair = (org_en, trans_vi.lower().strip())
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                augment_backtranslation_en.write(f"{en}\n")
                augment_backtranslation_vi.write(f"{vi}\n")
                augment_backtranslation_en.flush()
                augment_backtranslation_vi.flush()