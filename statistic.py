"""
Thống kê số lượng records
"""

import json
import re

VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
EN_CHARS = r"a-zA-Z"
VI_CHARS = rf"a-zA-Z{VN_CHARS}"

def check_character_en(en_sen: str) -> bool:
    return bool(re.search(EN_CHARS, en_sen))

def check_character_vi(vi_sen: str) -> bool:
    return bool(re.search(VI_CHARS, vi_sen))

if __name__ == '__main__':
    jsonDic = json.load(open("./data/Final/Medical_Dictionary.json", 'r', encoding='utf-8'))
    print(f"Size of Dictionary: {len(jsonDic)}")
    
    f_pairs = open("./data/OtherDataset/MedEV.vi-en.pairs.txt", 'r', encoding='utf-8')
    pairs_MedEV = set()
    sizeMedEV = 0
    indices_MedEV = []
    for idx, line in enumerate(f_pairs):
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError("Bug Formmat in MedEV Dataset")
        vi, en = parts
        vi = vi.strip().lower()
        en = en.strip().lower()
        if check_character_vi(vi) or check_character_en(en):
            continue
        sizeMedEV += 1
        if (en, vi) in pairs_MedEV:
            continue
        indices_MedEV.append(idx)
        pairs_MedEV.add((en, vi))
    print(f"Size of Original MedEV Dataset: {sizeMedEV}")
    
    f_en, f_vi = open("./data/Corpus_VLSP2025/train.en.txt", 'r', encoding='utf-8'), open("./data/Corpus_VLSP2025/train.vi.txt", 'r', encoding='utf-8')
    pairs_CorpusTraining = set()
    sizeTraining = 0
    for en_line, vi_line in zip(f_en, f_vi):
        if not isinstance(en_line.strip(), str) or not isinstance(vi_line.strip(), str) or check_character_en(en_line.strip()) or check_character_vi(vi_line.strip()):
            continue
        vi = vi_line.strip().lower()
        en = en_line.strip().lower()
        sizeTraining += 1
        if (en, vi) in pairs_CorpusTraining:
            continue
        pairs_CorpusTraining.add((en, vi))
    print(f"Size of Original Corpus Dataset Training: {sizeTraining}")
    
    f_en, f_vi = open("./data/Corpus_VLSP2025/test.en.txt", 'r', encoding='utf-8'), open("./data/Corpus_VLSP2025/test.vi.txt", 'r', encoding='utf-8')
    sizeTesting = 0
    pairs_CorpusTesting = set()
    for en_line, vi_line in zip(f_en, f_vi):
        if not isinstance(en_line.strip(), str) or not isinstance(vi_line.strip(), str) or check_character_en(en_line.strip()) or check_character_vi(vi_line.strip()):
            continue
        sizeTesting += 1
        vi = vi_line.strip().lower()
        en = en_line.strip().lower()
        if (en, vi) in pairs_CorpusTesting:
            continue
        pairs_CorpusTesting.add((en, vi))
    print(f"Size of Original Corpus Dataset Testing: {sizeTesting}")
    
    # Cleaning pairs
    cleaned_pairs_MedEV = pairs_MedEV - pairs_CorpusTesting
    cleaned_pairs_CorpusTraining = pairs_CorpusTraining - pairs_CorpusTesting
    assert len(cleaned_pairs_CorpusTraining & pairs_CorpusTesting) == 0 and len(cleaned_pairs_MedEV & pairs_CorpusTesting) == 0, "Testing-Pairs exist in Training-Pairs & MedEV Dataset"
    
    print(f"Size of Cleaned MedEV Dataset: {len(cleaned_pairs_MedEV)}")
    print(f"Size of Cleaned Corpus Training: {len(cleaned_pairs_CorpusTraining)}")
    print(f"Size of Cleaned Corpus Testing: {len(pairs_CorpusTesting)}")
    
    cleaned_Training = cleaned_pairs_CorpusTraining | cleaned_pairs_MedEV
    print(f"Final Training Dataset: {len(cleaned_Training)}")
    print(f"Final Testing Dataset: {len(pairs_CorpusTesting)}")
    
    f_training_en = open("./data/Final/train.en.txt", 'w', encoding='utf-8')
    f_training_vi = open("./data/Final/train.vi.txt", 'w', encoding='utf-8')
    for (en, vi) in cleaned_Training:
        f_training_en.write(f"{en}\n")
        f_training_vi.write(f"{vi}\n")
    
    f_testing_en = open("./data/Final/test.en.txt", 'w', encoding='utf-8')
    f_testing_vi = open("./data/Final/test.vi.txt", 'w', encoding='utf-8')
    for (en, vi) in pairs_CorpusTesting:
        f_testing_en.write(f"{en}\n")
        f_testing_vi.write(f"{vi}\n")