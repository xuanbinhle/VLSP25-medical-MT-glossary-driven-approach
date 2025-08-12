# Open file & Process
import re
import json

VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
ALLOW_PUNC = r"\.,:;\-!\"\'%+^/=“”"
EN_CHARS = r"a-zA-Z0-9"
VI_CHARS = rf"a-zA-Z0-9{VN_CHARS}"
EN_PATTERN = rf"""
    [{EN_CHARS}{ALLOW_PUNC} ]+           
    (?:\s*[\(\[]\s*[{EN_CHARS}{ALLOW_PUNC} ]*\s*[\)\]]?)?
"""
VI_PATTERN = rf"""
    [{VI_CHARS}{ALLOW_PUNC} ]+           
    (?:\s*[\(\[]\s*[{VI_CHARS}{ALLOW_PUNC} ]*\s*[\)\]]?)?
"""

FINAL_DICTIONARY = {}

def strip_trailing_punctuation(word: str) -> str:
    TRAILING_PUNC = r"[.,:;!\?\"\'…%+^/-]+"
    return re.sub(rf"{TRAILING_PUNC}$", "", word.strip()).strip()

def process_matching_terms(en_term, vi_term) -> None:
    en_terms = re.split(r"(?:\s+or\s+|\s+=\s+|\s*,\s*|\s*/\s*)", en_term.strip().lower())
    en_terms = [re.sub(r'and\s+', "", en).strip() for en in en_terms]
    vi_terms = re.split(r"(?:\s+hoặc\s+|\s+=\s+|\s*,\s*|\s*/\s*)", vi_term.strip().lower())
    vi_terms = [re.sub(r'^và\s+', "", vi).strip() for vi in vi_terms]
    for en in en_terms:
        FINAL_DICTIONARY.setdefault(en, [])
        for vi in vi_terms:
            if en != "" and vi != "":
                if vi not in FINAL_DICTIONARY[en]:
                    vi = re.sub(r"\(như nhau\)", "", vi).strip()
                    FINAL_DICTIONARY[en].append(vi)


def is_valid_term_pair(en_term, vi_term):
    return re.fullmatch(EN_PATTERN, en_term, re.VERBOSE) and \
           re.fullmatch(VI_PATTERN, vi_term, re.VERBOSE)

if __name__ == '__main__':
    # with open("/content/drive/MyDrive/[VLSP] MT in Medical Domain/Other Dataset/EN-VI Medical Dictionary.txt", encoding='utf-8') as dic_vi_en:
    #     en_temp, vi_temp = "", ""
    #     prev_key, prev_value = "", ""
    #     pos_error = []
    #     one_line, eng_word = False, False
    #     for idx, line in enumerate(dic_vi_en):
    #         if idx < 5:
    #             pos_error.append(False)
    #             continue
    #         elif not bool(re.search(r"\w+", line)):
    #             one_line = True
    #             continue
    #         try:
    #             if one_line:
    #                 if bool(re.search(r"\w+", line)):
    #                     if not eng_word:
    #                         en_temp = line.strip()
    #                         eng_word = True
    #                     else:
    #                         vi_temp = line.strip()
    #                         eng_word = False
    #                 if en_temp != "" and vi_temp != "":
    #                     if en_temp not in FINAL_DICTIONARY:
    #                         FINAL_DICTIONARY[en_temp] = []
    #                     FINAL_DICTIONARY[en_temp].append(vi_temp)
    #                     en_temp, vi_temp = "", ""

    #             else:
    #                 extracted_terms = re.split(r"\t|\s{3,}|(?=\s+soi vòm)", line)
    #                 cleaned_terms = list(filter(lambda x: x != "", map(strip_trailing_punctuation, extracted_terms)))
    #                 if len(cleaned_terms) == len(extracted_terms):
    #                     if len(cleaned_terms) == 1:
    #                         pos_error.append(True)
    #                         en_temp += cleaned_terms[0] + " "
    #                     else:
    #                         if pos_error[idx - 1]:
    #                             pos_error.append(True)
    #                             en_temp += cleaned_terms[0]
    #                             vi_temp += cleaned_terms[1] + " "
    #                         else:
    #                             pos_error.append(False)
    #                             vi_temp, en_temp = "", "" # Reset khi đã gặp 1 pair mới
    #                             cleaned_terms[0] = cleaned_terms[0].replace("(person in charge of)", "").strip()
    #                             if is_valid_term_pair(cleaned_terms[0], cleaned_terms[1]) is not None:
    #                                 print("Chuẩn CMNR!!!")
    #                                 process_matching_terms(cleaned_terms[0], cleaned_terms[1])
    #                                 prev_key, prev_value = cleaned_terms[0], cleaned_terms[1]
    #                             else:
    #                                 print(f"Previous: {prev_key}, {prev_value}")
    #                                 print(f"Now: {cleaned_terms[0]}, {cleaned_terms[1]}")
    #                                 new_key = prev_key + " " + cleaned_terms[0]
    #                                 new_value = prev_value + " " + cleaned_terms[1]
    #                                 del FINAL_DICTIONARY[prev_key.lower()] # Xóa cặp key-value cũ để thêm vào key-value mới
    #                                 process_matching_terms(new_key, new_value)
    #                 else:
    #                     if pos_error[idx - 1]:
    #                         vi_temp += cleaned_terms[0] + " " 
    #                         pos_error.append(True)
    #                     elif sum(pos_error) % 3 == 0:
    #                         if len(cleaned_terms[0]) < 15:
    #                             process_matching_terms(prev_key, prev_value + " " + cleaned_terms[0])
    #                             pos_error.append(False)
    #                         else:
    #                             vi_temp += cleaned_terms[0] + " "
    #                             pos_error.append(True)
    #                     else:
    #                         pos_error.append(False)

    #                 if pos_error[idx] and pos_error[idx - 1] and pos_error[idx - 2]:
    #                     process_matching_terms(en_temp, vi_temp)
    #                     prev_key, prev_value = en_temp, vi_temp
    #                     pos_error[idx - 2 : idx + 1] = [False, False, False]
    #                     en_temp, vi_temp = "", ""
    #         except Exception as err:
    #             raise ValueError(f"Line {idx}: Solving Error Format EN-VI Terms in Dictionary with {err}")

    # for en, vi_list in FINAL_DICTIONARY.items():
    #     print(f"{en} → {'; '.join(vi_list)}")

    # with open('/content/drive/MyDrive/[VLSP] MT in Medical Domain/Other Dataset/EN-VI Medical Dictionary.json', 'w', encoding='utf-8') as f:
    #     json.dump(dict(FINAL_DICTIONARY), f, ensure_ascii=False, indent=4)
    
    
    # Combine Dictionary of Terminology & VinUni
    dict_MedicalTerm = json.load(open('./OtherDataset/EN-VI_Medical_Dictionary.json', 'r', encoding='utf-8'))
    medicalVinUni = json.load(open('./OtherDataset/VinUni_Medical_Dictionary.json', 'r', encoding='utf-8'))
    dict_MedicalVinUni = {}
    for item in medicalVinUni:
        if not isinstance(item['vn'], str) or not isinstance(item['en'], str):
            print(ValueError(f"VN-Term is not suitable data type: {item['vn']}"))
            continue
        en_term = item['en'].lower().strip()
        if en_term in dict_MedicalVinUni and dict_MedicalVinUni[en_term][-1] == item['vn'].lower().strip():
            print(ValueError(f"EN-Term is duplicate with {en_term}"))
            continue
        dict_MedicalVinUni[en_term] = [item['vn'].lower().strip()]
    
    results = {}
    dicts = [dict_MedicalTerm, dict_MedicalVinUni]
    for d in dicts:
        for en_term, list_vi_terms in d.items():
            results.setdefault(en_term, []).extend(list_vi_terms)
    
    with open('./OtherDataset/Final_Medical_Dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)