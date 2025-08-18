import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path

# --- Config ---
TOP_K = 3
SIM_THRESHOLD = 0.6

# --- Load Medical Dictionary ---
def load_dictionary(json_path):
    with open(json_path, 'r') as f:
        raw_dict = json.load(f)

    rows = []
    for en, vi_list in raw_dict.items():
        for vi in (vi_list if isinstance(vi_list, list) else [vi_list]):
            rows.append({"English Term": en.strip(), "Vietnamese Term": vi.strip()})
    return pd.DataFrame(rows)

# --- Embed Dictionary Terms ---
def build_embeddings(df_dict, model_sbert):
    terms = df_dict["English Term"].tolist()
    embeddings = model_sbert.encode(terms, convert_to_tensor=True)
    return terms, embeddings

# --- Find Similar Terms ---
def get_similar_terms(sentence, model_sbert, en_terms, en_embeddings, df_dict, top_k=3, threshold=0.6):
    emb = model_sbert.encode(sentence, convert_to_tensor=True)
    cos_scores = util.cos_sim(emb, en_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    terms = []
    for idx, score in zip(top_results.indices, top_results.values):
        if score >= threshold:
            en = en_terms[idx]
            vi = df_dict[df_dict["English Term"] == en]["Vietnamese Term"].values[0]
            terms.append((en, vi))
    return terms

# --- Prompt Builder ---
def highlight_terms_in_sentence(sentence, terms):
    """
    Wraps each matched term in <term>...</term> in the source sentence.
    """
    for term, _ in sorted(terms, key=lambda x: -len(x[0])):  # longer first to avoid overlap
        sentence = sentence.replace(term, f"<term>{term}</term>")
    return sentence

def build_prompt(src_text, lang, similar_terms):
    if not similar_terms:
        ref_block = ""
        dedup_terms = []
    else:
        # De-duplicate by target to avoid repeated mappings
        if lang == "en":  # EN→VI
            seen_vi = set()
            dedup_terms = [(en, vi) for en, vi in similar_terms if not (vi in seen_vi or seen_vi.add(vi))]
        else:  # VI→EN
            seen_en = set()
            dedup_terms = [(en, vi) for en, vi in similar_terms if not (en in seen_en or seen_en.add(en))]

        # Create dictionary hint block
        if lang == "en":
            ref_block = "Refer to these medical terms for accuracy:\n" + \
                        "\n".join([f'- "{en}" → <target>"{vi}"</target>' for en, vi in dedup_terms]) + "\n"
        else:
            ref_block = "Refer to these medical terms for accuracy:\n" + \
                        "\n".join([f'- "{vi}" → <target>"{en}"</target>' for en, vi in dedup_terms]) + "\n"

    # Highlight terms in the sentence
    if lang == "en":
        # Highlight English terms in English sentence
        highlighted_sentence = highlight_terms_in_sentence(src_text, dedup_terms)
        role = "You are a professional translator."
        header = "Translate the following English sentence into Vietnamese."
    else:  # lang == "vi"
        # Highlight Vietnamese terms in Vietnamese sentence
        highlighted_sentence = highlight_terms_in_sentence(src_text, [(vi, en) for en, vi in dedup_terms])
        role = "You are a professional translator."
        header = "Translate the following Vietnamese sentence into English."

    return f"""{role}
{header}
{ref_block}Sentence: "{highlighted_sentence}"
""".strip()

def build_chat_messages(src_text, tgt_text, lang, similar_terms):
    has_terms = bool(similar_terms)

    if has_terms:
        # De-duplicate by target
        if lang == "en":
            seen_vi = set()
            dedup_terms = [(en, vi) for en, vi in similar_terms if not (vi in seen_vi or seen_vi.add(vi))]
            ref_block = "Refer to these medical terms for accuracy:\n" + \
                        "\n".join([f'- "{en}" → <target>"{vi}"</target>' for en, vi in dedup_terms]) + "\n"
        else:
            seen_en = set()
            dedup_terms = [(en, vi) for en, vi in similar_terms if not (en in seen_en or seen_en.add(en))]
            ref_block = "Refer to these medical terms for accuracy:\n" + \
                        "\n".join([f'- "{vi}" → <target>"{en}"</target>' for en, vi in dedup_terms]) + "\n"
    else:
        dedup_terms = []
        ref_block = ""

    # Highlight terms in sentence
    if lang == "en":
        highlighted_sentence = highlight_terms_in_sentence(src_text, dedup_terms)
        header = "Translate the following English sentence into Vietnamese."
    else:
        highlighted_sentence = highlight_terms_in_sentence(src_text, [(vi, en) for en, vi in dedup_terms])
        header = "Translate the following Vietnamese sentence into English."

    # Add flag only if there are medical terms
    flag_line = "has_medical_terms: true\n" if has_terms else ""

    user_msg = f"""{flag_line}You are a professional translator.
{header}
{ref_block}Sentence: "{highlighted_sentence}"
""".strip()

    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": tgt_text.strip()}
    ]

# --- Process Dataset from TXT ---
def process_txt_dataset(src_txt_path, tgt_txt_path, output_jsonl_path, lang, df_dict, model_sbert, en_terms, en_embeddings):
    with open(src_txt_path, "r", encoding="utf-8") as f_src, open(tgt_txt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    assert len(src_lines) == len(tgt_lines), "Mismatch in line count between source and target files."

    results = []
    for src, tgt in zip(src_lines, tgt_lines):
        src = src.strip()
        tgt = tgt.strip()
        if not src or not tgt:
            continue

        similar_terms = get_similar_terms(src, model_sbert, en_terms, en_embeddings, df_dict, TOP_K, SIM_THRESHOLD)
        chat_messages = build_chat_messages(src, tgt, lang, similar_terms)

        results.append({
            "messages": chat_messages
        })

    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for r in results:
            json.dump(r, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Saved {len(results)} chat-format examples to: {output_jsonl_path}")

def load_sbert_model(lang):
    if lang == "vi":
        return SentenceTransformer("thang1943/vietnamese-sbert-v2")  
    else:
        return SentenceTransformer("all-MiniLM-L6-v2")


# --- Main ---
def main():
    dict_path = "Medical_Dictionary.json"
    df_dict = load_dictionary(dict_path)

    en_data = '/content/drive/MyDrive/[VLSP] MT in Medical Domain/data/Final/test.en.txt'
    vi_data = '/content/drive/MyDrive/[VLSP] MT in Medical Domain/data/Final/test.vi.txt'

    # --- EN → VI ---
    model_sbert_en = load_sbert_model("en")
    en_terms, en_embeddings = build_embeddings(df_dict, model_sbert_en)
    # process_txt_dataset("out_of_domain.en.txt", "out_of_domain.vi.txt", "prompts_en2vi_outdomain.jsonl", "en", df_dict, model_sbert_en, en_terms, en_embeddings)
    # process_txt_dataset("in_domain.en.txt", "in_domain.vi.txt", "prompts_en2vi_indomain.jsonl", "en", df_dict, model_sbert_en, en_terms, en_embeddings)
    process_txt_dataset(en_data, vi_data, "improved_prompts_en2vi.jsonl", "en", df_dict, model_sbert_en, en_terms, en_embeddings)
    # --- VI → EN ---
    model_sbert_vi = load_sbert_model("vi")
    en_terms_vi, en_embeddings_vi = build_embeddings(df_dict, model_sbert_vi)
    # process_txt_dataset("out_of_domain_vi.vi.txt", "out_of_domain_vi.en.txt", "prompts_vi2en_outdomain.jsonl", "vi", df_dict, model_sbert_vi, en_terms_vi, en_embeddings_vi)
    # process_txt_dataset("in_domain_vi.vi.txt", "in_domain_vi.en.txt", "prompts_vi2en_indomain.jsonl", "vi", df_dict, model_sbert_vi, en_terms_vi, en_embeddings_vi)
    process_txt_dataset(vi_data, en_data, "improved_prompts_vi2en.jsonl", "vi", df_dict, model_sbert_vi, en_terms_vi, en_embeddings_vi)

if __name__ == "__main__":
    main()