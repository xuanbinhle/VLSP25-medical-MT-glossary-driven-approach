import json
import re
import pandas as pd
import torch
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util

# --- Config ---
TOP_K = 5
SIM_THRESHOLD = 0.6

# ========== Dictionary Utilities ==========
def load_dict(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        k.strip(): sorted(set(v if isinstance(v, list) else [v.strip()]))
        for k, v in raw.items()
        if k.strip()
    }

def invert_dict(input_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    inverted = {}
    for k, values in input_dict.items():
        for v in values:
            inverted.setdefault(v.strip(), set()).add(k.strip())
    return {k: sorted(v) for k, v in inverted.items()}

# ========== Embeddings ==========
def load_sbert_model(lang: str) -> SentenceTransformer:
    return SentenceTransformer("thang1943/vietnamese-sbert-v2") if lang == "vi" else SentenceTransformer("all-MiniLM-L6-v2")

def build_embeddings(terms: List[str], model: SentenceTransformer) -> Tuple[List[str], torch.Tensor]:
    with torch.no_grad():
        embeddings = model.encode(terms, convert_to_tensor=True, normalize_embeddings=True)
    return terms, embeddings

# ========== Similarity Search ==========
def get_similar_terms(sentence: str, model: SentenceTransformer, terms: List[str], embeddings: torch.Tensor) -> List[Tuple[str, float]]:
    if not terms:
        return []
    emb = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(emb, embeddings)[0]
    top_k = min(TOP_K, len(terms))
    top_results = torch.topk(cos_scores, k=top_k)
    return [(terms[i], float(s)) for i, s in zip(top_results.indices, top_results.values) if s >= SIM_THRESHOLD]

# ========== Highlighting ==========
_WORD_BOUNDARY = r"(?<![A-Za-zÀ-ỹ0-9]){term}(?![A-Za-zÀ-ỹ0-9])"

def highlight_terms(sentence: str, terms: List[str]) -> str:
    for term in sorted(set(terms), key=len, reverse=True):
        if not term: continue
        pattern = re.compile(_WORD_BOUNDARY.format(term=re.escape(term)), flags=re.IGNORECASE)
        sentence = pattern.sub(lambda m: f"<term>{m.group(0)}</term>", sentence)
    return sentence

# ========== Reference Block ==========
def format_ref_block(pairs: List[Tuple[str, List[str]]], src_label: str) -> str:
    if not pairs:
        return ""

    lines = []
    for src_term, tgt_terms in pairs:
        quoted_tgts = '; '.join([f'"{t}"' for t in tgt_terms])
        lines.append(f'- "{src_term}" → <target>{quoted_tgts}</target>')

    return "Refer to these medical terms for accuracy:\n" + "\n".join(lines) + "\n"


# ========== Prompt Builder ==========
def build_chat_messages(src: str, tgt: str, lang: str, matched: List[Tuple[str, float]],
                        en2vi: Dict[str, List[str]], vi2en: Dict[str, List[str]]) -> List[Dict[str, str]]:

    if lang == "en":
        dedup = [(en, en2vi.get(en, [])) for en, _ in matched]
        ref_block = format_ref_block(dedup, "en")
        highlight = highlight_terms(src, [en for en, _ in matched])
        header = "Translate the following English sentence into Vietnamese."
    else:
        dedup = [(vi, vi2en.get(vi, [])) for vi, _ in matched]
        ref_block = format_ref_block(dedup, "vi")
        highlight = highlight_terms(src, [vi for vi, _ in matched])
        header = "Translate the following Vietnamese sentence into English."

    user_msg = f"""You are a professional translator.
{header}
{ref_block}Sentence: "{highlight}"
""".strip()

    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": tgt.strip()}
    ]

# ========== Dataset Processor ==========
def process_dataset(src_path: str, tgt_path: str, output_path: str, lang: str,
                    model: SentenceTransformer, terms: List[str], embeddings: torch.Tensor,
                    en2vi: Dict[str, List[str]], vi2en: Dict[str, List[str]]):

    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = [l.strip() for l in f_src]
        tgt_lines = [l.strip() for l in f_tgt]

    assert len(src_lines) == len(tgt_lines), "Mismatch between source and target lines"

    results = []
    for src, tgt in zip(src_lines, tgt_lines):
        if not src or not tgt: continue
        matched = get_similar_terms(src, model, terms, embeddings)
        messages = build_chat_messages(src, tgt, lang, matched, en2vi, vi2en)
        results.append({"messages": messages})

    with open(output_path, "w", encoding="utf-8") as f_out:
        for r in results:
            json.dump(r, f_out, ensure_ascii=False)
            f_out.write("\n")
    print(f"Saved {len(results)} chat-format examples to: {output_path}")

# ========== Main ==========
def main():
    en2vi = load_dict("/content/final_dic_envi.json")
    vi2en = load_dict("/content/final_dic_vien.json")

    en_data = "/content/test.en.txt"
    vi_data = "/content/test.vi.txt"

    # --- EN → VI ---
    model_en = load_sbert_model("en")
    en_terms, en_emb = build_embeddings(sorted(en2vi.keys()), model_en)
    process_dataset(en_data, vi_data, "improved_prompts_en2vi.jsonl", "en", model_en, en_terms, en_emb, en2vi, vi2en)

    # --- VI → EN ---
    # model_vi = load_sbert_model("vi")
    # vi_terms, vi_emb = build_embeddings(sorted(vi2en.keys()), model_vi)
    # process_dataset(vi_data, en_data, "improved_prompts_vi2en.jsonl", "vi", model_vi, vi_terms, vi_emb, en2vi, vi2en)

if __name__ == "__main__":
    main()