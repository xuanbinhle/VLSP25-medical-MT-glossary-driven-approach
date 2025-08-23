import json
import re
import pandas as pd
import os
import torch
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

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
    device = torch.device("cuda")
    path = "./vietnamese-sbert-v2" if lang == "vi" else "./all-MiniLM-L6-v2"
    return SentenceTransformer(path).to(device)

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

# ========== Reference Block ==========
def format_ref_block(pairs: List[Tuple[str, List[str]]], src_label: str) -> str:
    if not pairs:
        return ""
    lines = []
    for src_term, tgt_terms in pairs:
        quoted_tgts = '; '.join([f'"{t}"' for t in tgt_terms])
        lines.append(f'   + "**{src_term}**" → **{quoted_tgts}**')
    return "**Refer to these medical terms for consistency**:\n" + "\n".join(lines) + "\n"


# ========== Prompt Builder ==========
def build_chat_messages(src: str, lang: str, matched: List[Tuple[str, float]],
                        en2vi: Dict[str, List[str]], vi2en: Dict[str, List[str]]) -> List[Dict[str, str]]:
    if lang == "en":
        header = "Translate the following English text into natural and accurate Vietnamese."
        dedup = [(en, en2vi.get(en, [])) for en, _ in matched] if matched else []
        ref_block = format_ref_block(dedup, "en")
    else:
        header = "Translate the following Vietnamese text into natural and accurate English."
        dedup = [(vi, vi2en.get(vi, [])) for vi, _ in matched] if matched else []
        ref_block = format_ref_block(dedup, "vi")

    user_msg = f"""{header}: {src.strip()}
{ref_block}
""".strip()

    return [
        {"role": "system", "content": "You are a professional translator. Translate all texts carefully. Do not change or approximate any numbers, dates, laboratory values, or medication dosages. Keep all measurement units unchanged."},
        {"role": "user", "content": user_msg}
    ]

# ========== Dataset Processor ==========
def process_dataset(src_path: str, output_path: str, lang: str,
                    model: SentenceTransformer, terms: List[str], embeddings: torch.Tensor,
                    en2vi: Dict[str, List[str]], vi2en: Dict[str, List[str]]) -> list:

    df = pd.read_csv(src_path)
    col = df.columns[0]
    results = []
    for i, src_line in tqdm(enumerate(df[col].fillna("")), desc='Creating Message'):
        if not isinstance(src_line, str):
            print(f"Bug line {i} - {src_line}")
            raise
        src = src_line.strip()
        matched = get_similar_terms(src, model, terms, embeddings) if src else []
        messages = build_chat_messages(src, lang, matched, en2vi, vi2en)
        results.append({"messages": messages})

    with open(output_path, "w", encoding="utf-8") as f_out:
        for r in results:
            json.dump(r, f_out, ensure_ascii=False)
            f_out.write("\n")
    print(f"Saved {len(results)} chat-format examples to: {output_path}")
    return results

# ========== Main ==========
def create_prompts(src_data):
    en2vi = load_dict("./Final/final_dic_envi.json")
    vi2en = load_dict("./Final/final_dic_vien.json")

    filename = os.path.basename(src_data)
    if "en" in filename:
        model_en = load_sbert_model("en")
        en_terms, en_emb = build_embeddings(sorted(en2vi.keys()), model_en)
        prompts = process_dataset(src_data, f"./test/prompts_en2vi.jsonl", "en", model_en, en_terms, en_emb, en2vi, vi2en)
    else:
        model_vi = load_sbert_model("vi")
        vi_terms, vi_emb = build_embeddings(sorted(vi2en.keys()), model_vi)
        prompts = process_dataset(src_data, f"./test/prompts_vi2en.jsonl", "vi", model_vi, vi_terms, vi_emb, en2vi, vi2en)
    return prompts