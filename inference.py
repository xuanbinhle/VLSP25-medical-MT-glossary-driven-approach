from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams
import torch
from datasets import load_dataset, concatenate_datasets
import time
from icecream import ic
from tqdm import tqdm
import re
import os
from create_test_prompt import create_prompts
import pandas as pd
import json

def extract_output(text: str) -> str:
    pattern = r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
        cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL).strip()
    else:
        cleaned_text = text.strip().split("\n")[-1]
    cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text, flags=re.DOTALL).strip()
    return cleaned_text

def main(en_file: str, vi_file: str, output_file: str) -> None:
    model_name = "/workspace/qwen3_1.7B_final_unsloth/checkpoint-11105"  # Finetuning part2
    max_seq_length = 2048  # Choose sequence length
    dtype = None  # Auto detection

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    test_envi = load_dataset("json", data_files="./test/prompts_en2vi.jsonl", split="train") if os.path.exists("./test/prompts_en2vi.jsonl") else create_prompts(en_file)
    prompts_envi = [example["messages"] for example in test_envi]
    test_vien = load_dataset("json", data_files="./test/prompts_vi2en.jsonl", split="train") if os.path.exists("./test/prompts_vi2en.jsonl") else create_prompts(vi_file)
    prompts_vien = [example["messages"] for example in test_vien]

    all_prompts = prompts_envi + prompts_vien
    num_envi_prompts = int(len(all_prompts) / 2)
    
    start_time = time.time()
    results = []
    total_words = 0
    total_sentences = len(all_prompts)
    
    for i in tqdm(range(0, total_sentences), desc="Inference"):
        inputs = tokenizer.apply_chat_template(
            all_prompts[i],
            tokenize=True,
            max_length=max_seq_length,
            add_generation_prompt=True,          
            return_tensors="pt",
        ).to("cuda")
        
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

        response = extract_output(tokenizer.batch_decode(outputs)[0])
        total_words += len(response.split())
        results.append(response)
    
    # with open("results.json", 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    sentences_per_sec = total_sentences / runtime if runtime > 0 else 0
    words_per_sec = total_words / runtime if runtime > 0 else 0
    
    print("\n===== Inference Statistics =====")
    print(f"Total runtime      : {runtime:.2f} seconds")
    print(f"Sentences/sec      : {sentences_per_sec:.2f}")
    print(f"Words/sec          : {words_per_sec:.2f}")
    
    en2vi_outputs = results[:num_envi_prompts]
    vi2en_outputs = results[num_envi_prompts:]
    
    print("\n===== Creating the final DataFrame =====")
    
    results_df = pd.DataFrame({
        'English': vi2en_outputs,
        'Vietnamese': en2vi_outputs
    })
    
    print("\n===== Sample of Results =====")
    print(results_df.head())
    print("=========================")
    
    # Save the final DataFrame to a CSV file
    try:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nDone. CSV file with bidirectional translations saved successfully at: {output_file}")
    except Exception as e:
        print(f"\nError: Could not save the file. Reason: {e}")

if __name__ == '__main__':
    main("./test/en.csv", "./test/vi.csv", "results.csv")