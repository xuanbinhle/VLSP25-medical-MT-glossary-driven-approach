from huggingface_hub import login
from unsloth import FastLanguageModel
import torch

if __name__ == '__main__':
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/qwen3_1.7B_ood_unsloth/checkpoint-9995",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    your_token = "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"
    login(token=your_token)
    
    model.push_to_hub("Tomaaaa/unsloth_qwen3_1.7B_ood")
    tokenizer.push_to_hub("Tomaaaa/unsloth_qwen3_1.7B_ood")