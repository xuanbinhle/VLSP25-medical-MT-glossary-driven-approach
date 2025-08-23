from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments


def apply_chat_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,              # chỉ lấy string
        add_generation_prompt=False  # để training (không thêm "assistant" trống ở cuối)
    )
    return {"text": text.strip()}

if __name__ == '__main__':
    model_name = "./unsloth_qwen3_1.7B_ood"
    max_seq_length = 2048  # Choose sequence length
    dtype = None  # Auto detection

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        attn_implementation="flash_attention_2"
    )

    train_ind_envi = load_dataset("json", data_files="./Final/improved_prompts_ind_train_en2vi.jsonl", split="train")
    train_ind_vien = load_dataset("json", data_files="./Final/improved_prompts_ind_train_vi2en.jsonl", split="train")
    test_ind_envi = load_dataset("json", data_files="./Final/improved_prompts_ind_test_en2vi.jsonl", split="train")
    test_ind_vien = load_dataset("json", data_files="./Final/improved_prompts_ind_test_vi2en.jsonl", split="train")
    
    train_ind_envi_dataset = train_ind_envi.map(apply_chat_template, remove_columns=["messages"])
    train_ind_vien_dataset = train_ind_vien.map(apply_chat_template, remove_columns=["messages"])
    test_ind_envi_dataset = test_ind_envi.map(apply_chat_template, remove_columns=["messages"])
    test_ind_vien_dataset = test_ind_vien.map(apply_chat_template, remove_columns=["messages"])
    
    train_dataset = concatenate_datasets([train_ind_envi_dataset, train_ind_vien_dataset])
    test_dataset = concatenate_datasets([test_ind_envi_dataset, test_ind_vien_dataset])
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 64,
            gradient_accumulation_steps = 1,
            per_device_eval_batch_size = 128,
            warmup_steps = 10,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            save_steps=500,
            seed = 3407,
            output_dir="qwen3_1.7B_final_unsloth",
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_pin_memory=False,
            report_to="none"
        ),
    )
    
    # Train the model
    trainer_stats = trainer.train()