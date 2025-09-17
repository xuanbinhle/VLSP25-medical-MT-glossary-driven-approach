# Methodology
This is the overview workflow of our method proposed for VLSP2025 shared task: *Medical Machine Translation with
Limited Parameters and Resources Using Pretrained Models*. 
We leverage terminology reference in the finetuning process using QLoRa technique for two way medical translation: English to Vietnamese and vise versa.

[![Workflow](image.jpg)](image.jpg)

# Code 
- augmentation.py: This code implements the method 'Dictionary-based Data Augmentation' to enhance in-domain data using an out-of-domain dataset and an in-domain dictionary.
- construct_dictionary.py: Construct an in-domain dictionary based on VinUni and EN-VI Medical Terms data in a structured format
- create_dic_corpus.py: Building in-domain dictionary based on corpus by extracting noun-phrase English and translating them by EnViT5 Model
- create_prompt.py: Construct Prompt for fine-tuning model
- create_test_prompt.py: Construct Prompt for inference model
- finetuning.py: Fine-tuning model on out-of-domain dataset
- finetuning_new.py: Fine-tuning model on in-domain dataset
- statistic.py: Cleaning Corpus and Statistic
## Create glossary from corpus

## Make prompt for supervised fine-tuning (SFT)

## Fine tune model

## Inference