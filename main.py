from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from icecream import ic

def check_vietnamese_text(text: str) -> bool:
    VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
    VI_PATTERN = rf"[{VN_CHARS}]"
    return bool(re.search(VI_PATTERN, text))

PROMPT_EN2VI = """
You are a professional medical translator. 
Translate the following {src_lang} medical text into natural and accurate {tgt_lang}, ensuring medical terms are translated precisely and consistently:

{input_text}
"""

PROMPT_VI2EN = """
You are a professional medical translator. 
Translate the following {src_lang} medical text into natural and accurate {tgt_lang}, ensuring medical terms are translated precisely and consistently.

{input_text}
"""

def create_dataset(en_file, vi_file):
    en_texts = open(en_file, 'r', encoding='utf-8')
    vi_texts = open(vi_file, 'r', encoding='utf-8')
    dataset = []
    for en_text, vi_text in zip(en_texts, vi_texts):
        en2vi_item = {
            "src": en_text.strip(),
            "src_lang": "English",
            "tgt": vi_text.strip(),
            "tgt_lang": "Vietnamese"
        }
        vi2en_item = {
            "src": vi_text.strip(),
            "src_lang": "Vietnamese",
            "tgt": en_text.strip(),
            "tgt_lang": "English"
        }
        dataset.append(en2vi_item)
        dataset.append(vi2en_item)
    return dataset
    

def convert_to_conversation(sample):
    src_sen = sample['src']
    src_lang = sample['src_lang']
    tgt_sen = sample['tgt']
    tgt_lang = sample['tgt_lang']
    if src_lang == "Vietnamese":
        content_input = PROMPT_VI2EN.format(src_lang=src_lang, tgt_lang=tgt_lang, input_text=src_sen)
        content_output = tgt_sen
    else:
        content_input = PROMPT_EN2VI.format(src_lang=src_lang, tgt_lang=tgt_lang, input_text=src_sen)
        content_output = src_sen
    
    conversation = [
        {
            "role": "user",
            "content": content_input
        },
        {
            "role": "assistant",
            "content": content_output
        }
    ]
    return { "messages": conversation }

if __name__ == '__main__':
    model_name = "./Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    dataset = create_dataset(en_file="./data/Final/train.en.txt", vi_file="./data/Final/train.vi.txt")
    conversations = [convert_to_conversation(sample) for sample in dataset]
    
    input_text = tokenizer.apply_chat_template(
        conversations[0]['messages'],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
        
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)