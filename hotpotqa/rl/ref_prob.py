import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import random
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import jsonlines
#set your cuda devices
#os.environ['CUDA_VISIBLE_DEVICES']="0"
def get_prob(input_text, output_text, model, tokenizer):
    with torch.no_grad():
        label_ids = tokenizer(output_text + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        input_ids = tokenizer(input_text + output_text + tokenizer.eos_token)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0).to('cuda')
        outputs = model(input_ids)
        logits = outputs.logits
        labels = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        CEFunc = CrossEntropyLoss(reduction='none')
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        log_prob = -CEFunc(shift_logits, shift_labels)
        log_prob = log_prob[-len(label_ids):]

        result = log_prob

        return result.tolist()


model_path = "your base model path"
lora_path = "lora path"
input_file_path = "input file"
output_file_path = "output file with log prob for each token"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

model = PeftModel.from_pretrained(model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

with open(input_file_path, 'r') as input_file:
    data = json.load(input_file)

# 使用 tqdm 包装循环
with tqdm(total=len(data), desc="Processing entries") as pbar:
    for entry in data:
        entry['ref_prob'] = get_prob(input_text=entry['input'], output_text=entry['output'], model=model, tokenizer=tokenizer)
        pbar.update(1)

with open(output_file_path, 'w') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)
        