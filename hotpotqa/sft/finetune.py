from dataclasses import dataclass, field

import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import fire
from typing import List
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from utils.prompter import Prompter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import(
    LoraConfig,
    get_peft_model,
)
import copy
from typing import Optional, Dict, Sequence
import logging
from torch.utils.data import Dataset
import json
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,  tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.sources, self.targets = [], []
        for path in data_path.split(','):
            with open(path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    input_text = entry['input']
                    self.sources.append(input_text.strip())
                    output_text = entry['output']
                    self.targets.append(input_text + output_text + tokenizer.eos_token)

    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.sources[i], #input部分
            labels=self.targets[i], #全部
        )

 
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #label就是mask后的input_id
        inputs = self.tokenizer(
            text=[instance['input_ids'] for instance in instances],
            text_target=[instance['labels'] for instance in instances],
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        ) #input_id(最终的input_id应该是inputs['label'])
        
        labels = copy.deepcopy(inputs['labels'])
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        labels[torch.where(inputs['input_ids'] != self.tokenizer.pad_token_id)] = IGNORE_INDEX

        inputs['input_ids'] = inputs['labels']
        inputs['labels'] = labels
        return inputs

def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_path: str = "./lora_alpaca",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 3e-5,
    cutoff_len: int = 2048,
    val_set_size: int = 0,
    
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj"
    ],
    # trainer params
    train_on_inputs: bool = False,
    group_by_length: bool = True,
    # other
    model_parallel = False,
    prompt_template_name: str = "vanilla",
):
    print(
        f"Training LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_path: {output_path}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"model_parallel: {model_parallel}\n"
        f"prompt_template_name: {prompt_template_name}\n"
        f"device count: {torch.cuda.device_count()}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    output_dir = output_path + f"/ep{num_epochs}_" + f"lr{learning_rate}"



    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    prompter = Prompter(prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=cutoff_len,
        padding_side="left",
    )

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    assert num_new_tokens == 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = DEFAULT_UNK_TOKEN



    dataset = SupervisedDataset(data_path=data_path, tokenizer=tokenizer)
    datacollator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="epoch",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=datacollator
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__=="__main__":
    fire.Fire(train)