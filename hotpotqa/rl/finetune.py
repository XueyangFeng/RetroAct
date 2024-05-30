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
from dataclasses import dataclass, field
import random
from modeling_policy import LlamaForPolicyLM
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

random.seed(42)
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, rl_data_path: str, regular_data_path: str, reflector_reward_coefficient: float, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.sources, self.targets, self.rewards, self.ref_prob, self.regular_sources, self.regular_targets  = [], [], [], [], [], []
        with open(rl_data_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                input_text = entry['input']
                output_text = entry['output']
                self.sources.append(input_text.strip())

                self.targets.append(input_text + output_text + tokenizer.eos_token)
                if entry['type'] == 'planner':
                    self.rewards.append(entry['reward'])
                elif entry['type'] == 'reflector':
                    reflector_reward = (entry['current_reward'] - entry['last_reward'])*(entry['last_reward']+reflector_reward_coefficient)/reflector_reward_coefficient
                    self.rewards.append(reflector_reward)
                else:
                    raise
                self.ref_prob.append(entry['ref_prob'])

        # Load and process regular data with oversampling
        with open(regular_data_path, 'r') as f:
            data = json.load(f)
            pairs = [(entry['input'].strip(), entry['input'] + entry['output'] + tokenizer.eos_token) for entry in data]

            # Oversample the pairs
            oversampled_pairs = self.oversample(pairs)

            # Unzip the pairs back into sources and targets
            self.regular_sources, self.regular_targets = zip(*oversampled_pairs)

    def oversample(self, data):
        target_length = len(self.sources)
        oversampled_data = random.choices(data, k=target_length)
        return oversampled_data

    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.sources[i], #input部分
            labels=self.targets[i], #全部
            regular_input_ids=self.regular_sources[i],
            ref_prob=self.ref_prob[i],
            regular_labels=self.regular_targets[i],
            rewards=self.rewards[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        print(instances[0]['regular_input_ids'])
        print("=================================================")
        print(instances[0]['regular_labels'])
        print("==================================================================================================")
        """
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

        regular_inputs = self.tokenizer(
            text=[instance['regular_input_ids'] for instance in instances],
            text_target=[instance['regular_labels'] for instance in instances],
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        regular_labels = copy.deepcopy(regular_inputs['labels'])
        regular_labels[regular_labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        regular_labels[torch.where(regular_inputs['input_ids'] != self.tokenizer.pad_token_id)] = IGNORE_INDEX

        regular_inputs['input_ids'] = regular_inputs['labels']
        regular_inputs['labels'] = regular_labels
        
        inputs['regular_input_ids'] = regular_inputs['input_ids']
        inputs['regular_labels'] = regular_inputs['labels']
        inputs['regular_attn_mask'] = regular_inputs['attention_mask']


        rewards = []
        ref_probs = []

        for idx, instance in enumerate(instances):
            instance_input_len = (inputs['input_ids'][idx] != self.tokenizer.pad_token_id).sum() # len all
            instance_label_len = (inputs['labels'][idx] != IGNORE_INDEX).sum() #len output 
            # print(instance['input_ids'])
            # print(instance['labels'])
            # print(instance_input_len)
            # print(instance_label_len); raise

            # Adjust reward and ref_prob per instance
            instance_reward = [0] * (instance_input_len - instance_label_len) + [instance['rewards']] * instance_label_len
            instance_ref_prob = [0] * (instance_input_len - instance_label_len) + instance['ref_prob'][:instance_label_len]

            rewards.append(instance_reward)
            ref_probs.append(instance_ref_prob)

            # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.bfloat16)
        ref_probs_tensor = torch.tensor(ref_probs, dtype=torch.bfloat16)

        # Add to inputs
        inputs['rewards'] = rewards_tensor
        inputs['ref_prob'] = ref_probs_tensor
        return inputs



def train(
    # model/data params
    base_model: str = "",
    rl_data_path: str = "",
    regular_data_path: str = "",
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
    #RL hyperparams
    regular_coefficient: float = 1.0,
    reflector_reward_coefficient: float = 1.0,
    clip_advantage: bool = False,
    clip_episode: float = 0.3,


    # other
    model_parallel = False,
    prompt_template_name: str = "vanilla",
):
    print(
        f"Training LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"rl_data_path: {rl_data_path}\n"
        f"regular_data_path: {regular_data_path}\n"
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
        f"clip_episode: {clip_episode}\n"
        f"clip_advantage: {clip_advantage}\n"
        f"reflector_reward_coefficient: {reflector_reward_coefficient}\n"
        f"regular_coefficient: {regular_coefficient}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    output_dir = output_path + f"/ep{num_epochs}_" + f"lr{learning_rate}_" + f"episode{clip_episode}_" + f"rrc{reflector_reward_coefficient}_" + f"rc{regular_coefficient}"



    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    model = LlamaForPolicyLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        clip_advantage=clip_advantage,
        clip_episode=clip_episode,
        regular_coefficient=regular_coefficient
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



    dataset = SupervisedDataset(rl_data_path=rl_data_path, regular_data_path=regular_data_path, tokenizer=tokenizer, reflector_reward_coefficient=reflector_reward_coefficient)
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