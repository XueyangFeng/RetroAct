import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from typing import Dict, Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoTokenizer, LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM
from transformers.file_utils import ModelOutput
from torch.distributions import Categorical
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import json
import random


class LlamaForPolicyLM(LlamaForCausalLM):
    def __init__(self, *args, regular_coefficient=1.0, clip_advantage=False, clip_episode=0.3, **kwargs):
        super().__init__(*args, **kwargs)  
        self.regular_coefficient = regular_coefficient
        self.clip_advantage = clip_advantage
        self.clip_episode = clip_episode


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, regular_coefficient=1.0, clip_advantage=False, clip_episode=0.3,**kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        model.regular_coefficient = regular_coefficient
        model.clip_advantage = clip_advantage
        model.clip_episode = clip_episode

        return model

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        regular_input_ids,
        regular_labels,
        regular_attn_mask,
        rewards,
        ref_prob,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)



        #[batch_size, sequence_length, vocab_size]
        logits = logits.float()

        loss = None
        if labels is not None:
            rewards = rewards[..., 1:].contiguous()
            rewards = rewards.view(-1).contiguous()


            ref_prob = ref_prob[..., 1:].contiguous()
            ref_prob = ref_prob.view(-1).contiguous()
            log_ref_prob = ref_prob
            

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            #[batch_size, sequence_length, vocab_size] -> [batch_size * sequence_length, vocab_size]
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            #[batch_size, sequence_length] -> batch_size * sequence_length
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            log_policy_prob = -loss_fct(shift_logits, shift_labels)

            epsilon= self.clip_episode




            if not self.clip_advantage:
                with torch.no_grad():
                    importance_sampling = torch.exp(log_policy_prob - log_ref_prob)
                importance_sampling_clip = torch.clip(importance_sampling, min = 1 - epsilon, max = 1 + epsilon)
                #importance_sampling = torch.min(importance_sampling * rewards, importance_sampling_clip * rewards)
                loss_ppo = -importance_sampling_clip * log_policy_prob * rewards      
                loss_ppo = loss_ppo.mean()
            """
            else:             
                importance_sampling = torch.exp(log_policy_prob - log_ref_prob)
                importance_sampling_clip = torch.clip(importance_sampling, min = 1 - epsilon, max = 1 + epsilon)
                loss_ppo = -torch.min(importance_sampling_clip* rewards, importance_sampling* rewards)
                loss_ppo = loss_ppo.mean()
            """

        regular_outputs = self.model(
            input_ids=regular_input_ids,
            attention_mask=regular_attn_mask,
        )

        hidden_states = regular_outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()
  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = regular_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss_regular = loss_fct(shift_logits, shift_labels)

        loss_regular = loss_regular.mean()



        loss = loss_ppo + self.regular_coefficient * loss_regular

        #print(loss_ppo)
        #print("===================")
        #print(loss_regular)



        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

