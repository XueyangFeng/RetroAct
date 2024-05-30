import os
import sys
from openai import OpenAI
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class LocalModel:
    def __init__(self, model_dir, lora_dir=None):
        if lora_dir == None:
            self.enable_lora = False
            self.llm = LLM(model=model_dir, tensor_parallel_size=1)  
        if lora_dir != None:
            self.enable_lora = True
            self.lora_dir = lora_dir
            self.llm = LLM(model=model_dir, enable_lora=True)
            
            
    def inference(self, prompt, temperature: float = 0.0, max_tokens: int = 100, stop_strs=["\n"]): 
        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens, logprobs=1,stop=stop_strs)   
        if not self.enable_lora:
            outputs = self.llm.generate(prompts=prompt, sampling_params=sampling_params)
        else:
            outputs = self.llm.generate(prompts=prompt, sampling_params=sampling_params,  lora_request=LoRARequest("student_lora", 1, self.lora_dir))
        #print(outputs)
        #return outputs[0].outputs[0].logprobs
        return outputs[0].outputs[0].text


class OpenaiModel:
    def __init__(self, api_key, api_base):
        self.client = OpenAI(api_key='sk-nmf6vzAXRbpI7basAc0b1d23435f4f2eAcC1D9905f9aF3B8',
                             base_url='https://api.chatgpt-3.vip/v1/')

            
    def inference(self, prompt, model: str = "gpt-3.5-turbo", temperature: float = 0.0, max_tokens: int = 100, stop_strs=["\n"], is_batched: bool = False):
        messages = [{"role": "user", "content": prompt}]
        while(True):
            try:
                response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop_strs
                    )
                content = response.choices[0].message.content
                return content.split('\n')[0]
            except:
                time.sleep(1)
