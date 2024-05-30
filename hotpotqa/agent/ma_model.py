from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import openai
import time
import os
class LocalModel:
    def __init__(self, model_dir, lora_dir=None):
            self.enable_lora = True
            self.llm = LLM(model=model_dir, enable_lora=True)
            
            
    def inference(self, prompts, lora_dir,stop=["\n"], max_len=250): 
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_len, logprobs=1,stop=stop)   
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params,  lora_request=LoRARequest("student_lora", 1, lora_dir))
        #print(outputs)
        #return outputs[0].outputs[0].logprobs
        return outputs[0].outputs[0].text


class LocalMaModel:
    def __init__(self, model_dir, lora_dir=None):
        if lora_dir == None:
            self.enable_lora = False
            self.llm = LLM(model=model_dir, gpu_memory_utilization=0.45) 
        if lora_dir != None:
            self.enable_lora = True
            self.lora_dir = lora_dir
            self.llm = LLM(model=model_dir, enable_lora=True, gpu_memory_utilization=0.45)
            
            
    def inference(self, prompts, stop=["\n"], max_len=250): 
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_len, stop=stop)   
        if not self.enable_lora:
            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        else:
            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params,  lora_request=LoRARequest("student_lora", 1, self.lora_dir))
        return outputs[0].outputs[0].text


class OpenaiModel:
    def __init__(self, model, api_key, api_base):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
            
    
    def inference(self, prompt, stop=["\n"], max_len=250):
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        messages = [{"role": "user", "content": prompt}]
        while(True):
            try:
                response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=0,
                        max_tokens=max_len,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        return response.choices[0]['message']["content"]