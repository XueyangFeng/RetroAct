import os
import sys
import json
import openai
sys.path.append('/data/fengxueyang/intercode')
from intercode.envs import (
    BashEnv, SqlEnv, ACTION_EXEC
)

from typing import Dict, List
from utils import ACTION_PARSER_MAP_REACT
from prompts.prompts import TemplateReAct
import time
from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple
from utils import LocalModel 


#openai.api_type = "azure"
#openai.api_base = "https://se-algos-datagen8.openai.azure.com/"
#openai.api_key = "8ae4da0f686248e181b41bff23d64d12"
#openai.api_version = "2024-02-01"


openai.api_base = 'https://api.chatgpt-3.vip/v1/'
openai.api_key = 'sk-nmf6vzAXRbpI7basAc0b1d23435f4f2eAcC1D9905f9aF3B8'


def llm(model, prompt: str,  stop: List[str] = ["\n"]):
    try:
        cur_try = 0
        while cur_try < 6:
            text = model.inference(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
            # dumb way to do this
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)


SETTING_MAP = {
    "sql": "MySQL Database",
    "bash": "Bourne Shell"
}

def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]

class ExperimentWrapper():
    def __init__(self, args):
        self.args = args

        # Set environment (No logging for env)
        self.env = None
        if args.env == 'sql':
            self.env = SqlEnv(image_name=args.image_name,
                data_path=args.data_path, preprocess=preprocess_sql)
        elif args.env == 'bash':
            self.env = BashEnv(image_name=args.image_name,
                data_path=args.data_path)
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        self.log_data = {}
        
        # Initialize prompt template
        self.template = TemplateReAct(args.env, SETTING_MAP[args.env])

        # Initialize parser
        self.action_parser = ACTION_PARSER_MAP_REACT[args.env]


    def intercode_run(self ,model, base_prompt, memory: List[str], to_print=True, ob=' ') -> Tuple[EnvironmentHistory, bool]:
        #取最近的三次memory
        if len(memory) > 3:
            env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
        else:
            env_history = EnvironmentHistory(base_prompt, ob, memory, [])

        env_history.reset()

        if to_print:
            print(ob)
            sys.stdout.flush()
        
        cur_step = 0
        done = False
        while cur_step < 10:
            
            

            print("******")
            print(str(env_history))
            print("******")
            thought_action = model.inference(prompt=str(env_history) + f"Thought {cur_step + 1}:", stop_strs=["Observation","\nObservation","\n\nObservation",f"\nObservation {cur_step + 1}:"])

            text_list = thought_action.strip().split('Observation')
            thought_action = text_list[0].replace("\n",'')   

            print(f"llm_thought_action {cur_step + 1}:<",thought_action,">")

            try:
                thought, action = thought_action.strip().split(f"Action {cur_step + 1}: ")
            except:
                # Retry action generation if initial `llm` call did not yield any action
                print('ohh...', thought_action)
                thought = thought_action.strip().split('\n')[0]
                action = model.inference(prompt=str(env_history)  + f"Thought {cur_step + 1}: {thought}\nAction {cur_step + 1}:", stop_strs=[f"\n"]).strip()
                text_list_tmp = thought_action.strip().split('Observation')
                action = text_list_tmp[0].replace("\n",'')

            print("Action:",action)

            env_history.add("thought", f'Thought {cur_step + 1}: {thought}')
            env_history.add("action", f'Action {cur_step + 1}: {action}')

            # Parse action + execute in Intercode environment
            action_parsed, is_code = self.action_parser(action)
            if not is_code:
                reward = 0.0
                observation = f"Error executing query: Your last `execute` action did not contain {self.args.env} code"
            else:
                observation, reward, done, info = self.env.step(action_parsed)
                valid_action = info[ACTION_EXEC]

            print("Thought:",thought)
            print("Last_Action:",action_parsed)

            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > 350:
                observation = observation[:350]
            elif isinstance(observation, list) and len(observation) > 25:
                observation = observation[:25]

            ## Update Prompt with latest turn information
            #step_str = f"Thought {cur_step}: {thought}\nAction {cur_step}: {action}\nObservation {cur_step}: {observation}\n"
            #prompt += step_str

            #print(type(reward))
            #print(reward)
            
            env_history.add("observation", f'Observation {cur_step + 1}: {str(observation)}')

            if to_print:
                #print(f' Thought {cur_step}: {thought}\nAction {cur_step}: {action}\nObservation {cur_step}: {str(observation)}')
                print(f'Observation {cur_step + 1}: {str(observation)}')
                sys.stdout.flush()
            if reward ==  1.0:
                return env_history, True
            elif env_history.check_is_exhausted():
                return env_history, False
            
            if done:
                break  

            cur_step += 1

        return env_history, False





def run_trial(
        args,
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model
    ) -> List[Dict[str, Any]]:
    
    expr_wrapper = ExperimentWrapper(args)
    env = expr_wrapper.env

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)
    rewards : float = 0
        
    #env_config是一个列表，每个元素是一个字典 name memory skip success,env config决定了任务数量
    for z, env_config in enumerate(env_configs):

        env.reset(z) #ob是task信息， info是meta information

        record = expr_wrapper.env.data_loader.get(z)
        print("########\n")
        print("record:",record)
        
        if env_config["is_success"]:
            num_successes += 1
            #比如trial 1做对了，写在trial 2里面
            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue
        
        base_prompt = expr_wrapper.template.get_init_msg() + expr_wrapper.template.get_demos() + '\n'
                     
        ob = expr_wrapper.template.get_query_msg(expr_wrapper.env.query) 


        print(f'Query {z}: {expr_wrapper.env.query}')

        final_env_history, is_success= expr_wrapper.intercode_run( model=model, base_prompt=base_prompt, memory=env_config["memory"] if use_memory else [], to_print=True, ob=ob)     

        #6类任务，k，v是字典的元素
        #for i, (k, v) in enumerate(PREFIXES.items()):
        #    if name.startswith(k):
        #        base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
        #        final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob, model=model)
#
        # update env config
        if is_success:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            env_configs[z]['is_success'] = True
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'
        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')
        # log env results to trial log
        with open(trial_log_path, 'a') as wf:
            wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
        


    # close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs