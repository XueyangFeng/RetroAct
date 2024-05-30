"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import re
import importlib
import alfworld
import alfworld.agents.environment
from utils import LocalModel, OpenaiModel
from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple
 
FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts_backup.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

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

def format_action(act):
    # print(act)
    action_list = act.lower().split()
    # print(action_list)
    if 'THOUGHT:' in action_list:
        return act, True
    entities = []
    for i, a in enumerate(action_list):
        if a.isdigit() and i > 1:
            entity = action_list[i - 1] + ' ' + a
            entities.append(entity)
    if 'go' in action_list and len(entities) == 1:
        action = f"go to {entities[0]}"
    elif 'take' in action_list and len(entities) == 2:
        action = f"take {entities[0]} from {entities[1]}"
    elif 'put' in action_list and len(entities) == 2:
        action = f"put {entities[0]} in/on {entities[1]}"
    elif 'open' in action_list and len(entities) == 1:
        action = f"open {entities[0]}"
    elif 'close' in action_list and len(entities) == 2:
        action = f"close {entities[0]}"
    elif 'toggle' in action_list and len(entities) == 2:
        action = f"toggle {entities[0]} {entities[1]}"
    elif 'clean' in action_list and len(entities) == 2:
        action = f"clean {entities[0]} with {entities[1]}"
    elif 'heat' in action_list and len(entities) == 2:
        action = f"heat {entities[0]} with {entities[1]}"
    elif 'cool' in action_list and len(entities) == 2:
        action = f"cool {entities[0]} with {entities[1]}"
    else:
        action = act
        if len(action_list) > 8:  # 因为我算着最长的正确指令split出来是6，是为了排除他胡说八道导致长度超出openai限制，进入调用api的死循环
            return act, False
    return action, True

#去掉这个开头
def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def alfworld_run(model, env, base_prompt, memory: List[str], to_print=True, ob='') -> Tuple[EnvironmentHistory, bool]:
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
    while cur_step < 49:
        action = llm(model, str(env_history) + ">", stop=['\n']).strip()
        action, flag = format_action(act=action)
        env_history.add("action", action)
        if action.startswith('think:'):
            observation = 'OK.'
            reward, done = False, False
        elif not flag:        # 好处就是不用循环三遍
            observation = "Please generate an action follow the action format in the examples."
            reward, done = False, False
        else:
            cnt = 0
            observation = 'Nothing happens.'
            while observation == 'Nothing happens.' and cnt < 1:
                observation, reward, done, info = env.step([action])
                observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
                cnt += 1
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}



def run_trial(
        model: LocalModel,
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"
    #split = "train"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    #env_config是一个列表，每个元素是一个字典 name memory skip success,env config决定了任务数量
    for z, env_config in enumerate(env_configs):
        ob, info = env.reset() #ob是task信息， info是meta information
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        #name是任务名，这个任务名一般会比较抽象
        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1
            #比如trial 1做对了，写在trial 2里面
            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')           
            continue

        #6类任务，k，v是字典的元素
        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                #如果batch化的话，env config应该和batch_size是对应的。也就是说env_config是一个n*batchsize的列表，列表里面有config
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                final_env_history, is_success = alfworld_run(model, env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob)
                #这里也要重写，加一个for，判断
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