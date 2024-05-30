import openai
from environment import QAEnv
import time
import os
from enum import Enum
import re
import tiktoken
from typing import List, Union, Literal
from prompts_zs import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from prompts_zs import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from langchain.prompts import PromptTemplate
import json
import string
from collections import Counter

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


        


class ReactAgent:
    def __init__(self,                 
                 action_model,
                 question_id: int,
                 max_steps: int = 5,
                 environment: QAEnv = QAEnv("train"),
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 save_dir: str = "./result/react_result.jsonl"
                 ) -> None:
        self.question_id = question_id
        self.scratchpad: str = ''
        self.finished = False

        self.max_steps = max_steps

        self.action_model = action_model
        self.agent_prompt = agent_prompt
        self.answer = ''
        self.env = environment
        self.question = environment.reset(question_id=question_id)
        self.save_dir = save_dir

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        #Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        
        #Act
        self.scratchpad += f'\nAction {self.step_n}:'

        action = self.prompt_agent()
        self.scratchpad += ' ' + action

        #Obs
        self.scratchpad += f'\nObservation {self.step_n}:'

        if parse_action(action) != None:
            action_type, argument = parse_action(action)

            if action_type == 'finish':
                self.answer = argument
                if self.is_correct():
                    self.scratchpad += 'Answer is CORRECT'
                else: 
                    self.scratchpad += 'Answer is INCORRECT'
                self.finished = True
                self.step_n += 1
                return

            if action_type == 'search':
                try:
                    self.scratchpad += format_step(self.env.step(action))
                except Exception as e:
                    print(e)
                    self.scratchpad += 'Invalid Action. Valid Actions are Search[<entity>] and Finish[<answer>].'

            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Search[<entity>] and Finish[<answer>].'
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Search[<entity>] and Finish[<answer>].'
        
        self.step_n += 1

    def prompt_agent(self) -> str:
        return format_step(self.action_model.inference(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            question = self.question,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished
    
    def f1(self) -> float:
        key = self.env.get_key()
        return F1(self.answer, key)

    def is_correct(self) -> bool:
        key = self.env.get_key()
        return EM(self.answer, key)
    """
    def is_correct(self) -> bool:
        return self.env.get_reward(self.answer)
    """ 
    def is_halted(self) -> bool:
            return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished
    
        
    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def save_scratchpad(self) -> None:
        data_to_save = {
        "question_id": self.question_id,
        "scratchpad": self.question + self.scratchpad
        }

        with open(self.save_dir, 'a') as file:
            json_line = json.dumps(data_to_save)
            file.write(json_line+'\n')


class ReflectionAgent(ReactAgent):
    def __init__(self,
                 action_model,
                 reflect_model,
                 question_id: int,
                 max_steps: int = 5,
                 max_trials: int = 5,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 environment: QAEnv = QAEnv("train"),
                 save_dir: str = "./result/reflexion_result.jsonl"
                 ) -> None:
        super().__init__(question_id=question_id, action_model=action_model, environment=environment, agent_prompt=agent_prompt, max_steps=5)
        self.reflect_model = reflect_model
        self.reflect_prompt = reflect_prompt
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.trial_n = 0
        self.save_dir = save_dir
        self.reflection_prompt = ""

    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> float:
        if self.is_correct():
            return self.f1()
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)
        self.reflection_prompt = self._build_reflection_prompt()
        self.trial_n += 1
        ReactAgent.run(self, reset)
        self.save_scratchpad()
        return self.f1()

    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION: 
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')

    def prompt_reflection(self) -> str:
        return format_step(self.reflect_model.inference(self._build_reflection_prompt()))
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            question = self.question,
                            scratchpad = truncate_scratchpad(self.scratchpad, tokenizer=self.enc))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            reflections = self.reflections_str,
                            question = self.question,
                            scratchpad = self.scratchpad)

    def save_scratchpad(self) -> None:
        data_to_save = {
        "question_id": self.question_id,
        "trial": self.trial_n,
        "reflection_prompt": self.reflection_prompt,
        "reflection": self.reflections_str,
        "scratchpad": self.question + self.scratchpad
        }

        with open(self.save_dir, 'a') as file:
            json_line = json.dumps(data_to_save)
            file.write(json_line+'\n')
    """
    def is_correct(self) -> bool:
        key = self.env.get_key()
        return EM(self.answer, key)
    """

    def f1(self) -> float:
        key = self.env.get_key()
        return F1(self.answer, key)
    
    def is_correct(self) -> bool:
        return self.f1()==1.0

### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1).lower()
        argument = match.group(2).lower()
        return action_type, argument
    
    else:
        return None


def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = gpt2_enc) -> str:
    lines = scratchpad.split('\n')

    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    if (key.lower() in answer.lower()):
        return True
    else:
        return False

def F1(answer, key) -> bool:
    normalized_prediction = normalize_answer(answer)
    normalized_ground_truth = normalize_answer(key)

    ZERO_METRIC = 0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
"""
def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)
"""