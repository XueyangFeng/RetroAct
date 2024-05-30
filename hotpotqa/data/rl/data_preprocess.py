import json
import jsonlines

def process_jsonl(input_file_path, output_file_path):
    processed_data = []
    last_planner_reward = None
    last_question_id = None

    with jsonlines.open(input_file_path) as file:
        for entry in file:
            if last_question_id != entry['question_id']:
                last_question_id = entry['question_id']
                last_planner_reward = None  # Reset for new question

            processed_entries, last_planner_reward = process_entry(entry, last_planner_reward)
            processed_data += processed_entries

    with jsonlines.open(output_file_path, mode='w') as outfile:
        for entry in processed_data:
            outfile.write(entry)

def process_entry(entry, last_planner_reward):
    trial = entry['trial']
    entries = []
    current_planner_reward = entry['reward']

    # Process planner entry
    planner_input, question = create_planner_input(entry)
    planner_output = extract_planner_output(entry['scratchpad'], question)

    planner_entry = {
            "type": "planner",
            "question_id": entry['question_id'],
            "trial": trial,
            "input": planner_input,
            "output": planner_output,
            "reward": current_planner_reward,
        }
    entries.append(planner_entry)

    # Process reflector entry if applicable
    if trial != 1 and last_planner_reward is not None:
        reflector_entry = process_reflector_entry(entry, current_planner_reward, last_planner_reward)
        entries.append(reflector_entry)

    return entries, current_planner_reward

def create_planner_input(entry):

    reflections = entry['reflection']
    question = entry['scratchpad'].split('\n', 1)[0]  
    INSTRUCTION = f"""
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types: 
(1) Search[entity], Invoke a local searcher with the chosen keywords or entities to gather relevant information. You only need to complete the thought step and output Search[Entity] in the action step, and we will return the relevant content in \"Observation\" for you. 
(2) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary. 
    {reflections}
    Question: {question}
    """
    return INSTRUCTION.strip(), question


def extract_planner_output(scratchpad, question):
    return scratchpad.split(question, 1)[-1].strip()

def process_reflector_entry(entry, current_planner_reward, last_planner_reward):
    reflector_output = extract_last_reflection(entry['reflection'])
    return {
        "type": "reflector",
        "question_id": entry['question_id'],
        "trial": entry['trial'],
        "input": entry['reflection_prompt'],
        "output": reflector_output,
        "current_reward": current_planner_reward,
        "last_reward": last_planner_reward
    }


def extract_last_reflection(reflection):
    last_reflection_index = reflection.rfind("\n-")
    if last_reflection_index != -1:
        return reflection[last_reflection_index + 2:].strip()
    else:
        return reflection

# 使用示例
input_file_path = 'raw_data.jsonl'  
output_file_path = 'processed_traj.jsonl'  
process_jsonl(input_file_path, output_file_path)


def filter_json(input_file_path, output_file_path):
    try:
        # 初始化空列表来存储过滤后的数据
        filtered_data = []

        # 读取JSONL文件，每行都是一个JSON对象
        with jsonlines.open(input_file_path) as file:
            for entry in file:
                # 单独检查 'reward' 键是否存在于 entry 中且其值不为 0
                if 'reward' in entry and entry['reward'] != 0:
                    filtered_data.append(entry)
                # 检查 'current_reward' 和 'last_reward' 中至少有一个键存在于 entry 中且其值不为 0
                elif ('current_reward' in entry and entry['current_reward'] != 0) or \
                    ('last_reward' in entry and entry['last_reward'] != 0):
                    filtered_data.append(entry)

        # 将过滤后的数据写入新的JSONL文件
        with jsonlines.open(output_file_path, mode='w') as file:
            file.write_all(filtered_data)

        print(f"Filtered data has been saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
input_file_path = 'processed_traj.jsonl'  
output_file_path = 'processed_traj_pos.jsonl' 

filter_json(input_file_path, output_file_path)


def convert_jsonl(input_file, output_file):
    converted_data = []
    instruction = ""
    with jsonlines.open(input_file) as file:
        for entry in file:
            input_text = entry["input"]
            output_text = entry["output"]
            # 构建新的数据结构
            if entry['type'] == "planner":
                converted_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "type": entry['type'],
                    "reward": entry['reward']
                })
            elif entry['type'] == "reflector":
                converted_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "type": entry['type'],
                    "last_reward": entry['last_reward'],
                    "current_reward": entry['current_reward'],               
                })
        with open(output_file, 'a', encoding='utf-8') as outfile:
            json.dump(converted_data, outfile, ensure_ascii=False, indent=4)


input_file = './processed_traj_pos.jsonl'
output_file = './alpaca_train.json'


convert_jsonl(input_file, output_file)
