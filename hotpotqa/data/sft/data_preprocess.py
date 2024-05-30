import jsonlines

def process_jsonl(input_file_path, output_file_path):
    processed_data = []
    last_question_id = None

    with jsonlines.open(input_file_path) as file:
        for entry in file:
            if last_question_id != entry['question_id']:
                last_question_id = entry['question_id']

            processed_entries = process_entry(entry)
            processed_data += processed_entries

    with jsonlines.open(output_file_path, mode='w') as outfile:
        for entry in processed_data:
            outfile.write(entry)

def process_entry(entry):
    trial = entry['trial']
    entries = []

    # Process planner entry
    planner_input, question = create_planner_input(entry)
    planner_output = extract_planner_output(entry['scratchpad'], question)
    #splitted_planner_steps = split_planner_steps(planner_input, planner_output)

    planner_entry = {
            "type": "planner",
            "question_id": entry['question_id'],
            "trial": trial,
            "input": planner_input,
            "output": planner_output,
        }
    entries.append(planner_entry)

    if entry["trial"] != 1:
        reflector_entry = process_reflector_entry(entry)
        entries.append(reflector_entry)

    return entries


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
    # 从 scratchpad 中提取除 question 之外的部分作为 output
    return scratchpad.split(question, 1)[-1].strip()

def process_reflector_entry(entry):
    reflector_input = create_reflector_input(entry)
    reflector_output = extract_last_reflection(entry['reflection'])
    return {
        "type": "reflector",
        "question_id": entry['question_id'],
        "trial": entry['trial'],
        "input": reflector_input,
        "output": reflector_output,
    }


def create_reflector_input(entry):
    reflection_prompt = entry['reflection_prompt']
    parts = reflection_prompt.split(" \nHere are some examples:")
    instruction = parts[0] if len(parts) > 1 else ""
    remaining_part = parts[1] if len(parts) > 1 else ""
    trials = remaining_part.split("Previous trial:")
    if len(trials) > 1:
        reflector_input = "Previous trial:" + trials[1]
    else:
        reflector_input = ""
    return instruction + " " + reflector_input

def extract_last_reflection(reflection):
    last_reflection_index = reflection.rfind("\n-")
    if last_reflection_index != -1:
        return reflection[last_reflection_index + 2:].strip()
    else:
        return reflection

# 使用示例
input_file_path = 'input.jsonl'
output_file_path = 'processed_traj.jsonl'
process_jsonl(input_file_path, output_file_path)


import json
import jsonlines

def convert_jsonl(input_file, output_file):
    converted_data = []
    instruction = ""
    with jsonlines.open(input_file) as file:
        for entry in file:



            input_text = entry["input"]
            output_text = entry["output"]


            # 构建新的数据结构
            converted_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "type": entry['type']
            })

        # 将转换后的数据写入 JSON 文件
        with open(output_file, 'a', encoding='utf-8') as outfile:
            json.dump(converted_data, outfile, ensure_ascii=False, indent=4)


# 输入文件和输出文件路径
input_file = './processed_traj.jsonl'
output_file = './alpaca_train.json'



# 执行转换
convert_jsonl(input_file, output_file)
