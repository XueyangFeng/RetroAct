import jsonlines

def filter_json(input_file_path, output_file_path):
    try:
        filtered_data = []

        with jsonlines.open(input_file_path) as file:
            for entry in file:
                if entry['scratchpad'].endswith("Answer is CORRECT"):
                    filtered_data.append(entry)

        with jsonlines.open(output_file_path, mode='w') as file:
            file.write_all(filtered_data)

        print(f"Filtered data has been saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
input_file_path = 'input.jsonl'  
output_file_path = 'output.jsonl' 

filter_json(input_file_path, output_file_path)
