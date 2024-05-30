import json

def remove_empty_output(file_path, output_file_path):
    # Read the data from the file
    with open(file_path, 'r') as file:
        records = json.load(file)

    filtered_records = [record for record in records if record['output'] != "" 
                        and ((record['type'] == 'planner') or 
                        (record['type'] == 'reflector' and record['last_reward'] != record['current_reward'])) 
                        and len(record['input']) + len(record['output']) <= 5000]

    filtered_records = [record for record in filtered_records if ((record['type']=='planner') or (record['type']=='reflector' and record['last_reward']!=record['current_reward']))]

    # Write the cleaned data to the output file
    with open(output_file_path, 'w') as outfile:
        json.dump(filtered_records, outfile, indent=2)

# Example usage
if __name__ == "__main__":
    # Paths to the input and output files
    input_file_path = 'alpaca_train_prob.json'
    output_file_path = 'alpaca_train_prob_filter.json'

    # Remove records with empty output and save to a new file
    remove_empty_output(input_file_path, output_file_path)
