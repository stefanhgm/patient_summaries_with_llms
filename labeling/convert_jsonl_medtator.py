import json
import argparse
import glob
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help='The mode to run the script in: "jsonl_to_txt_files", "txt_files_to_jsonl".',
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help='The input file pattern. Must be either "*.jsonl" or a prefix of txt files.',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help='The output file path. Must be either "*.jsonl" or a prefix of txt files.',
    )
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()

    # Convert the file.
    if args.mode == 'jsonl_to_txt_files':
        jsonl_to_txt_files(args.input, args.output)
    elif args.mode == 'txt_files_to_jsonl':
        txt_files_to_jsonl(args.input, args.output)
    else:
        print('Invalid mode. Must be either "jsonl_to_txt_files" or "txt_files_to_jsonl".')

def jsonl_to_txt_files(input, output):
    # Read the JSONL file.
    with open(input, 'r') as f:
        jsonl = f.read().splitlines()

    # Convert the JSONL to text files.
    for i, json_str in enumerate(jsonl):
        # Convert the JSON string to a dictionary.
        json_dict = json.loads(json_str)

        # Get the text from the dictionary.
        text = ''
        for key, value in json_dict.items():
            # Add key with separot and value to text.
            text += f'### JSON Key: {key}\n{value}\n\n'

        # Write the text to a text file index by i filled with leading zeros.
        num_leading_zeros = len(str(len(jsonl)))
        with open(f'{output}_{str(i).zfill(num_leading_zeros)}.txt', 'w') as f:
            f.write(text)


def txt_files_to_jsonl(input, output):
    # Read the text files starting with input and ending with .txt using glob.
    text_files = []
    for text_file in glob.glob(f'{input}*.txt'):
        text_files.append(text_file)    
    text_files.sort()

    # Convert the text files to a JSONL file.
    json_dicts = []
    for text_file in text_files:
        # Read the text file.
        with open(text_file, 'r') as f:
            text = f.read()

        # Convert the text to JSON key and values.
        # Use regex to split the text into key and value pairs.
        json_dict = {}
        for key, value in re.findall(r'### JSON Key: (.*)\n(.*)\n\n', text):
            json_dict[key] = value

        json_dicts.append(json_dict)
    
    # Write the JSONL file.
    dataframe = pd.DataFrame(json_dicts)
    dataframe.to_json(output, orient='records', lines=True)


if __name__ == '__main__':
    main()
