import argparse
import json
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--text_max_chars",
        type=int,
        default=99999,
    )
    parser.add_argument(
        "--summary_min_chars",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args

def main():
    # Read input jsonl file
    args = parse_args()
    with open(args.input_file, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
        
    # Filter out texts with less than text_max_chars characters via list comprehension
    data = [d for d in data if len(d['text']) <= args.text_max_chars]
    data = [d for d in data if len(d['summary']) >= args.summary_min_chars]
    
    # Write output as jsonl file using pandas
    file_name = Path(args.input_file).name
    file_name = file_name[:file_name.rfind('.')]
    output_file = Path(args.output_dir) / (file_name + f"_{args.text_max_chars}_{args.summary_min_chars}_chars.json")
    print(f"Writing {len(data)} text-summary pairs to {output_file}")
    df = pd.DataFrame(data)
    df.to_json(output_file, orient='records', lines=True)
    
if __name__ == "__main__":
    main()