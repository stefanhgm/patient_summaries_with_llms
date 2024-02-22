import re
import json
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_examples",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_file_revised_examples_txt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--excluded_ids",
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Read file
    with open(args.input_file_examples, "r") as f:
        file_examples = f.readlines()
    with open(args.input_file_revised_examples_txt, "r") as f:
        file_revised_examples = f.readlines()
        
        
    # Convert revised examples to dictionary
    # Find excluded examples
    # TODO: Does not match example 44 that follows another excluded example
    # regex = re.compile(r"(\d+): (.*)\n\n\d+:", re.MULTILINE)
    # for match in regex.finditer("".join(file_revised_examples)):
    #     # print(match.groups())
    #     print(f"Excluded example {match.group(1)}: {match.group(2)}")
        
    # Find labeled example
    revised_dict = {}
    regex = re.compile(r"(\d+):(.*)\n\n([a-zA-Z_].*)\n\n([a-zA-Z_].*)\n", re.MULTILINE)
    for match in regex.finditer("".join(file_revised_examples)):
        # print(match.groups())
        revised_dict[int(match.group(1))] = {"comment": match.group(2), "hallucination_free_summary": match.group(3), "revised_summary": match.group(4)}
    print(f"Read {len(revised_dict)} examples from {args.input_file_revised_examples_txt}")
    
    # Source file is jsonl format with field "text" and "summary"
    # Read in the lines of the source file according to the keys in revised_dict and add the entries for "text" and "summary" to revised_dict
    for i, line in enumerate(file_examples):
        if i in revised_dict.keys():
            line = json.loads(line)
            revised_dict[i]["text"] = line["text"]
            revised_dict[i]["summary"] = line["summary"]
    print(f"Read {len(revised_dict)} according examples from {args.input_file_examples}")
    
    # Exclude ids excluding bad examples
    num_deleted = 0
    if args.excluded_ids != "":
        for index in sorted(args.excluded_ids.split(","), reverse=True):
            del revised_dict[int(index)]
            num_deleted += 1
        print(f"Exlcuded {num_deleted} examples. {len(revised_dict)} examples remaining.")
    else:
        print(f"No examples excluded. {len(revised_dict)} examples remaining.")
    

    # Write out "text" and "summary" to jsonl file
    revised_jsonl = []
    output_path = Path(args.output_dir)
    for i in revised_dict.keys():
        revised_jsonl.append(json.dumps({"text": revised_dict[i]["text"], "summary": revised_dict[i]["summary"]}))
    with open(output_path / 'hallucination_summaries_original.json', "w") as f:
        f.write("\n".join(revised_jsonl))
    # Write out "text" and "hallucination_free_summary" to jsonl file
    revised_jsonl = []
    for i in revised_dict.keys():
        revised_jsonl.append(json.dumps({"text": revised_dict[i]["text"], "summary": revised_dict[i]["hallucination_free_summary"]}))
    with open(output_path / 'hallucination_summaries_cleaned.json', "w") as f:
        f.write("\n".join(revised_jsonl))
    # Write out "text" and "revised_summary" to jsonl file
    revised_jsonl = []
    for i in revised_dict.keys():
        revised_jsonl.append(json.dumps({"text": revised_dict[i]["text"], "summary": revised_dict[i]["revised_summary"]}))
    with open(output_path / 'hallucination_summaries_cleaned_improved.json', "w") as f:
        f.write("\n".join(revised_jsonl))
    print(f"Wrote datasets with {len(revised_dict)} examples to {output_path}")


if __name__ == "__main__":
    main()
