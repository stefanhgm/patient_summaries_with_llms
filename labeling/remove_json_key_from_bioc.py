import os
import re
import argparse
from bioc import biocxml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        # default="/home/s_hegs02/MedTator/13_agreed_label_silver_validation_examples/hallucinations_10_valid_mimic_agreed_old_key.xml",
        default="/home/s_hegs02/MedTator/12_agreed_label_silver_examples/hallucinations_100_mimic_agreed_old_key.xml",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        # default="/home/s_hegs02/MedTator/13_agreed_label_silver_validation_examples/hallucinations_10_valid_mimic_agreed.xml",
        default="/home/s_hegs02/MedTator/12_agreed_label_silver_examples/hallucinations_100_mimic_agreed.xml",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Read dataset
    def read_bioc(path):
        with open(path, 'r') as fp:
            return biocxml.load(fp)
    
    input = read_bioc(args.input_file)
        
    # Replace keys
    source_text = '### JSON Key: text\n'
    target_text = 'Text:\n'
    source_summary = '### JSON Key: summary\n'
    target_summary = 'Summary:\n'
    count_text = 0
    count_summary = 0
    
    for document in input.documents:
        for passage in document.passages:
            count_text += passage.text.count(source_text)
            count_summary += passage.text.count(source_summary)
            passage.text = passage.text.replace(source_text, target_text)
            passage.text = passage.text.replace(source_summary, target_summary)
        
    print(f"Replaced {count_text} occurrences of '{source_text}' with '{target_text}'")
    print(f"Replaced {count_summary} occurrences of '{source_summary}' with '{target_summary}'")
    
    # Now fix offsets of label
    # Find digits in expression ' offset="123"/>'
    # Replace with digits - length change
    length_change = (len(target_text) - len(source_text)) + (len(target_summary) - len(source_summary))

    def sum_all_offsets(input_text):
        re_offset = re.compile(r' offset="\d+"')
        offsets = re_offset.findall(input_text)
        return sum([int(offset.split('"')[1]) for offset in offsets])
    
    old_sum = sum_all_offsets(str(biocxml.dumps(input)))
    print(f"Old sum of all offsets: {old_sum}")
    
    count_offset = 0
    for document in input.documents:
        for passage in document.passages:
            for annotation in passage.annotations:
                for location in annotation.locations:
                    # Get offset
                    offset = int(location.offset)
                    # Change offset
                    location.offset = str(offset + length_change)
                    count_offset += 1
    
    # Check if sum of all offsets is correct
    assert old_sum + count_offset * length_change == sum_all_offsets(str(biocxml.dumps(input)))
    print(f"Changed {count_offset} offsets by {length_change}")
    
    # Write to output file  
    with open(args.output_file, 'w') as f:
        biocxml.dump(input, f)
    
    

if __name__ == "__main__":
    main()