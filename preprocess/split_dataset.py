
import argparse
import pandas as pd
import pickle
import math
from pathlib import Path


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
        "--prefix",
        type=str,
        default='',
    )
    parser.add_argument(
        "--hospital_course_column",
        type=str,
    )
    parser.add_argument(
        "--summary_column",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        mimic_df = pd.read_pickle(args.input_file)
    except pickle.UnpicklingError:
        mimic_df = pd.read_csv(args.input_file) 
    except:
        raise ValueError("Could not read input file. Please provide a valid pickle or csv file.")
    print(f"Found total of {len(mimic_df)} texts")

    # Rename columns and shuffle
    mimic_df = mimic_df[[args.hospital_course_column, args.summary_column]]
    mimic_df.rename(columns={args.hospital_course_column: 'text', args.summary_column: 'summary'}, inplace = True)
    mimic_df = mimic_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train, valid, test
    num_train = math.floor(len(mimic_df) * 0.8)
    num_valid = math.floor(len(mimic_df) * 0.1)
    num_test = math.floor(len(mimic_df) * 0.1)

    all_out = Path(args.output_dir) / (args.prefix + 'all.json')
    train_out = Path(args.output_dir) / (args.prefix + 'train.json')
    valid_out = Path(args.output_dir) / (args.prefix + 'valid.json')
    test_out = Path(args.output_dir) / (args.prefix + 'test.json')

    mimic_df.to_json(all_out, orient='records', lines=True)
    mimic_df.iloc[0:num_train].to_json(train_out, orient='records', lines=True)
    mimic_df.iloc[num_train:num_train+num_valid].to_json(valid_out, orient='records', lines=True)
    mimic_df.iloc[num_train+num_valid:].to_json(test_out, orient='records', lines=True)

    print(f"  Wrote {num_train} train, {num_valid} valid, and {num_test} test examples to {args.output_dir}")


if __name__ == '__main__':
    main()