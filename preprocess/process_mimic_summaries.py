import argparse
import random
import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None
import re
import pickle
import nltk
from collections import Counter
from tqdm import tqdm
import swifter
import string

from src.preprocess.constants import *
from src.preprocess.regular_expressions import *
from src.preprocess.utils import *


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
        "--start_from_step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--reproduce_avs_extraction",
        action='store_true'
    )
    args = parser.parse_args()
    return args


def extract_di(txt):
    # This method reproduces the original AVS source text extraction, slightly adapted for this pipeline
    # https://github.com/pengshancai/AVS_gen
    start = txt.find("Discharge Instructions:")
    end = txt.find("Followup Instructions:")
    if start < 0 or end < 0:
        return None
    di = txt[start: end].replace('\n',  ' ')
    di = ' '.join(di.split())
    return di


def extract_hc(txt):
    # This method reproduces the original AVS source text extraction, slightly adapted for this pipeline
    # https://github.com/pengshancai/AVS_gen
    start = txt.find("Brief Hospital Course:")
    if start < 0:
        return None
    end = txt.find("Medications on Admission:")
    if end == -1:
        end = txt.find("Discharge Medications:")
    if end == -1:
        end = txt.find("Discharge Disposition:")
    if end == 0 or start >= end:
        return None
    hc = txt[start: end].replace('\n',  ' ')
    hc = ' '.join(hc.split())
    # Quality check
    num_words = len(txt.split(' '))
    if num_words < 30:
        return None
    return hc


def remove_empty_and_short_summaries(mimic_df, min_length_summary=350):
    """ Removes empty summaries and summaries that are too short."""
    old_len = len(mimic_df)
    mimic_df.drop(mimic_df[mimic_df['summary'].str.len() == 0].index, axis=0, inplace=True)
    empty_len = old_len - len(mimic_df)
    mimic_df.drop(mimic_df[mimic_df['summary'].str.len() < min_length_summary].index, axis=0, inplace=True)
    short_len = old_len - empty_len - len(mimic_df)
    print(f"Removed {empty_len} / {old_len} empty summaries and {short_len} / {old_len} summaries < {min_length_summary} chars.")
    return mimic_df


def change_why_what_next_pattern_to_text(summaries):
    """ Change all occurrences of the static why, what, next pattern occuring in many MIMIC summaries to fluent text. """

    # Determine random string used instead of headings
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + '\n- '  # Replace removed dash
    summaries = summaries.apply(lambda s: WHY_WHAT_NEXT_HEADINGS_DASHED_LIST.sub(random_string, s))
    # Now replace all items after the random string, end of paragraph marked by double newline
    dash_regex = re.compile(r'(?:\.)?\n-\s{0,4}', re.MULTILINE|re.IGNORECASE)  # Also removes 
    # Filter summaries that contain random_string, replace dash with fullstop and whitespace
    def remove_dash_from_paragraphs(s):
        paragraphs = s.split(random_string)
        # For each paragraph remove all dashes until \n\n
        res = [paragraphs[0]]
        paragraphs = paragraphs[1:]
        for p in paragraphs:
            items = p.split('\n\n', 1)[0]
            items = '. '.join(dash_regex.split(items))
            if '\n\n' in p:
                items = items + '\n\n' + p.split('\n\n', 1)[1]
            res.append(items.strip())
        return '\n\n'.join(res)
    summaries = summaries.apply(lambda s: remove_dash_from_paragraphs(s) if random_string in s else s)
    return summaries


def remove_regex_dict(mimic_df, regexes, postprocess, keep=0):
    total_changed = 0
    for delimiter_name, regex in regexes.items():
        # Debug: Set column remove_regex_delimiter_name to True if regex exists in summary to see which regex was used on which example
        # mimic_df.loc[:, f"remove_regex {delimiter_name}"] = mimic_df['summary'].swifter.apply(lambda s: regex.search(s) is not None)

        mimic_df.loc[:, 'matches'] = mimic_df['summary'].swifter.apply(lambda s: regex.search(s) is not None)
        total_changed += sum(mimic_df['matches'])
        print(f"  {delimiter_name}: {sum(mimic_df['matches'])} / {len(mimic_df)}")
        mimic_df.loc[mimic_df['matches'], 'summary'] = mimic_df.loc[mimic_df['matches'], 'summary'].swifter.apply(lambda s: regex.split(s, 1)[keep])
        # Strip whitespace, remove multiple whitespaces, remove leading and trailing punctuation
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: postprocess(s))
        
    print(f"Changed total of {total_changed} / {len(mimic_df)} summaries.") 
    mimic_df.drop(columns=['matches'], inplace=True)
    return mimic_df


def write_out_sample(mimic_df, sample_idx, fields, output_file=None):
    result = ''
    for i, idx in enumerate(sample_idx):
        entries = []
        for f in fields:
            entries.append(f"{f} {i} ({idx}):\n {mimic_df.loc[idx, f]}")
        result = result + '\n' + '#'*80 + '\n' + '#'*80 + '\n' + '#'*80 + '\n' + ('\n' + '-'*80 + '\n' + '-'*80 + '\n').join(entries)
    
    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(result)
    else:
        print(result)
          

def print_current_statistics(df):
    print(f"Total entries: {len(df)}.")


def main():
    args = parse_args()
    try:
        mimic_df = pd.read_pickle(args.input_file)
    except pickle.UnpicklingError:
        mimic_df = pd.read_csv(args.input_file) 
    except:
        raise ValueError("Could not read input file. Please provide a valid pickle or csv file.")
    print(f"\nFound total of {len(mimic_df)} texts ")

    # Debug: Save intermediate results
    # mimic_df.to_pickle('/home/s/s_hegs02/scratch/mimic-iv-avs/dataset/discharge-step2.csv.pkl')

    # Debug: Sample fewer summaries from mimic_df
    # mimic_df = mimic_df.sample(n=10000, random_state=42)

    if args.reproduce_avs_extraction:
        # Reproduces avs extraction based on mimic-iii data from
        # https://aclanthology.org/2022.coling-1.544/
        if 'category' in mimic_df.columns:
            print("Found mimic-iii data. Filter discharge summaries.")
            mimic_df = mimic_df.loc[mimic_df['category'] == 'Discharge summary']
        mimic_df.loc[:, 'brief_hospital_course'] = mimic_df['text'].apply(lambda s: extract_hc(s))
        mimic_df.loc[:, 'summary'] = mimic_df['text'].apply(lambda s: extract_di(s))
        # Remove nan entries in hospital)_course and summary
        quality_check = lambda s: (s is not None) and (len(s.split(' ')) >= 30)
        mimic_df = mimic_df[mimic_df['summary'].apply(quality_check) & mimic_df['brief_hospital_course'].apply(quality_check)]
        mimic_df.to_csv(Path(args.output_dir) / 'avs_mimic_processed_summaries.csv', index=False)
        return


    if args.start_from_step <= 1:
        print("\nStep 1: Remove exact duplicates, and only keep most recent note per hospital stay.")
        # For duplicates keep first entry (most recent note)
        mimic_df = mimic_df.sort_values(['subject_id', 'hadm_id'], ascending=[False, False], ignore_index=True)
        old_len = len(mimic_df)
        mimic_df['text'] = mimic_df['text'].str.strip()
        mimic_df.drop_duplicates(subset=['subject_id', 'hadm_id', 'text'], keep='first', inplace=True)
        print(f"  Removed {old_len - len(mimic_df)} / {old_len} exact duplicates.")

        # Keep only most recent note per hospital stay
        old_len = len(mimic_df)
        mimic_df.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='first', inplace=True)
        print(f"  Removed {old_len - len(mimic_df)} / {old_len} notes from same hospital stay.")

        # Extract service given in note
        re_service = re.compile(r'^Service: (.*)$', re.IGNORECASE|re.MULTILINE)  # Either after Serive:
        re_service_extra = re.compile(r'^Date of Birth:.*Sex:\s{0,10}\w\s{0,10}___: (.*)$', re.IGNORECASE|re.MULTILINE)  # Fallback if deidentified
        mimic_df['service'] = mimic_df['text'].swifter.apply(lambda s: re_service.search(s).group(1) if re_service.search(s) is not None else None)
        mimic_df.loc[~mimic_df['service'].notnull(), 'service'] =\
            mimic_df.loc[~mimic_df['service'].notnull(), 'text'].swifter.apply(lambda s: re_service_extra.search(s).group(1) if re_service_extra.search(s) is not None else None)
        mimic_df.loc[~mimic_df['service'].notnull(), 'service'] = ''
        mimic_df['service'] = mimic_df['service'].str.strip()
        mimic_df['service'] = mimic_df['service'].str.strip(string.punctuation)
        print(f"  Map services to common names.")
        mimic_df['service'] = mimic_df['service'].apply(lambda s: SERVICE_MAPPING[s] if s in SERVICE_MAPPING else s)
        print(f"    Services: {Counter(mimic_df['service']).most_common(100)}")

        print("  Replace special character with common equivalents.")
        mimic_df['text'] = mimic_df['text'].replace(SPECIAL_CHARS_MAPPING_TO_ASCII, regex=True)
        print_current_statistics(mimic_df)


    if args.start_from_step <= 2:
        print("\nStep 2: Filter for discharge summaries and split into note and summary.")
        # Reasons for no 'Discharge Instructions:' found in 20 examples:
        #   patient deceased, incomplete note, complete note but no discharge instructions, transfer to another unit, transfer to another hospital/facility,
        #   referring to discharge instructions in another note
        old_len = len(mimic_df)
        re_ds = re.compile(r"Discharge Instructions:\n", re.IGNORECASE)
        mimic_df = mimic_df[mimic_df['text'].str.contains(re_ds, regex=True)]
        mimic_df.loc[:, ['hospital_course', 'summary']] = mimic_df.apply(lambda x: re_ds.split(x['text'], 1), axis='columns', result_type='expand').values
        mimic_df.loc[:, 'hospital_course'] = mimic_df['hospital_course'].str.strip()
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].str.strip()
        mimic_df.loc[:, 'original_summary'] = mimic_df['summary']
        mimic_df.loc[:, 'brief_hospital_course'] = mimic_df['hospital_course'].apply(lambda s: extract_hc(s))
        print(f"  New columns in dataframe: {mimic_df.columns}")
        
        # Debug: Output unfiltered notes for t-SNE embedding
        # mimic_df.to_csv(Path(args.output_dir) / 'mimic_unprocessed_summaries.csv', index=False)
        # return
        
        print("  Replace special meaning strings for processing. Replaced back at end.")
        # Check that ENCODE_STRINGS_DURING_PREPROCESSING values not in summary
        for s in ENCODE_STRINGS_DURING_PREPROCESSING.values():
            assert s not in mimic_df['summary'].values
        for k, v in ENCODE_STRINGS_DURING_PREPROCESSING.items():
            mimic_df['summary'] = mimic_df['summary'].str.replace(k, v, regex=False)
        print(f"Removed {old_len - len(mimic_df)} / {old_len} texts that do not contain 'Discharge Instruction:' and split.")
        mimic_df = remove_empty_and_short_summaries(mimic_df)
        print_current_statistics(mimic_df)


    if args.start_from_step <= 3:
        print("\nStep 3: Truncate unnecessary prefixes of summaries.")
        # Checked 1000 examples for common prefixes and performed with visual control for this prefix removal
        # Careful: Removing a preprocessing step can alter the following results
        # Strip whitespace and remove multiple whitespaces
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].str.strip()
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_multiple_whitespace.sub(' ', s))
        # Remove puncutation, keep anonymized field ___ if at beginning
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_line_punctuation_wo_underscore.sub('', s))

        print("  Remove prefixes:")
        postprocess = lambda s: re_ds_punctuation_wo_underscore.sub('', s.strip())
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: postprocess(s))
        mimic_df = remove_regex_dict(mimic_df, UNNECESSARY_SUMMARY_PREFIXES, keep=1, postprocess=postprocess)
        mimic_df = remove_empty_and_short_summaries(mimic_df)
        print_current_statistics(mimic_df)


    if args.start_from_step <= 4:
        print("\nStep 4: Remove static patterns from summaries to make them more fluent.")
        # Preprocessing
        print("  Remove all leading and trailing whitespaces in each line of text")
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: '\n'.join([x.strip() for x in s.split('\n')]))
        print("  Remove all lines that only contain punctuation")
        # Do fullstop separately, because do not want to remove the pattern '___.' for censored information in a single line.
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_line_punctuation_wo_fs.sub('', s))
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_fullstop.sub('', s))
        # Remove newlines in continuous text to allow easier matching of truncation patters (sometimes sentences) over multiple lines
        print("  Multiple whitespaces to single whitespace")
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_multiple_whitespace.sub(' ', s))
        
        print(f"  Change Why in hospital / What was done / What next pattern into fluent text.")
        # This pattern using lists is used regularly and contains useful summaries. Try to convert it into fluent text.
        print(f"  Transform {mimic_df['summary'].apply(lambda s: WHY_WHAT_NEXT_HEADINGS_DASHED_LIST.findall(s)).apply(len).sum()} list template paragraphs into text.") 
        mimic_df.loc[:, 'summary'] = change_why_what_next_pattern_to_text(mimic_df['summary'])
        # Remove all subheadings that are not followed by a list
        subheading_regex = re.compile(WHY_WHAT_NEXT_HEADINGS, re.MULTILINE|re.IGNORECASE)
        print(f"  Remove {mimic_df['summary'].apply(lambda s: subheading_regex.findall(s)).apply(len).sum()} list template headings.") 
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: subheading_regex.sub('\n', s))

        # Remove additional delimiters after paragraphs not necessary anymore
        print("  Remove all newlines in continuous text")
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_newline_in_text.sub(' ', s))
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_multiple_whitespace.sub(' ', s))
        
        print("  Replace some simplistic deidentification patterns")
        # Much more sophisticated deidentification patterns possible
        # These are based on routines for quality control at the end and were experienced during further experiments
        for replacement, regex in SIMPLE_DEIDENTIFICATION_PATTERNS:
            # Print number of replacements
            print(f"    Replace {mimic_df['summary'].apply(lambda s: len(regex.findall(s))).sum()} ___ with \'{replacement}\'.")
            mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re.sub(regex, replacement, s))

        mimic_df = remove_empty_and_short_summaries(mimic_df)
        print_current_statistics(mimic_df)


    if args.start_from_step <= 5:
        print("\nStep 5: Truncate unnecessary suffixes of summaries.")
        # There are different sections that might follow the discharge instructions and, hence, can serve as a delimiter
        # Goals during truncation:
        # * Only keep fluent text descriptions
        # * Only keep part that contain content about current stay
        # * Remove all kind of general templates, empty phrases, copy-paste text
        # In general might create some false positive, i.e., truncate too much, but considered better than false negatives, i.e., keep too much
        old_len = len(mimic_df)

        print("  Remove delimiters:")
        postprocess = lambda s: re_multiple_whitespace.sub(' ', s.strip())
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: postprocess(s))
        mimic_df = remove_regex_dict(mimic_df, RE_SUFFIXES_DICT, postprocess, keep=0)
        # Remove incomplete senteces at the end
        mimic_df.loc[:, 'summary'] = mimic_df['summary'].apply(lambda s: re_incomplete_sentence_at_end.split(s, 1)[0])

        print(f"  Remove final artifacts: single lines with no text, leading itemize symbols")
        re_no_text = re.compile(r'^[^a-z_\n]+$', re.IGNORECASE|re.MULTILINE)
        print(f"    Replace {mimic_df['summary'].apply(lambda s: len(re_no_text.findall(s))).sum()} lines with no text.")
        mimic_df['summary'] = mimic_df['summary'].apply(lambda s: re_no_text.sub('', s))
        re_item_element_line_start = re.compile(r'^' + re_item_element, re.MULTILINE)
        print(f"    Replace {mimic_df['summary'].apply(lambda s: len(re_item_element_line_start.findall(s))).sum()} itemize symbols.")
        mimic_df['summary'] = mimic_df['summary'].apply(lambda s: re_item_element_line_start.sub('', s))

        mimic_df = remove_empty_and_short_summaries(mimic_df)
        print_current_statistics(mimic_df)

    
    if args.start_from_step <= 6:
        print("\nStep 6: Only keep summaries with some minimum requirements.")
        # Remove all summaries that do not fullfil some simple statistics
        min_chars = 350 # Based on experience from 100 examples
        # Checked 11 examples with more than 3000 characters and considered unnecessary to filter out
        # * 10 valid summaries that were very detailed
        # * Post-surgery template
        max_double_newlines = 5
        min_sentences = 3
        num_words_per_deidentified = 10

        old_len = len(mimic_df)
        mimic_df = mimic_df[mimic_df['summary'].map(len) >= min_chars]
        print(f"  Removed {old_len - len(mimic_df)} summaries with less than {min_chars} characters.")
        print_current_statistics(mimic_df)

        old_len = len(mimic_df)
        # Checked 50 examples with less than 3 sentences.
        # * Approximnately half of them (24) were post-surgery instructions and hence not useful
        # * Remaining examples were rather short and mostly not of high quality, e.g. only complaint and medication changes.
        mimic_df['sentences'] = mimic_df['summary'].swifter.apply(lambda s: list(nltk.sent_tokenize(s)))
        mimic_df = mimic_df[mimic_df['sentences'].map(len) >= min_sentences]
        print(f"  Removed {old_len - len(mimic_df)} summaries with less than {min_sentences} sentences.")
        print_current_statistics(mimic_df)

        old_len = len(mimic_df)
        # Checked 50 examples more than five paragraphs.
        # * Several post-surgery/intervention templates with a single instruction per paragraph
        # * Some medication/isntructions lists
        # * Some free texts with single instructions per paragraph, not very high quality
        # * 12 where of good quality but ratio of this filter too low to keep them
        mimic_df = mimic_df[mimic_df['summary'].map(lambda s: s.count('\n\n')) <= max_double_newlines]
        print(f"  Removed {old_len - len(mimic_df)} summaries with more than {max_double_newlines} double newlines.")
        print_current_statistics(mimic_df)
               
        # Remove newlines
        print(f"  Combine all sentences with single whitespaces.")
        mimic_df.loc[:, 'summary'] = mimic_df['sentences'].swifter.apply(lambda s: re_whitespace.sub(' ', ' '.join(s)))
        mimic_df.drop(columns=['sentences'], inplace=True)

        print(f"  Decode special strings.")
        for k, v in ENCODE_STRINGS_DURING_PREPROCESSING.items():
            mimic_df['summary'] = mimic_df['summary'].str.replace(v, k, regex=False)
        
        # Count the occurrences of ___ in each summary using a Counter()
        print(f"  Count occurrences of ___ in each summary.")
        mimic_df['num_deidentified'] = mimic_df['summary'].swifter.apply(lambda s: s.count('___'))
        # Debug:
        # c = Counter(list(mimic_df['num_deidentified']))
        # print(c.most_common(100))
        old_len = len(mimic_df)
        mimic_df = mimic_df[mimic_df['num_deidentified'] <= mimic_df['summary'].map(lambda s: len(s.split(' ')) / num_words_per_deidentified)]
        # mimic_df = mimic_df[mimic_df['num_deidentified'] <= 10]
        print(f"  Removed {old_len - len(mimic_df)} summaries with more than one ___ per {num_words_per_deidentified} words.")
        print_current_statistics(mimic_df)


    if args.start_from_step <= 7:
        print("\nStep 7: Remove entries with insufficient hospital courses (short context) and records.")
        min_chars_bhc = 500 

        old_len = len(mimic_df)
        mimic_df = mimic_df[mimic_df['hospital_course'].notnull()]
        print(f"Removed {old_len - len(mimic_df)} / {old_len} records with no extracted hospital course.")
        old_len = len(mimic_df)
        mimic_df = mimic_df[mimic_df['brief_hospital_course'].notnull()]
        print(f"Removed {old_len - len(mimic_df)} / {old_len} records with no extracted brief hospital course.")

        # Change more than double newlines to double newlines
        re_more_than_double_newline = re.compile(r'\n{3,}', re.MULTILINE)
        mimic_df.loc[:, 'hospital_course'] = mimic_df['hospital_course'].apply(lambda s: re_more_than_double_newline.sub('\n\n', s))
        mimic_df.loc[:, 'brief_hospital_course'] = mimic_df['brief_hospital_course'].apply(lambda s: re_more_than_double_newline.sub('\n\n', s))
        # Keep newlines in general for better structure of the text - can still be removed later
        # mimic_df['hospital_course'] = mimic_df['hospital_course'].apply(lambda s: ' '.join((s.replace('\n', ' ')).split()))
        # mimic_df['brief_hospital_course'] = mimic_df['brief_hospital_course'].apply(lambda s: ' '.join((s.replace('\n', ' ')).split()))

        old_len = len(mimic_df)
        # Remove BHC with less than 500 characters
        # * Many of these very short and no content
        # * Some contain content but very little (often surgery report)
        mimic_df = mimic_df[mimic_df['brief_hospital_course'].map(len) >= min_chars_bhc]
        print(f"  Removed {old_len - len(mimic_df)} brief hospital courses with less than {min_chars_bhc} characters.")
        print_current_statistics(mimic_df)

    # Output data
    print(f"\nOutput data to {args.output_dir}")
    # Create output directory if it does not exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mimic_df.to_csv(Path(args.output_dir) / 'mimic_processed_summaries.csv', index=False)


    # Routines for quality control

    # 1. Check random 100 positive (included) summaries
    # write_out_sample(mimic_df, mimic_df.sample(10, random_state=42).index, ['original_summary', 'summary'], None)


    # 2. Check 100 random negative (exluded) summaries
    # First, print indices of kept entries and use them to sample negative examples before filtering
    # kept_indices = list(mimic_df.index)
    # Write kept_indices to file in output directory
    # with open(Path(args.output_dir) / 'kept_indices.txt', 'w') as f:
    #     f.write('\n'.join([str(i) for i in kept_indices]))

    # Second, sample negative examples and run pipeline only with editing (to see what changed) but not removal, i.e.
    # * Changed remove_empty_and_short_summaries to return mimic_df
    # * Skipped steps 6 and 7
    # with open(Path(args.output_dir) / 'kept_indices.txt', 'r') as f:
    #     kept_indices = [int(i) for i in f.read().split('\n')]
    # removed_samples = mimic_df.loc[~mimic_df.index.isin(kept_indices)].sample(100, random_state=42).index
    # write_out_sample(mimic_df, removed_samples, ['original_summary', 'summary'], None)
    # write_out_sample(mimic_df, removed_samples, ['original_summary', 'summary'], args.output_dir + '/100_negative_examples.txt')
        

    # 3. Check very similar paragraphs
    # num_similar = 5
    # similar_threshold = 0.98
    # # Combine paragraphs
    # paragraphs = list(itertools.chain(*[split_into_paragraphs(s) for s in mimic_df['summary']]))
    # paragraphs = [p for p in paragraphs if len(p) > 10]
    # similar = np.sum(get_pairwise_text_similarity(mimic_df['summary'].tolist(), batch_size=1000, threshold=similar_threshold), axis=0).tolist()[0]
    # sim_paragraphs = [(similar[i], paragraphs[i]) for i in range(len(similar)) if similar[i] >= num_similar]
    # print('\n\n'.join([f"# similar {sim:.2f}:\n{text}" for sim, text in sim_paragraphs]))


    # 4. External: Check summary embeddings with t-SNE labelled with test performance
    # -> Identify clusters of very similar summaries and with a high test performance (likely static text blocks)


if __name__ == "__main__":
    main()
