# Labeling Errors in Patient Summaries

## Dataset Subselection

To select more relevant data and improve labeling quality, we subselect the dataset.
We filtered for reference texts with at most 4000 characters and summaries with at least 600 characters.
This can be done with the following commands:

```
python /root/patient_summaries_with_llms/labeling/select_dataset_subset_with_chars.py --input_file /root/mimic-iv-note-di-bhc/dataset/train.json --output_dir /root/mimic-iv-note-di-bhc/dataset --text_max_chars 4000 --summary_min_chars 600
Writing 20931 text-summary pairs to /root/mimic-iv-note-di-bhc/dataset/train_4000_600_chars.json

python /root/patient_summaries_with_llms/labeling/select_dataset_subset_with_chars.py --input_file /root/mimic-iv-note-di-bhc/dataset/valid.json --output_dir /root/mimic-iv-note-di-bhc/dataset --text_max_chars 4000 --summary_min_chars 600
Writing 2608 text-summary pairs to /root/mimic-iv-note-di-bhc/dataset/valid_4000_600_chars.json

python /root/patient_summaries_with_llms/labeling/select_dataset_subset_with_chars.py --input_file /root/mimic-iv-note-di-bhc/dataset/test.json --output_dir /root/mimic-iv-note-di-bhc/dataset --text_max_chars 4000 --summary_min_chars 600
Writing 2639 text-summary pairs to /root/mimic-iv-note-di-bhc/dataset/test_4000_600_chars.json
```

Of these we selected 250 training examples for labeling and 100 training examples for doing parameter tuning.

```
cd /root/mimic-iv-note-di-bhc/dataset
cat train_4000_600_chars.json | awk 'NR >= 0  && NR <= 250 { print }' > train_4000_600_chars_250_labeling.json
cat train_4000_600_chars.json | awk 'NR >= 251  && NR <= 350 { print }' > train_4000_600_chars_251-350_pt.json
```

## Creating MedTator Datset

To convert the 250 labeling examples into the right format for MedTator, we run the following command:

```
mkdir -p /root/medtator/data
python /root/patient_summaries_with_llms/labeling/convert_jsonl_medtator.py --mode jsonl_to_txt_files --input /root/mimic-iv-note-di-bhc/dataset/train_4000_600_chars_250_labeling.json --output /root/medtator/data/train_4000_600_chars_250
```

This will create the files `train_4000_600_chars_250_*.txt` in the directory `/root/medtator/data/` for all 250 examples.
We used these for our labeling task in MedTator.
 
## Creating Hallucination Free and Revised Datasets

Based on the labeling results, we created three new versions of the dataset.
The first one contains the original summaries, the second one the original summaries with hallucination removed, and the third contains the original summaries with hallucination removed and revised summaries.
We stored them in a .txt file with the format `id:\n\nhallucination free summary\n\nrevised summary\n\n`.
We provide the .txt file in our repository at physionet.org.
To create the datasets from this file and store them in the `dataset` folder run:

```
python /root/patient_summaries_with_llms/labeling/create_revised_dataset.py --input_file_examples /root/mimic-iv-note-di-bhc/dataset/train_4000_600_chars_250_labeling.json --input_file_revised_examples_txt /root/MedTator/revised_examples.txt --output_dir /root/mimic-iv-note-di-bhc/dataset/ --excluded_ids 0,1,2,3,4,5,6,7,8,9,11,12
Read 112 examples from /home/s/s_hegs02/scratch/MedTator/MedTator/10_label_silver_examples_annotator_1/revised_examples.txt
Read 112 according examples from /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset/train_4000_600_chars_250_labeling.json
Exlcuded 12 examples. 100 examples remaining.
Wrote datasets with 100 examples to /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset
```

## Analyzing Labeling Results

We used the notebook `analyse_labelings.ipynb` to analyse the labeling results.

## TODO
* Continue adding step into MedTator and back into dataset