# Create MIMIC-IV-Note-DI Dataset

## Prepare the MIMIC-IV-Note Database

We used the data from the [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) dataset version 2.2. 
Please applpy for access and download the data.
We assume the dataset is located in `/root/physionet.org/files/mimic-iv-note`.
Extract the files and move the database to the `/root` path to simplify the use of our scripts.

```
gunzip /root/physionet.org/files/mimic-iv-note/2.2/note/*.gz
mv /root/physionet.org/files/mimic-iv-note /root
rm -r /root/physionet.org
```

## Process the MIMIC-IV Summaries

First, create a folder to store the newly created dataset in.
Then, we set the `PYTHONPATH` to the repository and exceute the preprocessing script.
The scripts goes through several steps.
Using `--start_from_step 1` will start it from scratch.

```
mkdir /root/mimic-iv-note-di
mkdir /root/mimic-iv-note-di/dataset

export PYTHONPATH=/root/patient_summaries_with_llms
cd /root/patient_summaries_with_llms
python /root/patient_summaries_with_llms/preprocess/process_mimic_summaries.py \
  --start_from_step 1 \
  --input_file /home/s/s_hegs02/scratch/mimic-iv-note/2.2/note/discharge.csv \
  --output_dir /home/s/s_hegs02/scratch/mimic-iv-note-di/dataset
```

You should get an output starting with:

```
Found total of 331793 texts

Step 1: Remove exact duplicates, and only keep most recent note per hospital stay.
  Removed 0 / 331793 exact duplicates.
  Removed 0 / 331793 notes from same hospital stay.
Pandas Apply: 100%|█████████████████████████████████████████████████████████████████████████| 331793/331793 [00:01<00:00, 326553.88it/s]
Pandas Apply: 100%|████████████████████████████████████████████████████████████████████████████████| 318/318 [00:00<00:00, 91795.50it/s]

[...]

Total entries: 100175.

Output data to /root/mimic-iv-note-di/dataset
```

The resulting csv still contains all comlumns of the MIMIC-IV-Note database and should be stored in the `output_dir`.
Based on this we can create different datasets using the complete hopsital course or only the brief hospital course as a reference.
We will create both here.

## Select Dataset Columns and Create Splits

To select the relevant colums and create dataset splits, we use a separate script `split_dataset.py`.
The following command will use the full `hospital course` as a reference and the preprocessed `summary` column as summary.

```
python /root/patient_summaries_with_llms/preprocess/split_dataset.py \
  --input_file /root/mimic-iv-note-di/dataset/mimic_processed_summaries.csv \
  --output_dir /root/mimic-iv-note-di/dataset \
  --hospital_course_column hospital_course \
  --summary_column summary

Found total of 100175 texts
  Wrote 80140 train, 10017 valid, and 10017 test examples to /root/mimic-iv-note-di/dataset
```

To create a separate version of the dataset using the shorter `brief_hospital_course` as a reference execute:

```
mkdir /root/mimic-iv-note-di-bhc
mkdir /root/mimic-iv-note-di-bhc/dataset
python /root/patient_summaries_with_llms/preprocess/split_dataset.py \
  --input_file /root/mimic-iv-note-di/dataset/mimic_processed_summaries.csv \
  --output_dir /root/mimic-iv-note-di-bhc/dataset \
  --hospital_course_column brief_hospital_course \
  --summary_column summary
Found total of 100175 texts
  Wrote 80140 train, 10017 valid, and 10017 test examples to /root/mimic-iv-note-di-bhc/dataset
```

As a consequence, we have the jsonl files `all.json`, `train.json`, `valid.json`, `test.json` in the directories `/root/mimic-iv-note-di/dataset` and `/root/mimic-iv-note-di-bhc/dataset` for the full hospital course and the brief hospital course as references, respectively.
In this work we focus on the brief hospital course.

## Create Dataset Embeddings

To create the t-SNE embeddings of the summaries, we used a Jupyter notebook `visualize_summary_embeddings.ipynb`.
Based on the dataset splits above, one has to adapt the following paths at the top of the notebook.
The file `all_unprocessed.json` contains the unprocessed summaries.
It was obtained using the `split_dataset.py` script on the summaries outputted from the preprocessing directly after the split into hospital course and summary.
Hence, these summaries were not altered.

```
mimic4_unfiltered_path = '/root/mimic-iv-note-di/dataset/all_unprocessed.json'
mimic4_filtered_path = '/root/mimic-iv-note-di/dataset/all.json'
```

There are different ways to label the t-SNE embeddings.
By default the dummy labels of `1` for evert summary can be used.
To create an embedding labeled by the medical service, we abused the `hospital_course_column` option for the dataset splits to hold the medical service instead of the reference text.

```
python /root/patient_summaries_with_llms/preprocess/split_dataset.py \
  --input_file /root/mimic-iv-note-di/dataset/mimic_processed_summaries.csv \
  --output_dir /root/mimic-iv-note-di/dataset \
  --hospital_course_column service
  --summary_column summary
```

Assume this was done for the processed summaries `all_services.json` and all unprocessed `all_unprocessed_services.json` set the paths in the notebook to the following an be sure to use the labels for `Medical Services`.

```
mimic4_unfiltered_path = '/root/mimic-iv-note-di/dataset/all_unprocessed_services.json'
mimic4_filtered_path = '/root/mimic-iv-note-di/dataset/all_services.json'
```
