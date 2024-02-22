# Run summarization with GPT-4 

We used the `1106-Preview` version of GPT-4. 

## Setup 

```bash
pip install openai==0.27.0 guidance==0.0.64
```

## Run Summarization 

```bash
python run_summarization.py --task_id 4 --model_name gpt-4 --n_shot 3 --verbose

# or 
bash run_all.sh
```

## Preparing Data for GPT-4 Experiments

Following commands were used to generate the data:

```bash
tail -n 10 train.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/prompt_train.json
tail -n 10 valid.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/prompt_valid.json

tail -n 10 train.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_1_in-context.json
tail -n 100 test.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_1_test.json
tail -n 10 train_4000_600_chars.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_2_in-context.json
tail -n 100 test_4000_600_chars.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_2_test.json

# Use dataset of 10 cleaned and improved (validation of annotation) examples for exp_3_in-context.json
cp hallucination_summaries_cleaned_improved_valid_10.json ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_3_in-context.json
# Used dataset of 100 cleaned and improved examples for exp_3_test.json
cp hallucination_summaries_cleaned_improved_test_100.json ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_3_test.json

# Also use the 10 validation examples here for in-context examples
cp hallucination_summaries_original_valid_10.json ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_4_in-context.json
cp hallucination_summaries_cleaned_valid_10.json ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_5_in-context.json
cp hallucination_summaries_cleaned_improved_valid_10.json ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_6_in-context.json
tail -n 50 test_4000_600_chars.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_4_test.json
tail -n 50 test_4000_600_chars.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_5_test.json
tail -n 50 test_4000_600_chars.json > ~/patient_summaries_with_llms/gpt-4/summarization_data/exp_6_test.json
```