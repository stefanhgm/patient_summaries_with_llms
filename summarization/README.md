# Summarization Models

This folder contains the scripts to run and evanluate the summarization models LED and Llama 2.

## Final Performance Runs

* Choose best model according to BERTScore on 100 validation examples
* Then evluate on 100 test examples

### LED-base

* Training with paramter tuning on wandb

Evaluation on long data test set:
```
python summarization/run_summarization.py --model_name_or_path ~/scratch/mimic-iv-note-di-bhc/models/led-base-16384/mimic-iv-note-di-bhc_led-base-16384_long_data_100_valid_done/dropout_0.05_learning_rate_5e-5/ --do_predict --test_file ~/scratch/mimic-iv-note-di-bhc/dataset/test_last_100.json --output_dir ~/scratch/mimic-iv-note-di-bhc/models/led-base-16384/mimic-iv-note-di-bhc_led-base-16384_long_data_100_valid_done/test_160000_output --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --max_source_length 4096 --max_target_length 350

{'rouge1': 43.32018605114362, 'rouge2': 17.04703537496032, 'rouge3': 8.255309078577513, 'rouge4': 4.301309886142394, 'rougeL': 29.214412778548695, 'words': 74.36, 'bert_score': 87.98089069128036, 'bert_score_deberta-large': 63.5219652056694, 'sari': 46.390301194434564}
$43.32$ & $17.05$ & $8.26$ & $4.30$ & $29.21$ & $87.98$ & $63.52$ & $46.39$ & $74.36$
```


Evaluation on 4000_600_chars test set:
```
python summarization/run_summarization.py --model_name_or_path ~/scratch/mimic-iv-note-di-bhc/models/led-base-16384/mimic-iv-note-di-bhc_led-base-16384_4000_600_chars_100_valid_done/dropout_0.2_learning_rate_1e-5/ --do_predict --test_file ~/scratch/mimic-iv-note-di-bhc/dataset/test_4000_600_chars_last_100.json --output_dir ~/scratch/mimic-iv-note-di-bhc/models/led-base-16384/mimic-iv-note-di-bhc_led-base-16384_4000_600_chars_100_valid_done/test_160000_output --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --max_source_length 4096 --max_target_length 350

{'rouge1': 42.2980476882839, 'rouge2': 14.976131359106965, 'rouge3': 7.043920762758955, 'rouge4': 3.872723295137177, 'rougeL': 26.502820195241306, 'words': 117.81, 'bert_score': 86.71253889799118, 'bert_score_deberta-large': 60.84960976243019, 'sari': 44.38245205873611}
$42.30$ & $14.98$ & $7.04$ & $3.87$ & $26.50$ & $86.71$ & $60.85$ & $44.38$ & $117.81$
```

### LED-large

* Training with paramter tuning on wandb

Evaluation on long data test set:
```
CUDA_VISIBLE_DEVICES=0 python summarization/run_summarization.py --model_name_or_path ~/scratch/mimic-iv-note-di-bhc/models/led-large-16384/mimic-iv-note-di-bhc_led-large-16384_long_data_100_valid_done/dropout_0.2_learning_rate_5e-5/ --do_predict --test_file ~/scratch/mimic-iv-note-di-bhc/dataset/test_last_100.json --output_dir ~/scratch/mimic-iv-note-di-bhc/models/led-large-16384/mimic-iv-note-di-bhc_led-large-16384_long_data_100_valid_done/test_160000_output --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --max_source_length 4096 --max_target_length 350

{'rouge1': 43.822986792431145, 'rouge2': 17.331937668711536, 'rouge3': 8.846417725899846, 'rouge4': 4.9231165894123174, 'rougeL': 29.887635210786037, 'words': 76.99, 'bert_score': 88.10762703418732, 'bert_score_deberta-large': 64.11935889720917, 'sari': 46.71329387536656}
$43.82$ & $17.33$ & $8.85$ & $4.92$ & $29.89$ & $88.11$ & $64.12$ & $46.71$ & $76.99$
```


Evaluation on 4000_600_chars test set:
```
CUDA_VISIBLE_DEVICES=2 python summarization/run_summarization.py --model_name_or_path ~/scratch/mimic-iv-note-di-bhc/models/led-large-16384/mimic-iv-note-di-bhc_led-large-16384_4000_600_chars_100_valid_done/dropout_0.2_learning_rate_1e-5/ --do_predict --test_file ~/scratch/mimic-iv-note-di-bhc/dataset/test_4000_600_chars_last_100.json --output_dir ~/scratch/mimic-iv-note-di-bhc/models/led-large-16384/mimic-iv-note-di-bhc_led-large-16384_4000_600_chars_100_valid_done/test_160000_output --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --max_source_length 4096 --max_target_length 350

{'rouge1': 46.206983410265465, 'rouge2': 17.376355377659934, 'rouge3': 8.715205915075776, 'rouge4': 5.13851072281858, 'rougeL': 28.865386144461212, 'words': 117.59, 'bert_score': 87.4952802658081, 'bert_score_deberta-large': 63.51826936006546, 'sari': 45.84487176324999}
$46.21$ & $17.38$ & $8.72$ & $5.14$ & $28.87$ & $87.50$ & $63.52$ & $45.84$ & $117.59$
```

### LED-large long data



### LLaMA 7B

* Training with paramter tuning on wandb

#### Long data
* Best model uses only _steps=80_ (best model in best_val_loss folder)

Evaluation on long data test set:
```
# With data_files["test"] = test_last_100.json

python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-70b-hf  --evaluation --evaluation_model_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/models/Llama-2-70b-hf/mimic-iv-note-di-bhc_Llama-2-70b-hf_long_data_100_valid_done/lora_rank_32_lora_alpha_32_lora_dropout_0.05_num_target_modules_2_learning_rate_2e-4/best_val_loss --data_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset --output_path /home/s/s_hegs02/scratch/debug --num_train_examples 100 --num_val_examples 100 --num_test_examples 100

{'rouge1': 38.8343594182866, 'rouge2': 12.97963450275313, 'rouge3': 5.273994688087981, 'rouge4': 2.388816250393242, 'rougeL': 24.811244853956904, 'words': 71.92, 'bert_score': 86.49026721715927, 'bert_score_deberta-large': 60.77595293521881, 'sari': 44.27485176179068}
{'rouge1': 39.47507417776543, 'rouge2': 13.508055870992669, 'rouge3': 5.474262681411284, 'rouge4': 2.4299066163652707, 'rougeL': 25.201416964411987, 'words': 77.79, 'bert_score': 84.75045335292816, 'bert_score_deberta-large': 60.05594950914382, 'sari': 44.487989828395825}
{'rouge1': 38.167512768514875, 'rouge2': 12.432498880471945, 'rouge3': 5.166171874893524, 'rouge4': 2.2279453931573583, 'rougeL': 24.912878164301006, 'words': 69.49, 'bert_score': 86.43438655138016, 'bert_score_deberta-large': 60.743097960948944, 'sari': 44.387922498396605}
{'rouge1': 38.474542505246575, 'rouge2': 12.59729023685818, 'rouge3': 5.032502045215025, 'rouge4': 2.08290036294411, 'rougeL': 24.771698689284257, 'words': 71.22, 'bert_score': 86.25028610229492, 'bert_score_deberta-large': 60.663696229457855, 'sari': 43.91790431156874}
{'rouge1': 36.83328812388325, 'rouge2': 11.758537910375646, 'rouge3': 4.723083386770241, 'rouge4': 2.0583976340860497, 'rougeL': 23.944460740287273, 'words': 75.22, 'bert_score': 84.4628956913948, 'bert_score_deberta-large': 58.90815430879592, 'sari': 43.55259789375479}
```


#### 4000_600_chars

* Training with paramter tuning on wandb
* Best model uses steps=100

Evaluation on 4000_600_chars test set:
```
# With data_files["test"] = test_4000_600_chars_last_100.json

python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-7b-hf  --evaluation --evaluation_model_path /home/s_hegs02/mimic-iv-note-di-bhc/models/Llama-2-7b-hf/mimic-iv-note-di-bhc_Llama-2-7b-hf_4000_600_chars_100_valid_done/lora_rank_8_lora_alpha_32_lora_dropout_0.1_num_target_modules_2_learning_rate_2e-5/best_val_loss --data_path /home/s_hegs02/mimic-iv-note-di-bhc/dataset --output_path /home/s_hegs02/mimic-iv-note-di-bhc/models/Llama-2-7b-hf/mimic-iv-note-di-bhc_Llama-2-7b-hf_4000_600_chars_100_valid_done/lora_rank_8_lora_alpha_32_lora_dropout_0.1_num_target_modules_2_learning_rate_2e-5_test --num_train_examples 100 --num_val_examples 100 --num_test_examples 100

{'rouge1': 38.18726652084839, 'rouge2': 12.522356326993874, 'rouge3': 5.300379506691891, 'rouge4': 2.6844528863937667, 'rougeL': 23.425086591887453, 'words': 104.15, 'bert_score': 84.68257343769073, 'bert_score_deberta-large': 58.7152236700058, 'sari': 43.39504555237768}
{'rouge1': 36.55885611308579, 'rouge2': 11.770071884328013, 'rouge3': 4.96413117973415, 'rouge4': 2.3760043053086815, 'rougeL': 22.520381900800416, 'words': 100.56, 'bert_score': 82.0653041601181, 'bert_score_deberta-large': 56.56696653366089, 'sari': 41.94318296286558}
{'rouge1': 36.865581109022166, 'rouge2': 12.035793003259116, 'rouge3': 5.477773544226487, 'rouge4': 2.831216527519416, 'rougeL': 22.85251778524732, 'words': 94.03, 'bert_score': 81.28817218542099, 'bert_score_deberta-large': 56.447242498397834, 'sari': 42.631882102051144}
{'rouge1': 36.45585401833812, 'rouge2': 11.507301855132011, 'rouge3': 4.979182106965462, 'rouge4': 2.525371873858721, 'rougeL': 22.339373505129323, 'words': 100.47, 'bert_score': 82.16391408443451, 'bert_score_deberta-large': 56.96498113870621, 'sari': 42.04524834049528}
{'rouge1': 36.698585402455606, 'rouge2': 11.78729495150544, 'rouge3': 4.868574815166207, 'rouge4': 2.2399180637599803, 'rougeL': 22.530625996895047, 'words': 103.47, 'bert_score': 81.9952797293663, 'bert_score_deberta-large': 56.640745639801025, 'sari': 42.02560221098953}
```


#### Train on hallucination dataset

Important Parameters:
* Set max_steps to 100 (best in parameter tuning)
* Set gradient_accumulation_steps to 16

Set training data (`data_files["train"]`) depending on setting:
* Original: `hallucination_summaries_original_test_100.json`
* Cleaned: `hallucination_summaries_cleaned_test_100.json`
* Cleaned and Improved: `hallucination_summaries_cleaned_improved_test_100.json`

Careful: Set `output_path` accordingly for each setting!

```
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-7b-hf --data_path /home/s_hegs02/mimic-iv-note-di-bhc/dataset --output_path /home/s_hegs02/mimic-iv-note-di-bhc/models/Llama-2-7b-hf/mimic-iv-note-di-bhc_Llama-2-7b-hf_4000_600_chars_100_hallucinations/original --device cuda --max_steps 100 --save_and_logging_steps 100 --batch_size 1 --gradient_accumulation_steps 16 --lora_rank 8 --lora_alpha 32 --lora_dropout 0.1 --num_target_modules 2 --learning_rate 2e-5 --num_train_examples 100 --num_val_examples 100 --num_test_examples 100
```


### LLaMA 70B

* Training with paramter tuning on wandb
* Both 70B models best with steps=20 (folder checkpoint_20 = best_val_loss)

#### Long data

* Best model uses only _steps=20_

Evaluation on long data test set:
```
# With data_files["test"] = test_last_100.json

python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-70b-hf  --evaluation --evaluation_model_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/models/Llama-2-70b-hf/mimic-iv-note-di-bhc_Llama-2-70b-hf_long_data_100_valid_done/lora_rank_32_lora_alpha_32_lora_dropout_0.05_num_target_modules_2_learning_rate_2e-4/best_val_loss --data_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset --output_path /home/s/s_hegs02/scratch/debug --num_train_examples 100 --num_val_examples 100 --num_test_examples 100


{'rouge1': 40.5555217296232, 'rouge2': 14.380567615873911, 'rouge3': 6.145952549654063, 'rouge4': 2.9839005077717538, 'rougeL': 25.812246308651094, 'words': 76.46, 'bert_score': 86.00773721933365, 'bert_score_deberta-large': 61.7718161046505, 'sari': 45.00554637223204}
{'rouge1': 40.54058585622222, 'rouge2': 14.166953530916869, 'rouge3': 5.695409156176871, 'rouge4': 2.3296289556276344, 'rougeL': 26.20942324367626, 'words': 80.28, 'bert_score': 86.76197403669357, 'bert_score_deberta-large': 62.13716307282448, 'sari': 44.9823358329503}
{'rouge1': 40.34610201932814, 'rouge2': 14.411568668138443, 'rouge3': 6.339125512527877, 'rouge4': 2.953904152985041, 'rougeL': 26.32607735907873, 'words': 75.4, 'bert_score': 86.87447029352188, 'bert_score_deberta-large': 62.064355462789536, 'sari': 45.71918948150783}
{'rouge1': 40.1893462851726, 'rouge2': 14.055680359041899, 'rouge3': 5.956359809720371, 'rouge4': 2.5952588267124463, 'rougeL': 25.947485210072838, 'words': 74.67, 'bert_score': 85.03679966926575, 'bert_score_deberta-large': 61.02019691467285, 'sari': 44.80799764604085}
{'rouge1': 41.26608906368885, 'rouge2': 14.527248259553481, 'rouge3': 6.298794377100548, 'rouge4': 2.8508339445193314, 'rougeL': 26.633192382147357, 'words': 77.67, 'bert_score': 86.83461207151413, 'bert_score_deberta-large': 62.4643184542656, 'sari': 45.28758563082025}
```


#### 4000_600_chars

* Best model uses only _steps=20_

Evaluation on 4000_600_chars test set:
```
# With data_files["test"] = test_4000_600_chars_last_100.json

python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-70b-hf  --evaluation --evaluation_model_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/models/Llama-2-70b-hf/mimic-iv-note-di-bhc_Llama-2-70b-hf_4000_600_chars_100_valid_done/lora_rank_32_lora_alpha_32_lora_dropout_0.1_num_target_modules_2_learning_rate_2e-4/best_val_loss --data_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset --output_path /home/s/s_hegs02/scratch/debug --num_train_examples 100 --num_val_examples 100 --num_test_examples 100

{'rouge1': 42.1991438443541, 'rouge2': 13.563337998083213, 'rouge3': 5.7237180188916845, 'rouge4': 2.6722184607364134, 'rougeL': 24.777895343443596, 'words': 111.31, 'bert_score': 87.0364066362381, 'bert_score_deberta-large': 61.91603672504426, 'sari': 44.16604762927569}
{'rouge1': 41.67500471009928, 'rouge2': 13.840969941914366, 'rouge3': 5.947293259680258, 'rouge4': 2.844910436528473, 'rougeL': 24.836899438706126, 'words': 120.9, 'bert_score': 85.90889036655426, 'bert_score_deberta-large': 60.57257956266403, 'sari': 44.07321356056379}
{'rouge1': 41.88252074463912, 'rouge2': 14.074882768448884, 'rouge3': 6.039503368150073, 'rouge4': 2.8148856591135196, 'rougeL': 25.643550342550594, 'words': 112.34, 'bert_score': 87.158194065094, 'bert_score_deberta-large': 62.24099487066269, 'sari': 44.35337536946284}
{'rouge1': 41.83728919015555, 'rouge2': 13.09889608467368, 'rouge3': 5.487954667140488, 'rouge4': 2.4686725964920826, 'rougeL': 24.435506275256007, 'words': 112.07, 'bert_score': 86.0190686583519, 'bert_score_deberta-large': 61.00947117805481, 'sari': 43.20317982433489}
{'rouge1': 41.5120357352563, 'rouge2': 13.574013168402391, 'rouge3': 5.627399063893296, 'rouge4': 2.4955528332014443, 'rougeL': 24.480105189357122, 'words': 113.79, 'bert_score': 86.04067796468735, 'bert_score_deberta-large': 60.975450932979584, 'sari': 43.51100259877742}
```


#### Train on hallucination dataset

Important Parameters:
* Set max_steps to 20 (best in parameter tuning)
* Set gradient_accumulation_steps to 16

Set training data (`data_files["train"]`) depending on setting:
* Original: `hallucination_summaries_original_test_100.json`
* Cleaned: `hallucination_summaries_cleaned_test_100.json`
* Cleaned and Improved: `hallucination_summaries_cleaned_improved_test_100.json`

Careful: Set `output_path` accordingly for each setting!

```
CUDA_VISIBLE_DEVICES=0,1 python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-70b-hf --data_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset --output_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/models/Llama-2-70b-hf/mimic-iv-note-di-bhc_Llama-2-70b-hf_4000_600_chars_100_hallucinations/original --device cuda --max_steps 20 --save_and_logging_steps 20 --batch_size 1 --gradient_accumulation_steps 16 --lora_rank 32 --lora_alpha 32 --lora_dropout 0.1 --num_target_modules 2 --learning_rate 2e-4 --num_train_examples 100 --num_val_examples 100 --num_test_examples 100
```

Prediction on hallucination dataset:

Careful: Set `data_files["train"] = test_4000_600_chars_last_50.json` for each setting!

```
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py --model_name_or_path meta-llama/Llama-2-70b-hf  --evaluation --evaluation_model_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/models/Llama-2-70b-hf/mimic-iv-note-di-bhc_Llama-2-70b-hf_4000_600_chars_100_hallucinations/original/checkpoint-20 --data_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/dataset --output_path /home/s/s_hegs02/scratch/mimic-iv-note-di-bhc/models/Llama-2-70b-hf/mimic-iv-note-di-bhc_Llama-2-70b-hf_4000_600_chars_100_hallucinations/original_50_test --num_train_examples 100 --num_val_examples 100 --num_test_examples 50
```