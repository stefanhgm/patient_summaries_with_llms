import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel
import evaluate
from rouge_score import rouge_scorer
import wandb

transformers.logging.set_verbosity_info()


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter tuning of LLM on mimic summarization.")
    parser.add_argument(
        "--model_name_or_path", type=str
    )
    parser.add_argument(
        "--data_path", type=str
    )
    parser.add_argument(
        "--output_path", type=str
    )
    parser.add_argument(
        "--device", type=str, default="cuda"
    )
    parser.add_argument(
        "--evaluation", action="store_true"
    )
    parser.add_argument(
        "--evaluation_model_path", type=str, default=None
    )
    
    
    # General parameters
    parser.add_argument(
        "--num_train_examples", type=int, default=None
    )
    parser.add_argument(
        "--num_val_examples", type=int, default=None
    )
    parser.add_argument(
        "--num_test_examples", type=int, default=None
    )
    parser.add_argument(
        "--max_steps", type=int, default=100
    )
    parser.add_argument(
        "--save_and_logging_steps", type=int, default=10
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16
    )
    
    # Parameters to tune
    parser.add_argument(
        "--lora_rank", type=int, default=8
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=8
    )   
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1
    )
    parser.add_argument(
        "--num_target_modules", type=int, default=4
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4
    )
    args = parser.parse_args()
    return args


# Use custom rouge function to obtain rouge 3/4 which are not available in huggingface
def get_rouge_score(gold, pred):
    rouge_scores = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_scores, use_stemmer=True)
    scores = scorer.score(gold, pred)
    return {k: scores[k].fmeasure * 100 for k in rouge_scores}

def compute_custom_metrics(srcs, golds, preds, device):
    scores = defaultdict(list)
    bertscore = evaluate.load("bertscore")
    sari = evaluate.load("sari")
    
    # For rouge and length go over examples one by one and determine mean
    for gold, pred in zip(golds, preds):
        for k, v in get_rouge_score(gold, pred).items():
            scores[k].append(v)
        scores['words'].append(len(pred.split(' ')))
    for k, v in scores.items():
        scores[k] = np.mean(v)

    # This is the default call using model_type="roberta-large"
    # This is the same as in the paper "Generation of Patient After-Visit Summaries to Support Physicians" (AVS_gen/eval_summarization.py) using the libary SummerTime
    scores['bert_score'] = np.mean((bertscore.compute(predictions=preds, references=golds, lang="en", device=device))['f1']) * 100
    # BERTScore authors recommend "microsoft/deberta-large-mnli" (https://github.com/Tiiiger/bert_score)
    scores['bert_score_deberta-large'] = np.mean((bertscore.compute(predictions=preds, references=golds, device=device, model_type="microsoft/deberta-large-mnli"))['f1']) * 100
    scores['sari'] = sari.compute(sources=srcs, predictions=preds, references=[[g] for g in golds])['sari']
    # scores['sari'] = scores['sari'][0]
    # Importing readability for dallc score not working: https://pypi.org/project/py-readability-metrics/    

    return scores

def print_metrics_as_latex(metrics):
    # Print latex table row
    order = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'bert_score', 'bert_score_deberta-large', 'sari', 'words']
    print(' & '.join([f'${metrics[k]:.2f}$' for k in order]))
        

def main():
    # Code based on summarization mimic iv notebook
    args = parse_args()
    output_dir = args.output_path
    
    print('data:', args.data_path)
    
    # Load device
    # device = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    # For manual run with CUDA_VISIBLE_DEVICES use "cuda:0" as device
    device = "cuda"
    
    # Log parameter tuning
    target_modules = {1: ['q_proj'], 2: ['q_proj', 'v_proj'], 4: ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
    lora_config = LoraConfig(
        r=args.lora_rank,
        # Scaling lora weight matrix by alpha/r (in paper: used equals r, helps to return hyperparams when varying r)
        # So effectively tuning alpha similar to tuning LR
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules[args.num_target_modules],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Register wandb
    # To disable uploading of output.log (possible information leakage)
    # os.environ["WAND_CONSOLE"] = "off"
    # os.environ["WANDB_IGNORE_GLOBS"] = "output.log"
    short_model_name = args.model_name_or_path.split('/')[-1]
    if args.evaluation:
        wandb.init(project=f"mimic-iv-note-di-bhc_{short_model_name}_evaluation", config=args, tags=[], notes="")
    else:
        wandb.init(project=f"mimic-iv-note-di-bhc_{short_model_name}_parameter-tuning", config=args, tags=[], notes="")
        
    # Load model
    hf_token = ''
    model_name = args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True,
                                                # For manual run with CUDA_VISIBLE_DEVICES use "cuda:0" as device_map
                                                device_map='auto',  # can be specific with device, but if CUDA_VISIBLE_DEVICES is set, auto should work
                                                token=hf_token,
                                                )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)  # Padding size for batched prediction from HF warning
    # Workaround for missing padding token
    # From https://github.com/huggingface/transformers/issues/22312 (comment Jun 28)
    # tokenizer.pad_token='[PAD]' 
    # assert tokenizer.pad_token_id == 0
    
    
    # Read data
    data_path = args.data_path

    data_files = {}
    extension = "json"
    # Model trained on shortened texts and summaries
    data_files["train"] = data_path + '/train_4000_600_chars_251-350_pt.json'
    data_files["validation"] = data_path + '/valid_4000_600_chars.json'
    data_files["test"] = data_path + '/valid_4000_600_chars.json'
    # For testing
    # data_files["test"] = data_path + '/test_4000_600_chars_last_100.json'

    # Model trained on full texts and summaries
    # data_files["train"] = data_path + '/train.json'
    # data_files["validation"] = data_path + '/valid.json'
    # data_files["test"] = data_path + '/valid.json'
    
    print(f"Loading data from {data_files}")
    data = load_dataset(extension, data_files=data_files)

    # data = load_dataset("samsum")
    data_train, data_test, data_val = data["train"], data["test"], data["validation"]

    # Limit number of examples
    # Take examples from the end to prevent data leake with experiments done on beginning examples
    if args.num_train_examples:
        data_train = data_train.select(range(0, len(data_train))[-args.num_train_examples:])
    if args.num_val_examples:
        data_val = data_val.select(range(0, len(data_val))[-args.num_val_examples:])
    if args.num_test_examples:
        data_test = data_test.select(range(0, len(data_test))[-args.num_test_examples:])
    
    print(f"Number of training examples: {len(data_train)}")
    print(f"Number of validation examples: {len(data_val)}")
    print(f"Number of test examples: {len(data_test)}")

    # # Include ':' for texts to exclude ':' from being treated as labels to train on
    # # Because using these texts later for the collator
    
    # Default
    instruction_text = "Summarize for the patient what happened during the hospital stay based on this doctor's note:"
    response_text = "Summary for the patient:"
    
    # Experiment (prompt 2 from slide)
    # instruction_text = "Summarize this clinical note for the patient in simple English. Only use the information provided in the clinical note itself and general medical knowledge or advice.\n\nClinical note:"
    # response_text = "Patient summary:"

    def generate_prompt(reference, summary=None, eos_token="</s>"):
        # Default
        instruction = f"{instruction_text}\n"
        input = f"{reference}\n"
        
        # New
        # instruction = f"{instruction_text} "
        # input = f"{reference}\n\n"

        response = f"{response_text} {summary + ' ' + eos_token if summary else ''} "
        return ''.join([instruction, input, response])
    
    def truncate_text(example, tokens=4096):
        # Crop sentences of the text at the end until the prompt has less than 4096 tokens
        # This is to avoid truncation of the prompt
        example['truncated'] = False
        while tokenizer(generate_prompt(example["text"], example["summary"]), return_tensors="pt")["input_ids"].shape[1] >= tokens:
            example["text"] = example["text"].rsplit('.', 1)[0]
            example['truncated'] = True
        return example
        
    # Truncate texts
    data_train = data_train.map(truncate_text)
    data_val = data_val.map(truncate_text)
    data_test = data_test.map(truncate_text)
    # Check how manyt texts are truncated
    print(f"Truncated texts: {sum(data_train['truncated'])} train, {sum(data_val['truncated'])} val, {sum(data_test['truncated'])} test")
    
    print("Prompted train example:")
    print(generate_prompt(data_train[0]["text"], data_train[0]["summary"]))
    print()
    print("Prompted valid example:")
    print(generate_prompt(data_val[0]["text"], data_val[0]["summary"]))


    # Prediction method
    def predict(prediction_model, dataset):
        predictions = []

        for ex in dataset:
            input_prompt = generate_prompt(ex["text"])
            input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to(device)

            with torch.cuda.amp.autocast():
                generation_output = prediction_model.generate(
                    input_ids=input_tokens,
                    # 98% CI of summaries is 322 tokens, 99% CI is 370 tokens
                    max_new_tokens=350,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            prediction = prediction[len(input_prompt):].strip()
            predictions.append(prediction)
            
        return predictions

    # Predict validation examples and compute metrics
    # predictions_val = predict(data_val)
    # compute_custom_metrics(data_val["text"], data_val["summary"], predictions_val)

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    
    if args.evaluation:
        print(f"Loading model for evaluation: {args.evaluation_model_path}")
        trained_model = PeftModel.from_pretrained(model, args.evaluation_model_path, torch_dtype=torch.float16).to(device)

    else: 
        # Training
        # Loading in 8 bit ..."
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        

        # Training setup
        training_args = transformers.TrainingArguments(
                    # To obtain decoded generations in compute_metrics one can use predict_with_generate=True for Seq2SeqArguments, but here no effect
                    # See used in summarization example and then possible to get rouge in compute_metrics: https://huggingface.co/docs/transformers/tasks/summarization
                    output_dir=output_dir,
                    per_device_train_batch_size=args.batch_size, 
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    per_device_eval_batch_size=args.batch_size,
                    eval_accumulation_steps=args.gradient_accumulation_steps,
                    evaluation_strategy="steps",
                    max_steps=args.max_steps,
                    save_steps=args.save_and_logging_steps, # for laod_best_model_at_end save_steps must be multiple of eval_steps
                    logging_steps=args.save_and_logging_steps, # eval_steps defaults to this
                    load_best_model_at_end=True,
                    # metric_for_best_model defaults to loss, rouge not possible since no decoding in compute_metrics
                    optim = "paged_adamw_32bit",
                    # no clear evidence for lr scheduler:
                    # * original llama and some sources use cosine (however llama with minimum of 0.1)
                    # * often when paged_adamw is used, constant is used (strongest evidence: https://www.philschmid.de/instruction-tune-llama-2)
                    lr_scheduler_type="constant",
                    learning_rate=args.learning_rate,
                    max_grad_norm=0.3,
                    warmup_ratio=0.03,
                    group_by_length=True,
                    ddp_find_unused_parameters=False,
                    report_to="wandb",
                )


        # Training
        def formatting_func(example):
            output = []
            for t, s in zip(example["text"], example["summary"]):
                prompt = generate_prompt(t, s)
                output.append(prompt)
            return output

        response_template_with_context = '\n' + response_text  # We added context here: "\n". This is enough for this tokenizer
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=model,
            train_dataset=data_train,
            eval_dataset=data_val,
            peft_config=lora_config,
            formatting_func=formatting_func,
            max_seq_length=4096, # 1024 so far, important to set high enough to avoid truncation
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_custom_metrics, # compute_metrics only get logits but no decoding and choosing argmax gives wrong results
            # Consider adding NEFTune here
            # neftune_noise_alpha=5,
            args=training_args,
        )

        # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

        trainer.train()
        trainer.save_model(f"{output_dir}/best_val_loss") 
        trained_model = trainer.model
    
    # Get validation score
    # print("\nPredicting validation examples.")
    # predictions_val = predict(trained_model, data_val)
    # metrics_val = compute_custom_metrics(data_val["text"], data_val["summary"], predictions_val, device)
    # print("Validation metrics:")
    # print(metrics_val)
    
    print("\nPredicting test examples.")
    predictions_test = predict(trained_model, data_test)
    metrics_test = compute_custom_metrics(data_test["text"], data_test["summary"], predictions_test, device)
    # Print test examples text and summary fields
    print("Test examples:")
    for i in range(min(len(data_test), 10)):
        print(f"\n\n\nExample {i}:\n\n{data_test[i]['text']}\n\n{data_test[i]['summary']}\n\n{predictions_test[i]}")
        
    # Store predictions into jsonl file as dict with keys: summary
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'predictions_test.jsonl', "w") as f:
        f.write("\n".join(predictions_test))

    with open(output_path / 'predictions_test_dict.jsonl', "w") as f:
        for pred in predictions_test:
            f.write(f'{{"summary": "{pred}"}}\n')

    print("Test metrics:")
    print(metrics_test)
    metrics_test = {k: round(v, 2) for k, v in metrics_test.items()}
    wandb.log(metrics_test)
    
    print("Test metrics rounded:")
    print(metrics_test)
    print_metrics_as_latex(metrics_test)
    wandb.finish()


if __name__ == "__main__":
    main()
