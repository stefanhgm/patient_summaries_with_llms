import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import fire
import guidance
import yaml
from tqdm import tqdm

ALL_PROMPTS = {
    "prompt_1": """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
You will be given a doctor's note and you will need to summarize the patient's brief hospital course.

Let's do a practice round.
{{~/user}}

{{#assistant~}}
Sounds great!
{{~/assistant}}

{{#each icl_examples}}
{{#user}}Here is the doctor's note on a patient's brief hospital course:

{{this.text}}

Summarize for the patient what happened during the hospital stay based on this doctor's note. Please make it short and concise and only include key events and findings. 
{{/user}}
{{#assistant}}
{{this.summary}}
{{/assistant}}
{{/each}}


{{#user~}}
Here is the doctor's note on a patient's brief hospital course:

{{final_text}}

Summarize for the patient what happened during the hospital stay based on this doctor's note. Please make it short and concise and only include key events and findings. 
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
""",
    "prompt_2": """
{{#system~}}
You are helping with a resident working at a large urban academic medical center.
{{~/system}}

{{#user~}}
You task is to help summarize a patient's brief hospital course based on the doctor's note. Please make it short and concise and only include key events and findings. 

Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

SUMMARY: 
{{this.summary}}
{{/each}}

Here is another doctor note on a patient's brief hospital course:

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
""",
    "prompt_3": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given some doctor's notes and you will need to summarize the patient's brief hospital course in one paragraph. Please only include key events and findings and avoid using medical jargons, and you MUST start the summary with "You were admitted".

{{#if icl_examples}}
Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

SUMMARY: 
{{this.summary}}
{{/each}}
{{/if}}

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
    """,
    "prompt_3.1": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given some doctor's notes and you will need to summarize the patient's brief hospital course in ONE paragraph with a few sentences. Please only include key events and findings and avoid using medical jargons, and you MUST start the summary with "You were admitted".

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' max_tokens=600 temperature=0}}
{{~/assistant}}
    """,
}


def read_jsonl(file_name):
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(file_name, data):
    with open(file_name, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_oai_model(model_name, max_calls_per_min=60):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    common_kwargs = {
        "max_calls_per_min": max_calls_per_min,  # Maximum number of calls that can be made per minute (default is 60)
    }

    assert model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-16k"]

    if config["openai_api_mode"] == "openai":
        os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
        model = guidance.llms.OpenAI(
            model_name, **common_kwargs, organization=config["openai_organization"]
        )
    elif config["openai_api_mode"] == "azure":
        deployment_id = model_name
        if model_name == "gpt-3.5-turbo":
            deployment_id = "gpt-35-turbo"
        elif model_name == "gpt-3.5-turbo-16k":
            deployment_id = "gpt-35-turbo-16k"

        model = guidance.llms.OpenAI(
            model_name,
            api_type="azure",
            api_key=config["azure_api_key"],
            api_base=config["azure_api_base"],
            api_version=config["azure_api_version"],
            deployment_id=deployment_id,
            **common_kwargs,
        )
    return model


def run_summarization(
    task_id: int,
    prompt_id: int,
    model_name: str,
    n_shot: int,
    save_path: Optional[str] = None,
    what_for: str = "exp",
    verbose: bool = False,
    debug: bool = False,
):
    demonstrations = read_jsonl(
        f"summarization_data/{what_for}_{task_id}_in-context.json"
    )
    test_examples = read_jsonl(f"summarization_data/{what_for}_{task_id}_test.json")

    bad_demonstration_ids = []
    for i, demonstration in enumerate(demonstrations):
        if demonstration["summary"].startswith("He came to the"):
            bad_demonstration_ids.append(i)

    assert len(demonstrations) >= n_shot
    if n_shot < len(demonstrations):
        random.seed(32)
        indices = list(range(len(demonstrations)))
        random.shuffle(indices)
        indices = [i for i in indices if i not in bad_demonstration_ids]
        icl_examples = [demonstrations[i] for i in indices[:n_shot]]
    else:
        icl_examples = demonstrations

    llm = load_oai_model(model_name)

    used_prompt = ALL_PROMPTS[f"prompt_{prompt_id}"]
    summarization_program_nshot = guidance(used_prompt)

    if verbose:
        print(f"Using {len(icl_examples)} ICL examples")
        print(icl_examples)

    if debug:
        test_examples = test_examples[:10]

    if save_path is None:
        save_path = f"summarization_results/{model_name}_{what_for}{task_id}_results_prompt{prompt_id}_{n_shot}shot.jsonl"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(save_path.replace(".jsonl", "_icl.jsonl"), icl_examples)

    with open(save_path.replace(".jsonl", "_prompt.txt"), "w") as f:
        f.write(used_prompt)

    failure_indices = []
    all_results = []

    for example_idx in tqdm(range(len(test_examples))):
        example = test_examples[example_idx]

        gen_answer = summarization_program_nshot(
            icl_examples=icl_examples,
            final_text=example["text"],
            llm=llm,
            verbose=verbose,
        )

        try:
            summary = gen_answer["summary"]
        except:
            print(f"Failed to generate answer for example {example_idx}")
            summary = ""
            failure_indices.append(example_idx)

        all_results.append(
            {
                "index": example_idx,
                "question": example["text"],
                "summary": summary,
            }
        )
        if verbose:
            print(f"Text: {example['text']}")
            print(f"Summary: {summary}")
            print("=====================================")

    write_jsonl(save_path, all_results)

    with open(save_path.replace(".jsonl", "_failures.json"), "w") as f:
        json.dump(failure_indices, f, indent=2)


if __name__ == "__main__":
    fire.Fire(run_summarization)
