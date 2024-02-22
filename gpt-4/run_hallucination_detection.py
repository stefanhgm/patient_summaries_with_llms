import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import fire
import guidance
import Levenshtein
import yaml
from nervaluate import Evaluator
from nltk.tokenize import wordpunct_tokenize
from run_summarization import load_oai_model, read_jsonl, write_jsonl
from tqdm import tqdm

HALLUCINATION_LABELS = {
    "c": "condition_unsupported",
    "p": "procedure_unsupported",
    "m": "medication_unsupported",
    "t": "time_unsupported",
    "l": "location_unsupported",
    "n": "number_unsupported",
    "na": "name_unsupported",
    "w": "word_unsupported",
    "o": "other_unsupported",
    "co": "contradicted_fact",
    "i": "incorrect_fact",
}

OVERALL_HALLUCINATION_LABEL_NAME = "hallucination"

DATASET_PATHS = {
    "valid_mimic": "hallucination_detection_data/hallucinations_10_valid_mimic_agreed.jsonl",
    "valid_mimic_v2": "hallucination_detection_data/icl_examples.jsonl",  # check the other code #TODO
    "test_mimic": "hallucination_detection_data/hallucinations_100_mimic_agreed.jsonl",
    "test_generated": "hallucination_detection_data/hallucinations_100_generated_agreed.jsonl",
}


def create_icl_example_v1(
    ex: Dict[str, Any],
    add_hallucination_type: bool = False,
    hallucination_tag_label: str = "error",  # hallucinate
) -> str:
    sorted_labels = sorted(ex["labels"], key=lambda item: item["start"], reverse=True)
    summary = ex["summary"]

    for label in sorted_labels:
        start = label["start"]
        end = label["end"]
        hallucination_label = label["label"]
        if add_hallucination_type:
            summary = (
                summary[:start]
                + f'<{hallucination_tag_label} class="{hallucination_label}">'
                + summary[start:end]
                + f"</{hallucination_tag_label}>"
                + summary[end:]
            )
        else:
            summary = (
                summary[:start]
                + f"<{hallucination_tag_label}>"
                + summary[start:end]
                + f"</{hallucination_tag_label}>"
                + summary[end:]
            )
    return summary


# PROMPT VERSIONS
# V1: simple 0-shot detection
# V2: documentation + cot + n-shot (with imperfect examples)
# V3: documentation + class explanation + cot + n-shot (with improved examples)
# V3.1: V3 + prompt + class-aware prediction
ALL_PROMPTS = {
    "prompt_v1": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given a doctor's notes and a summary with potentially incorrectness. Your task is to identify spans with erroneous, contradictory, or unsupported facts in the summary, and label them using the <error> tag (e.g. <error>incorrect fact</error>). There could be more than one error in the summary. 

{{#if icl_examples}}
Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

ORIGINAL SUMMARY: 
{{this.summary}}

SUMMARY WITH LABELED ERRORS:
{{this.labeled_summary}}
{{/each}}
{{/if}}
 

Can you identify the errors for the following document and summary?
DOCUMENT: 
{{final_text}}

ORIGINAL SUMMARY: 
{{final_summary}}

SUMMARY WITH LABELED ERRORS:
{{~/user}}

{{#assistant~}}
{{gen 'labeled_errors' max_tokens=600 temperature=0}}
{{~/assistant}}
    """,
    "prompt_v2": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
We will present you with a pair of a brief hospital course (BHC) and a patient after visit summary (AVS). The AVS is also referred to as discharge summary. The BHC contains a detailed summary of the hospital stay written by medical service. It usually contains medical jargon, and it can follow different structures based on the hospital course and responsible medical specialty. The AVS summarizes the hospital stay for the patient in plain language. In practice, the BHC is not the only source of information to write the AVS. However, in our setting we treat the BHC as the only context for the summary.

For this labeling task, we are interested in errors in the AVS that are either unsupported by the BHC, contradict content in the BHC, or are wrong medical facts. We allow statements that contain general medical knowledge or advice that are often used in patient summaries. Most errors are due to unsupported facts, so we further distinguish those based on their specific content. This leads to the following error types or labels:
1. Unsupported facts, including condition/procedure/medication/time/location/number/name/word/other
2. Contradicted fact
3. Incorrect fact
And below is the detailed guideline, and we label error spans with the <error> tag (e.g. <error>incorrect fact</error>).

### Allowed General Medical Knowledge and Medical Advice
We allow general medical knowledge and advice that is often part of the AVS. Usually, these are information that are not specific for the hospital course given in the BHC. For example
- "Please take your medications as prescribed" contains no error even though the BHC does not contain this instruction because this is general medical advice.
- "If the symptoms get worse, please contact your doctor" contains no error even when the BHC does not contain this fact, since it is general medical knowledge that a doctor should be seen for worsening symptoms. 

### Determining Span of Errors
We label the smallest possible consecutive span that specifies the error given the BHC as a context. Removing further parts from the span would remove important information. A useful heuristic is to identify the minimal span that must be replaced to obtain a correct statement that is grammatically correct. For example
- "We performed an <error>esophageal-gastro-duodenoscopy (EGD).<error>" when no such procedure is reported in the BHC. The article "an" is not labeled as an error. When no procedure at all was performed "performed an esophageal-gastro-duodenoscopy (EGD)" should be labeled as error because there is no suitable substitute for "esophageal-gastro-duodenoscopy (EGD)".
- "After the surgery, we <error>transitioned you to oral oxycodone</error>." when the BHC contains no information for such a transition. If another medication transition is mentioned in the BHC and makes sense in this sentence only "oral oxycodone" should be labeled. If another oral medication transition is mentioned in the BHC only "oxycodone" should be labeled.
- "<error>Your symptoms responded well</error>." when no part of the sentence makes sense in the given context of the AVS.

### Dealing with Deidentified Information
The data contains deidentified information shown with "___" in the text. We always treat this as non-existent information. So, the annotators should not infer what the deidentified information could be. In general, deidentified fields in the AVS should not be labeled as errors. However, sometimes they belong to a wrong statement or clearly contain unsupported information (e.g., a doctor's name or phone numbers) that are not given in the BHC. In these cases, deidentified fields should be included in the error span. For example
- "Take ___ <error>200mg daily</error> and try to rest" when no such dosage information is provided in the BHC, but the statement to rest. The deidentified medication name is excluded from the error span.
- "Please avoid going up <error>more than ___ stairs</error> at a time" when restrictions for the number of stairs taken at a time are note mentioned in the BHC.
- "<error>Dr. ___ will follow up with you</error>" when no follow-up is mentioned in the BHC.
- "Please stop taking Aspirin <error>on ___</error>" when no stopping date is given in the BHC. 
- "Your RBC peaked <error>at ___ million</error>" if there is no hint of a specific red blood cell count given in the BHC.

### One Error per Span
To get reliable error counts a span should only contain a single error.
- "You received <error>Tylenol</error> and <error>Ciprofloxacin</error>" when there is no evidence in the BHC that the two medications were administered to the patient.
- "You have a <error>follow-up appointment with your PCP</error> and <error>your cardiologist</error>" when no such follow up is mentioned in the BHC. Both errors are labeled separately.

{{#each icl_examples}}
### Example {{this.index}}

BHC: 
{{this.text}}

AVS: 
{{this.summary}}

ERRORS:
{{this.cot_description}}

AVS WITH ERRORS LABELED:
{{this.summary_with_errors}}

{{/each}}

### Example {{n_shot+1}}

BHC: 
{{final_text}}

AVS: 
{{final_summary}}

ERROR:
{{~/user}}

{{#assistant~}}
{{gen 'labeled_errors' max_tokens=1200 temperature=0}}
{{~/assistant}}
""",
    "prompt_v3": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
We will present you with a pair of a brief hospital course (BHC) and a patient after visit summary (AVS). The AVS is also referred to as discharge summary. The BHC contains a detailed summary of the hospital stay written by medical service. It usually contains medical jargon, and it can follow different structures based on the hospital course and responsible medical specialty. The AVS summarizes the hospital stay for the patient in plain language. In practice, the BHC is not the only source of information to write the AVS. However, in our setting we treat the BHC as the only context for the summary.

## Instructions

For this labeling task, we are interested in errors in the AVS that are either unsupported by the BHC, contradict content in the BHC, or are wrong medical facts. We allow statements that contain general medical knowledge or advice that are often used in patient summaries. Most errors are due to unsupported facts, so we further distinguish those based on their specific content. This leads to the following error types or labels:
1. Unsupported facts, including condition/procedure/medication/time/location/number/name/word/other
2. Contradicted fact
3. Incorrect fact
And below is the detailed guideline, and we label error spans with the <error> tag (e.g. <error>incorrect fact</error>).

### Determining Span of Errors
We label the smallest possible consecutive span that specifies the error given the BHC as a context. Removing further parts from the span would remove important information. A useful heuristic is to identify the minimal span that must be replaced to obtain a correct statement that is grammatically correct. For example
- "We performed an <error>esophageal-gastro-duodenoscopy (EGD).<error>" when no such procedure is reported in the BHC. The article "an" is not labeled as an error. When no procedure at all was performed "performed an esophageal-gastro-duodenoscopy (EGD)" should be labeled as error because there is no suitable substitute for "esophageal-gastro-duodenoscopy (EGD)".
- "After the surgery, we <error>transitioned you to oral oxycodone</error>." when the BHC contains no information for such a transition. If another medication transition is mentioned in the BHC and makes sense in this sentence only "oral oxycodone" should be labeled. If another oral medication transition is mentioned in the BHC only "oxycodone" should be labeled.
- "<error>Your symptoms responded well</error>." when no part of the sentence makes sense in the given context of the AVS.

We allow general medical knowledge and advice that is often part of the AVS. Usually, these are information that are not specific for the hospital course given in the BHC. For example
- "Please take your medications as prescribed" contains no error even though the BHC does not contain this instruction because this is general medical advice.
- "If the symptoms get worse, please contact your doctor" contains no error even when the BHC does not contain this fact, since it is general medical knowledge that a doctor should be seen for worsening symptoms. 

We try to ignore grammatical errors in the BHC and AVS. If the original meaning can still be inferred (e.g. "medictaions" instead of "medications"), the most likely corrected form can be used. If the meaning cannot be inferred, they can be ignored in the BHC or labeled as Unsupported Other in the AVS.

If a sentence or phrase is repeated, then please treat it as you would any other sentence and highlight all errors (even if you did so in a previous sentence). For example
- "Please take Tylenol. Please take Tylenol" when Tylenol was prescribed in the BHC.
- "Limit your <error>use of stairs</error>. Please limit <error>use of stairs</error>" when movement was encouraged.

To get reliable error counts a span should only contain a single error.
- "You received <error>Tylenol</error> and <error>Ciprofloxacin</error>" when there is no evidence in the BHC that the two medications were administered to the patient.
- "You have a <error>follow-up appointment with your PCP</error> and <error>your cardiologist</error>" when no such follow up is mentioned in the BHC. Both errors are labeled separately.

### Dealing with Deidentified Information
The data contains deidentified information shown with "___" in the text. We always treat this as non-existent information. So, the annotators should not infer what the deidentified information could be. In general, deidentified fields in the AVS should not be labeled as errors. However, sometimes they belong to a wrong statement or clearly contain unsupported information (e.g., a doctor's name or phone numbers) that are not given in the BHC. In these cases, deidentified fields should be included in the error span. For example
- "Take ___ <error>200mg daily</error> and try to rest" when no such dosage information is provided in the BHC, but the statement to rest. The deidentified medication name is excluded from the error span.
- "Please avoid going up <error>more than ___ stairs</error> at a time" when restrictions for the number of stairs taken at a time are note mentioned in the BHC.
- "<error>Dr. ___ will follow up with you</error>" when no follow-up is mentioned in the BHC.
- "Please stop taking Aspirin <error>on ___</error>" when no stopping date is given in the BHC. 
- "Your RBC peaked <error>at ___ million</error>" if there is no hint of a specific red blood cell count given in the BHC.

### Error Types
In general, we ask for the most specific error that is applicable. If there is uncertainty which type applies, prefer the one mentioned first in the enumeration of all error types shown earlier. For instance, if the error contains an unsupported medication name, the Unsupported medication type should be used instead of the Unsupported name type. Here is a detailed description of the error types:
- `Unsupported Condition`: includes unsupported symptoms, diseases, or findings of the patient. For example
    - "You were found to have a <error>left clavicle fracture</error>" when no information was given for this condition in the BHC.
- `Unsupported Procedure`: includes any unsupported medical procedures. For example
    - "You had a <error>filter placed in your vein</error>" when no intervention with a filter was mentioned.
- `Unsupported Medication`: contains all errors related to unsupported medications. This includes medication classes, substances, routes, frequencies, and dosages. For example
    - "You were placed on <error>antibiotics</error>" when only blood thinners were prescribed.
- `Unsupported Time`: includes all errors for unsupported time or interval statements. For example
    - "Keep your arm in a sling for the <error>next 6 weeks</error>" when no specific duration is given.
- `Unsupported Location`: Locations include both unsupported physical places as well as regions of the patient. For example
    - "The patient was admitted to the <error>Acute Surgery Service</error>" when no admission location was provided in the BHC.
- `Unsupported Number`: any number either as digits or written that are unsupported. This also includes words such as "a" and "an". For example
    - "Your pacemaker rate was increased to <error>50</error>" when the rate of 50 is not given in the BHC.
- `Unsupported Name`: named entities that are not supported by the BHC. For example
    - "You were seen by the <error>interventional pulmonary service</error>" when no consult with this service was mentioned in the BHC.
- `Unsupported Word`: incorrect or inappropriate words or phrases which do not fit in any of the above types. For example
    - "We will send you home with a <error>drain</error> in place" when drain not mentioned in the BHC.
- `Unsupported Other`: If there is a mistake which clearly does not belong to any of the above categories, you may use this category as a last resort. We cannot give precise instructions because the "other" category is very broad.
- `Contradicted Fact`: This error type is independent of the content and contains all facts that clearly contradict information provided in the BHC. For example
    - "Your pacemaker rate was increased to <error>50</error>" when the context state a pacemaker rate of 40.
- `Incorrect Fact`: This error type is independent of the content and contains all facts that clearly contradict general medical knowledge or advice. For example
    - "We diagnosed a seizure, and you <error>can continue driving your car</error>" when no reason for allowing driving after a seizure is provided this contradict common medical knowledge.

## Examples

{{#each icl_examples}}
### Example {{this.index}}

BHC: 
{{this.text}}

AVS: 
{{this.summary}}

ERRORS:
{{this.cot_description}}

AVS WITH ERRORS LABELED:
{{this.summary_with_errors}}

{{/each}}

### Example {{n_shot+1}}

BHC: 
{{final_text}}

AVS: 
{{final_summary}}

ERROR:
{{~/user}}

{{#assistant~}}
{{gen 'labeled_errors' max_tokens=1200 temperature=0}}
{{~/assistant}}
""",
    "prompt_v3_labeled": """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
We will present you with a pair of a brief hospital course (BHC) and a patient after visit summary (AVS). The AVS is also referred to as discharge summary. The BHC contains a detailed summary of the hospital stay written by medical service. It usually contains medical jargon, and it can follow different structures based on the hospital course and responsible medical specialty. The AVS summarizes the hospital stay for the patient in plain language. In practice, the BHC is not the only source of information to write the AVS. However, in our setting we treat the BHC as the only context for the summary.

## Instructions

For this labeling task, we are interested in errors in the AVS that are either unsupported by the BHC, contradict content in the BHC, or are wrong medical facts. We allow statements that contain general medical knowledge or advice that are often used in patient summaries. Most errors are due to unsupported facts, so we further distinguish those based on their specific content. This leads to the following error types or labels:
1. Unsupported facts, including condition/procedure/medication/time/location/number/name/word/other
2. Contradicted fact
3. Incorrect fact
And below is the detailed guideline, and we label error spans with the <error> tag (e.g. <error class="error_type">incorrect fact</error>).

### Determining Span of Errors
We label the smallest possible consecutive span that specifies the error given the BHC as a context. Removing further parts from the span would remove important information. A useful heuristic is to identify the minimal span that must be replaced to obtain a correct statement that is grammatically correct. For example
- "We performed an <error>esophageal-gastro-duodenoscopy (EGD).<error>" when no such procedure is reported in the BHC. The article "an" is not labeled as an error. When no procedure at all was performed "performed an esophageal-gastro-duodenoscopy (EGD)" should be labeled as error because there is no suitable substitute for "esophageal-gastro-duodenoscopy (EGD)".
- "After the surgery, we <error>transitioned you to oral oxycodone</error>." when the BHC contains no information for such a transition. If another medication transition is mentioned in the BHC and makes sense in this sentence only "oral oxycodone" should be labeled. If another oral medication transition is mentioned in the BHC only "oxycodone" should be labeled.
- "<error>Your symptoms responded well</error>." when no part of the sentence makes sense in the given context of the AVS.

We allow general medical knowledge and advice that is often part of the AVS. Usually, these are information that are not specific for the hospital course given in the BHC. For example
- "Please take your medications as prescribed" contains no error even though the BHC does not contain this instruction because this is general medical advice.
- "If the symptoms get worse, please contact your doctor" contains no error even when the BHC does not contain this fact, since it is general medical knowledge that a doctor should be seen for worsening symptoms. 

We try to ignore grammatical errors in the BHC and AVS. If the original meaning can still be inferred (e.g. "medictaions" instead of "medications"), the most likely corrected form can be used. If the meaning cannot be inferred, they can be ignored in the BHC or labeled as Unsupported Other in the AVS.

If a sentence or phrase is repeated, then please treat it as you would any other sentence and highlight all errors (even if you did so in a previous sentence). For example
- "Please take Tylenol. Please take Tylenol" when Tylenol was prescribed in the BHC.
- "Limit your <error>use of stairs</error>. Please limit <error>use of stairs</error>" when movement was encouraged.

To get reliable error counts a span should only contain a single error.
- "You received <error>Tylenol</error> and <error>Ciprofloxacin</error>" when there is no evidence in the BHC that the two medications were administered to the patient.
- "You have a <error>follow-up appointment with your PCP</error> and <error>your cardiologist</error>" when no such follow up is mentioned in the BHC. Both errors are labeled separately.

### Dealing with Deidentified Information
The data contains deidentified information shown with "___" in the text. We always treat this as non-existent information. So, the annotators should not infer what the deidentified information could be. In general, deidentified fields in the AVS should not be labeled as errors. However, sometimes they belong to a wrong statement or clearly contain unsupported information (e.g., a doctor's name or phone numbers) that are not given in the BHC. In these cases, deidentified fields should be included in the error span. For example
- "Take ___ <error>200mg daily</error> and try to rest" when no such dosage information is provided in the BHC, but the statement to rest. The deidentified medication name is excluded from the error span.
- "Please avoid going up <error>more than ___ stairs</error> at a time" when restrictions for the number of stairs taken at a time are note mentioned in the BHC.
- "<error>Dr. ___ will follow up with you</error>" when no follow-up is mentioned in the BHC.
- "Please stop taking Aspirin <error>on ___</error>" when no stopping date is given in the BHC. 
- "Your RBC peaked <error>at ___ million</error>" if there is no hint of a specific red blood cell count given in the BHC.

### Error Types
In general, we ask for the most specific error that is applicable. If there is uncertainty which type applies, prefer the one mentioned first in the enumeration of all error types shown earlier. For instance, if the error contains an unsupported medication name, the Unsupported medication type should be used instead of the Unsupported name type. Here is a detailed description of the error types:
- `Unsupported Condition`: includes unsupported symptoms, diseases, or findings of the patient. For example
    - "You were found to have a <error class="unsupported_condition">left clavicle fracture</error>" when no information was given for this condition in the BHC.
- `Unsupported Procedure`: includes any unsupported medical procedures. For example
    - "You had a <error class="unsupported_procedure">filter placed in your vein</error>" when no intervention with a filter was mentioned.
- `Unsupported Medication`: contains all errors related to unsupported medications. This includes medication classes, substances, routes, frequencies, and dosages. For example
    - "You were placed on <error class="unsupported_medication">antibiotics</error>" when only blood thinners were prescribed.
- `Unsupported Time`: includes all errors for unsupported time or interval statements. For example
    - "Keep your arm in a sling for the <error class="unsupported_time">next 6 weeks</error>" when no specific duration is given.
- `Unsupported Location`: Locations include both unsupported physical places as well as regions of the patient. For example
    - "The patient was admitted to the <error class="unsupported_location">Acute Surgery Service</error>" when no admission location was provided in the BHC.
- `Unsupported Number`: any number either as digits or written that are unsupported. This also includes words such as "a" and "an". For example
    - "Your pacemaker rate was increased to <error class="unsupported_number">50</error>" when the rate of 50 is not given in the BHC.
- `Unsupported Name`: named entities that are not supported by the BHC. For example
    - "You were seen by the <error class="unsupported_name">interventional pulmonary service</error>" when no consult with this service was mentioned in the BHC.
- `Unsupported Word`: incorrect or inappropriate words or phrases which do not fit in any of the above types. For example
    - "We will send you home with a <error class="unsupported_word">drain</error> in place" when drain not mentioned in the BHC.
- `Unsupported Other`: If there is a mistake which clearly does not belong to any of the above categories, you may use this category as a last resort. We cannot give precise instructions because the "other" category is very broad.
- `Contradicted Fact`: This error type is independent of the content and contains all facts that clearly contradict information provided in the BHC. For example
    - "Your pacemaker rate was increased to <error class="contradicted_fact">50</error>" when the context state a pacemaker rate of 40.
- `Incorrect Fact`: This error type is independent of the content and contains all facts that clearly contradict general medical knowledge or advice. For example
    - "We diagnosed a seizure, and you <error class="incorrect_fact">can continue driving your car</error>" when no reason for allowing driving after a seizure is provided this contradict common medical knowledge.

## Examples

{{#each icl_examples}}
### Example {{this.index}}

BHC: 
{{this.text}}

AVS: 
{{this.summary}}

ERRORS:
{{this.cot_description}}

AVS WITH ERRORS LABELED:
{{this.summary_with_errors}}

{{/each}}

### Example {{n_shot+1}}

BHC: 
{{final_text}}

AVS: 
{{final_summary}}

ERROR:
{{~/user}}

{{#assistant~}}
{{gen 'labeled_errors' max_tokens=1200 temperature=0}}
{{~/assistant}}
""",
}

# COT EXAMPLE VERSIONS
# V0: List-like items
#
# ERRORS:
# - "Your <error>red blood cell count</error> was followed and was stable." the blood cell count is not mentioned in the BHC.
# - "You were treated with <error>2 days</error> of antibiotics which were stopped prior to discharge." the time is not mentioned in the BHC.
#
# V1: Richer descriptions and grouped errors
# TODO
#

# N-SHOT SELECTION NOTES
# initially we started three shot without any examples w/o hallucination errors
# we need to work a bit to include examples without hallucination errors


def run_hallucination_detection(
    task_name: str,
    prompt_id: int,
    model_name: str,
    n_shot: int,
    add_label: bool = False,
    no_cot: bool = False,
    icl_version: Optional[int] = None,
    save_path: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
):
    if prompt_id == 1:
        demonstrations = read_jsonl(DATASET_PATHS["valid_mimic"])
    elif prompt_id == 2:
        demonstrations = read_jsonl(DATASET_PATHS["valid_mimic_v2"])
    elif prompt_id == 3:
        demonstrations = read_jsonl(
            f"hallucination_detection_data/icl_v{icl_version}.jsonl"
        )

    test_examples = read_jsonl(DATASET_PATHS[task_name])

    if n_shot < len(demonstrations):
        random.seed(32)
        indices = list(range(len(demonstrations)))
        random.shuffle(indices)
        icl_examples = [demonstrations[i] for i in indices[:n_shot]]
    else:
        icl_examples = demonstrations

    llm = load_oai_model(model_name)

    if prompt_id == 1:
        used_prompt = ALL_PROMPTS[f"prompt_v{prompt_id}"]
        hallucination_detection_program = guidance(used_prompt)

        for ex in icl_examples:
            ex["labeled_summary"] = create_icl_example_v1(
                ex, add_hallucination_type=False
            )
    elif prompt_id == 2:
        used_prompt = ALL_PROMPTS[f"prompt_v{prompt_id}"]
        used_prompt = used_prompt.replace(
            "{{n_shot+1}}", str(n_shot + 1)
        )  # This feels a bit ad-hoc although I guess it makes everything easier for us...
        hallucination_detection_program = guidance(used_prompt)

        for idx, ex in enumerate(icl_examples):
            ex["summary_with_errors"] = create_icl_example_v1(
                ex, add_hallucination_type=False
            )
            ex["index"] = idx + 1

    elif prompt_id == 3:
        if add_label:
            used_prompt = ALL_PROMPTS[f"prompt_v{prompt_id}_labeled"]
        else:
            used_prompt = ALL_PROMPTS[f"prompt_v{prompt_id}"]
        used_prompt = used_prompt.replace(
            "{{n_shot+1}}", str(n_shot + 1)
        )  # This feels a bit ad-hoc although I guess it makes everything easier for us...

        if no_cot:
            used_prompt = used_prompt.replace(
                "ERRORS:\n{{this.cot_description}}\n\n", ""
            ).replace(
                "ERROR:\n{{~/user}}", "{{~/user}}"
            )  # The last error

        hallucination_detection_program = guidance(used_prompt)

        if add_label:
            for ex in demonstrations:
                ex["cot_description"] = ex["cot_description_with_label"]
                # Manually switch

        for idx, ex in enumerate(icl_examples):
            ex["summary_with_errors"] = create_icl_example_v1(
                ex, add_hallucination_type=add_label
            )
            ex["index"] = idx + 1

    if verbose:
        print(f"Using {len(icl_examples)} ICL examples")
        print(icl_examples)

    if debug:
        test_examples = test_examples[:5]

    if save_path is None:
        save_path = f"hallucination_detection_results/{model_name}_{task_name}_prompt{prompt_id}_icl{icl_version}_{n_shot}shot.jsonl"
        if add_label:
            save_path = save_path.replace(".jsonl", "_labeled.jsonl")
        if no_cot:
            save_path = save_path.replace(".jsonl", "_no_cot.jsonl")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(save_path.replace(".jsonl", "_icl.jsonl"), icl_examples)

    with open(save_path.replace(".jsonl", "_prompt.txt"), "w") as f:
        f.write(used_prompt)

    failure_indices = []
    all_results = []

    for example_idx in tqdm(range(len(test_examples))):
        example = test_examples[example_idx]

        gen_answer = hallucination_detection_program(
            icl_examples=icl_examples,
            final_text=example["text"],
            final_summary=example["summary"],
            llm=llm,
            verbose=verbose,
        )

        try:
            labeled_errors = gen_answer["labeled_errors"]
        except:
            print(f"Failed to generate labeled_errors for example {example_idx}")
            labeled_errors = ""
            failure_indices.append(example_idx)

        gold_labeled_errors = create_icl_example_v1(
            example, add_hallucination_type=False
        )
        all_results.append(
            {
                "index": example_idx,
                "text": example["text"],
                "summary": example["summary"],
                "gold_labels": example["labels"],
                "predicted_labeled_errors": labeled_errors,
                "gold_labeled_errors": gold_labeled_errors,
            }
        )
        if verbose:
            print(f"Text: {example['text']}")
            print(f"Gold Errors: {gold_labeled_errors}")
            print(f"Pred Errors: {labeled_errors}")
            print("=====================================")

    write_jsonl(save_path, all_results)

    with open(save_path.replace(".jsonl", "_failures.json"), "w") as f:
        json.dump(failure_indices, f, indent=2)

    process_hallucination_detection_results(save_path, add_label=add_label)


def parse_hallucination_detection_results(input_string, add_label=False):
    tag_pattern = re.compile(r"<(\w+)(.*?)(?=/)?>(.*?)</\1>")
    text_without_tags = input_string
    label_positions = []

    while True:
        match = tag_pattern.search(text_without_tags)
        if not match:
            break

        full_tag = match.group(0)
        tag_name = match.group(1)
        tag_attributes = match.group(2)
        tag_content = match.group(3)
        start_tag_length = len("<{}{}>".format(tag_name, tag_attributes))

        start_content, end_content = match.span(3)

        extracted_label = {
            "start": match.start(3) - start_tag_length,
            "end": match.start(3) - start_tag_length + len(tag_content),
            "orig_label": tag_attributes.strip(),
            "text": tag_content,
        }

        if add_label:
            matches = re.findall(r'class="([^"]*)"', tag_attributes)
            # print(full_tag, matches)
            if matches and len(matches) == 1:
                extracted_label["label"] = matches[0]

        label_positions.append(extracted_label)

        text_without_tags = (
            text_without_tags[: match.start()]
            + tag_content
            + text_without_tags[match.end() :]
        )

    return text_without_tags, label_positions


def find_substring_with_max_distance(s, substring, max_distance=3):
    len_sub = len(substring)
    best_match = (None, None, max_distance + 1)  # (start_index, end_index, distance)

    for start in range(len(s) - len_sub + 1):
        end = start + len_sub
        slice_ = s[start:end]
        distance = Levenshtein.distance(slice_, substring)

        if distance <= max_distance and distance < best_match[2]:
            best_match = (start, end, distance)

    return best_match[:-1] if best_match[2] <= max_distance else None


def align_pred_labels_to_gold(
    pred_labels,
    pred_summary,
    gold_summary,
    search_padding=8,
    max_edit_distance=3,
):
    aligned_labels = []

    for pred_label in pred_labels:
        label_text = pred_label["text"]
        cur_start, cur_end = pred_label["start"], pred_label["end"]
        search_summary_area = gold_summary[
            max(0, cur_start - search_padding) : min(
                len(pred_summary), cur_end + search_padding
            )
        ]
        best_match = find_substring_with_max_distance(
            search_summary_area, label_text, max_edit_distance
        )

        if best_match is not None:
            start, end = best_match
            start += cur_start - search_padding
            end += cur_start - search_padding
            searched_text = gold_summary[start:end]
            aligned_labels.append(
                {
                    "start": start,
                    "end": end,
                    "orig_label": pred_label["orig_label"],
                    "orig_text": pred_label["text"],
                    "label": pred_label.get("label", OVERALL_HALLUCINATION_LABEL_NAME),
                    "text": searched_text,
                }
            )
        else:
            warnings.warn(
                f"Failed to find a match for label {label_text} in the summary"
            )

    return aligned_labels


def character_labels_to_word_labels(text, labels):
    tokenizer = lambda x: wordpunct_tokenize(x)

    # Convert character level labels to word level labels
    new_labels = []
    for label in labels:
        new_label = {"label": label.get("label", OVERALL_HALLUCINATION_LABEL_NAME)}
        new_label["start"] = len(tokenizer(text[: label["start"]]))
        new_label["end"] = new_label["start"] + len(tokenizer(label["text"]))
        new_label["length"] = new_label["end"] - new_label["start"]
        # Copy over old text because not tokenized version, but check it contains same text without whitespace
        new_label["text"] = label["text"]
        # Debug
        # print(f"Old chars: {label['text'].replace(' ', '')} -> New chards: {''.join(tokenizer(text)[new_label['start']:new_label['end']])}")
        # print(f"Old label: {label} -> New label: {new_label}")

        if "".join(tokenizer(text)[new_label["start"] : new_label["end"]]) != label[
            "text"
        ].replace(" ", ""):
            print("".join(tokenizer(text)[new_label["start"] : new_label["end"]]))
            print(label["text"].replace(" ", ""))
        new_labels.append(new_label)
    return new_labels


def process_hallucination_detection_results(
    pred_path: str,
    save_path: Optional[str] = None,
    add_label: bool = False,
):
    all_predictions = read_jsonl(pred_path)

    for ex in all_predictions:
        cur_pred = ex["predicted_labeled_errors"]
        cur_source_ans = cur_pred.split("AVS WITH ERRORS LABELED:")
        if len(cur_source_ans) < 2:
            # We consider no error is detected in the summary
            ex["predicted_labels"] = []
            ex["pred_token_labels"] = []
        else:
            pred_labeled_summary = cur_source_ans[1].strip()
            pred_summary, pred_labels = parse_hallucination_detection_results(
                pred_labeled_summary, add_label
            )

            if pred_summary != ex["summary"]:
                # It seems there are some mismatches in the summary text
                # (possibly due to errors like typos in the original summary)
                # and we should redo the (localized) search
                pred_labels = align_pred_labels_to_gold(
                    pred_labels, pred_summary, ex["summary"]
                )

            # TODO: add labels in the future
            ex["predicted_labels"] = pred_labels
            ex["pred_token_labels"] = character_labels_to_word_labels(
                ex["summary"], ex["predicted_labels"]
            )
        ex["gold_token_labels"] = character_labels_to_word_labels(
            ex["summary"], ex["gold_labels"]
        )

    if save_path is None:
        save_path = pred_path.replace(".jsonl", "_processed.jsonl")

    write_jsonl(save_path, all_predictions)


if __name__ == "__main__":
    fire.Fire(run_hallucination_detection)
