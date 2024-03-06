# A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models

![Figure1.pdf](https://github.com/stefanhgm/patient_summaries_with_llms/files/14378703/Figure1.pdf)

This repository contains the code to reproduce the results of the  paper [A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models](https://arxiv.org/abs/2402.15422) by Stefan Hegselmann, Shannon Zejiang Shen, Florian Gierse, Monica Agrawal, David Sontag, and Xiaoyi Jiang.

*We will release our custom MIMIC datasets on PhysioNet soon.*

## Overview

Here you will find the general procedures to setup the environment, download the data, and run the code.
More detailed instructions for each component of the project can be found in the respective folders.

* [gpt-4](gpt-4/README.md): All code related to the GPT-4 experiments.
* [hallucination_detection](hallucination_detection/README.md): All code related to the hallucination detection experiments without gpt-4.
* [labeling](labeling/README.md): Scripts to analyse and work with labeling data created with MedTator.
* [notebooks](notebooks/README.md): Jupyter notebooks for different experiments, helper tasks, and analyses.
* [preprocess](preprocess/README.md): Preprocessing pipeline as presented in the paper.
* [scripts](scripts/README.md): Scripts for parameter tuning of LED and LLama 2 models.
* [summarization](summarization/README.md): All code related to the summarization experiments with LED and Llama 2 models.


## Setting Correct Paths

We assume the root path to be `/root` in this readme and for the code.
Hence, we assume the repository is cloned to `/root/patient_summaries_with_LLMs`. 
Please adapt the paths according to your local setup.


## Preparing the Environment

We used conda to create the necessary virtual environments. For the `ps_llms` environment, we used python 3.9.18:

```
conda create -n ps_llms python==3.9.18
conda activate ps_llms
```

Next, install the nevessary requirements. For installing `torch` you might adapt the command in the first line based on [this suggestion](https://pytorch.org).

```
pip install torch torchvision torchaudio
pip install transformers bitsandbytes sentencepiece accelerate datasets peft trl py7zr scipy wandb evaluate rouge-score sacremoses sacrebleu seqeval bert_score swifter bioc medcat plotly nervaluate nbformat kaleido
pip install -U spacy
python -m spacy download en_core_web_sm
```
