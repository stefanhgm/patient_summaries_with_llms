# A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models

![Figure1-5](https://github.com/user-attachments/assets/fa631f08-9e56-4a37-aea3-3b46fd6d31ef)

This repository contains the code to reproduce the results of the  paper [A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models](https://proceedings.mlr.press/v248/hegselmann24a.html) by Stefan Hegselmann, Shannon Zejiang Shen, Florian Gierse, Monica Agrawal, David Sontag, and Xiaoyi Jiang.

We released the 100 doctor-written summaries from the MIMIC-IV-Note Discharge Instructions and hallucinations 100 LLM-generated patient summaries annotated for unsupported facts by two medical experts on PhysioNet. We also published all datasets created in our work to fully reproduce our experiments.

If you consider our work helpful or use our datasets, please consider the citations for our paper and PhysioNet repository:

```
@InProceedings{pmlr-v248-hegselmann24a,
  title = 	{A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models},
  author =      {Hegselmann, Stefan and Shen, Zejiang and Gierse, Florian and Agrawal, Monica and Sontag, David and Jiang, Xiaoyi},
  booktitle = 	{Proceedings of the fifth Conference on Health, Inference, and Learning},
  pages = 	{339--379},
  year = 	{2024},
  volume = 	{248},
  series = 	{Proceedings of Machine Learning Research},
  month = 	{27--28 Jun},
  publisher =   {PMLR},
  url = 	{https://proceedings.mlr.press/v248/hegselmann24a.html},
}

@Misc{hegselmann_ann-pt-summ2024,
  title = 	{Medical Expert Annotations of Unsupported Facts in {Doctor}-{Written} and LLM-Generated Patient Summaries},
  author =      {Hegselmann, Stefan and Shen, Zejiang and Gierse, Florian and Agrawal, Monica and Sontag, David and Jiang, Xiaoyi},
  booktitle = 	{Proceedings of the fifth Conference on Health, Inference, and Learning},
  year = 	{2024},
  publisher =   {PhysioNet},
  url = 	{https://physionet.org/content/ann-pt-summ/1.0.0/},
  doi = 	{https://doi.org/10.13026/a66y-aa53},
}
```

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
