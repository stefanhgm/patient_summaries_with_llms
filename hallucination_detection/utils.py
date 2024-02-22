import pandas as pd
import numpy as np
from bioc import biocxml
import re

# Label defintions
labels = {
    'c': 'condition_unsupported',
    'p': 'procedure_unsupported',
    'm': 'medication_unsupported',
    't': 'time_unsupported',
    'l': 'location_unsupported',
    'n': 'number_unsupported',
    'na': 'name_unsupported',
    'w': 'word_unsupported',
    'o': 'other_unsupported',
    'co': 'contradicted_fact',
    'i': 'incorrect_fact'
}

# Read dataset
def read_bioc(path):
    with open(path, 'r') as fp:
        return biocxml.load(fp)
    
# Create dict of document ids and their annotations
def extract_id(document_name):
    if re.search(r'\d+_(qualitative|hallucination)', document_name):
        return int(document_name.split('_')[0])
    elif re.search(r'train_4000_600', document_name):
        return int(document_name.split('_')[-1].split('.')[0])
    else:
        raise ValueError('Document name does not contain id')

def parse_label(annotation):
    # Create a dict of start index, end index, length, label, text
    start = annotation.locations[0].offset
    end = start + annotation.locations[0].length
    length = annotation.locations[0].length
    # Get all character before digit of annotation id
    label_prefix = str(re.findall(r'[^\d]+', annotation.id)[0])
    label = labels[label_prefix.lower()]
    text = annotation.text
    return {'start': start, 'end': end, 'length': length, 'label': label, 'text': text}

# Sort lists of dict by dict key start
def sort_by_start(l):
    return sorted(l, key=lambda k: k['start'])

def parse_text_labels(labeling):
    result = {}
    for document in labeling.documents:
        id = extract_id(document.id)
        labels = sort_by_start([parse_label(a) for a in document.passages[0].annotations])
        text = document.passages[0].text
        result[id] = {'labels': labels, 'text': text}
    return result
