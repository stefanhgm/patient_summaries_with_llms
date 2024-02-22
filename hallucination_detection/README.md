# Automatic Hallucination Detection

This folder contains the code to run automatic hallucination detection based on medical entities extracted with MedCAT and SapBERT embeddings.

## Notebooks

* [convert_bioc_to_json_datasets.ipynb](convert_bioc_to_json_datasets.ipynb): Convert the BIOC format exported with MedTator to JSON dataset.
* [create_medcat_entities_and_sapbert_embeddings.ipynb](create_medcat_entities_and_sapbert_embeddings.ipynb): Create MedCAT entities and SapBERT embeddings for the hallucination detection.
* [evaluate_hallucination_detection_entities_embeddings.ipynb](evaluate_hallucination_detection_entities_embeddings.ipynb): Evaluate the hallucination detection based on the entities and embeddings.