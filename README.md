# Triplet-Search-ConNorm
The Triplet-Search-ConNorm library provides models for normalizing textual mentions of concepts to concepts in an ontology.

## Triplet-Trained Vector Space Model
The approach we use for concept normalization is a transformer based vector-space model, which encodes mentions and concepts via transformer networks that are trained via a triplet objective with online hard triplet mining.

> Dongfang Xu, and Steven Bethard. 2018.
> [Triplet-Trained Vector Space and Sieve-Based Search Improve Biomedical Concept Normalization]().
> In: BioNLP 2021

To use the Triplet-Search-ConNorm, create an instance of `ConceptNormalizer`, and load an ontology from a tsv file (plese see the [ontology]())

```python3.8

>>> from ConceptNormalizer import ConceptNormalizer
>>> normalizer = ConceptNormalizer(model_name_or_path='path-to-bert')
>>> normalizer.load_ontology('path-to-ontology')
>>> normalizer.normalize("head spinning", top_k=4)
[('C0273239', 'head wound', 0.96794385), ('C0018670', 'HEAD', 0.96687186), ('C0018670', 'head', 0.96687186), ('C0230420', 'legs', 0.96390116)]
>>> normalizer.add_terms(concepts=["C0012833"], synonyms=["head spinning"])
>>> normalizer.normalize("head spinning", top_k=4)
[('C0012833', 'head spinning', 0.99999934), ('C0273239', 'head wound', 0.96794385), ('C0018670', 'HEAD', 0.96687186), ('C0018670', 'head', 0.96687186)]
```



## Setup
* We recommend Python 3.7 or higher.
* The triplet search is implemented with PyTorch (1.7.1) using [transformers v4.4.1](https://github.com/huggingface/transformers), and [sentence-transformers v0.4.1.2](https://github.com/UKPLab/sentence-transformers)

## Getting Started

### Data format
 * Please see data/ontology.tsv for the input format: each row is a pair of mention and concept.
 * The same input form is also used for training sentence-transformers.

### Pre-trained models
```
word_embedding_model = models.BERT(path_to_BERT-based-models)
pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),pooling_mode_mean_tokens=True)
concept_normalizer = SentenceTransformer(
                modules=[word_embedding_model, pooling_model])
```

### Fine-tuned sentence-transformer based models
```
concept_normalizer = SentenceTransformer(model_name_or_path)
```
 * SentenceTransformer model is fine-tuned using triplet_training.py

  ### Code to train sentence-transformer using triplet network
* code to run the triplet network for the concept normalization tasks using ncbi dataset:
```
python3.7 triplet_training.py \
--model_name_or_path bert-base-uncased \
--task_name conceptnorm \
--do_train \
--do_eval \
--do_predict \
--data_dir data/askapatient/0/ \
--label_dir data/askapatient/label.txt \
--max_seq_length 64 \
--per_device_eval_batch_size=8 \
--per_device_train_batch_size=8 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--output_dir /path/to/model_new/
```