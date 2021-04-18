# Triplet-Search-ConNorm
Approach concept normalization as information retrieval

## Setup
* We recommend Python 3.7 or higher. The code does **not** work with Python 2.7.
* The triplet search is implemented with PyTorch (1.5.1+cu101) using [transformers v3.0.2](https://github.com/huggingface/transformers), and [sentence-transformers v0.3.5.1](https://github.com/UKPLab/sentence-transformers)

## Getting Started

### Data format
 * Please see data/training/input.tsv for the input format: each row is a pair of mention and concept.
 * data/ontology/label.txt is a json file that contains all the concepts from the ontology.
 * To train and evaluate the model, please have the following files ready: "train.tsv", "dev.tsv", "test.tsv", and "label.txt".

### Different pre-trained models
 * [Pre-trained LM](https://huggingface.co/models?filter=pytorch)
 * [Pre-trained LM fine-tuned on text classification task](https://huggingface.co/models?filter=pytorch,text-classification)
 * If using model from local, in addition to the arguement "model_name_or_path",
 please also add arguements "--config_name", "--tokenizer_name".

  ### Code to run
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