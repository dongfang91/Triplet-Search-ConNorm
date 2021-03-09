"""
This script trains sentence transformers with a batch hard loss function.

The TREC dataset will be automatically downloaded and put in the datasets/ directory

Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
to select good triplets. If the negative sentence is selected randomly, the training objective is often
too easy and the network fails to learn good representations.

Batch hard triplet loss (https://arxiv.org/abs/1703.07737) creates triplets on the fly. It requires that the
data is labeled (e.g. labels A, B, C) and we assume that samples with the same label are similar:
A sent1; A sent2; B sent3; B sent4
...

In a batch, it checks for sent1 with label A what is the other sentence with label A that is the furthest (hard positive)
which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
all sentences with the same label should be close and sentences for different labels should be clearly seperated.
"""

from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceLabelDataset,
    LoggingHandler,
    losses,
)
import csv
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers import evaluation
from datetime import datetime
from BatchHardSoftMarginTripletLoss_custom import BatchHardSoftMarginTripletLoss

import logging
import os
import urllib.request
import random
from collections import defaultdict
import read_files as read

# Inspired from torchnlp
def n2c2_st():

    data = csv.reader(open(os.path.join("/xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology+data/ontology+train*100.tsv"), encoding="utf-8"), delimiter="\t",
                      quoting=csv.QUOTE_NONE)
    label_map = {}
    train_set = []
    guid = 0
    for line in data:
        # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
        text, label = line
        if label not in label_map:
            label_map[label] = len(label_map)

        label_id = label_map[label]
        guid += 1
        train_set.append(InputExample(guid=guid, texts=[text], label=label_id))


    # Create a dev set from train set



    # # For dev & test set, we return triplets (anchor, positive, negative)
    # random.seed(42) #Fix seed, so that we always get the same triplets
    # dev_triplets = triplets_from_labeled_dataset(dev_set)


    return train_set


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = '/groups/bethard/transformers/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/'

### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 800
output_path = (
    "/xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/ontology_100_train-"
    + "microsoft_"
    + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
)
os.makedirs(output_path, exist_ok=True)

num_epochs = 50

logging.info("Loading medmentions dataset")
train_set = n2c2_st()



# Load pretrained model
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
                               
 

# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=2048, activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.max_seq_length = 16

logging.info("Read medmentions train dataset")
train_dataset = SentenceLabelDataset(
    examples=train_set,
    model=model,
    provide_positive=True, #For BatchHardTripletLoss, we must set provide_positive and provide_negative to False
    provide_negative=False,
    max_processes=4
)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

### Triplet losses ####################
### There are 4 triplet loss variants:
### - BatchHardTripletLoss
### - BatchHardSoftMarginTripletLoss
### - BatchSemiHardTripletLoss
### - BatchAllTripletLoss
#######################################

#train_loss = losses.BatchAllTripletLoss(model=model)
#train_loss = losses.BatchHardTripletLoss(sentence_embedder=model)
train_loss = BatchHardSoftMarginTripletLoss(model)
#train_loss = losses.BatchSemiHardTripletLoss(sentence_embedder=model)

# evaluator = []

logging.info("Read n2c2 val dataset")
# # ir_queries_medra = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/MedRa/queries")
# ir_queries = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/n2c2/n2c2_sentence_search/ir/train_queries")
# ir_corpus_medra = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/umls/umls_sentence_search/snomed_rxnorm_all/corpus")
# # ir_relevant_docs_medra = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/MedRa/relevant_docs")
# ir_relevant_docs = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/n2c2/n2c2_sentence_search/ir/train_relevant_docs")
# ir_evaluator_n2c2 = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus_medra, ir_relevant_docs,name="mean_umls_n2c2train_train",
#                                                         map_at_k=[1,3,5,10],batch_size=1024,show_progress_bar=True)
# evaluator.append(ir_evaluator_n2c2)

# ir_queries_medra = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/MedRa/queries")
ir_queries = read.read_from_json("/xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology+data/ir/dev_queries")
ir_corpus_medra = read.read_from_json("/xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology+data/ir/corpus")
# ir_relevant_docs_medra = read.read_from_json("/xdisk/bethard/mig2020/extra/dongfangxu9/resources/MedRa/relevant_docs")
ir_relevant_docs = read.read_from_json("/xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology+data/ir/dev_relevant_docs")
ir_evaluator_n2c2_dev = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus_medra, ir_relevant_docs,name="mean_umls_n2c2st_dev",
                                                        map_at_k=[1,3,5,10],batch_size=1200,show_progress_bar=True)
# evaluator.append(ir_evaluator_n2c2_dev)
# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
# seq_evaluator = evaluation.SequentialEvaluator(evaluator, main_score_function=lambda scores: scores[1])

logging.info("Performance before fine-tuning:")
ir_evaluator_n2c2_dev(model)


# warmup_steps = int(
#     len(train_dataset) * num_epochs / train_batch_size * 0.1
# )  # 10% of train data
warmup_steps = 0

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator_n2c2_dev,
    output_path_ignore_not_empty=True,
    optimizer_params = {'lr': 3e-5, 'eps': 1e-6, 'correct_bias': False},
    epochs=num_epochs,
    # evaluation_steps=995, #142,
    warmup_steps=warmup_steps,
    output_path=output_path,
)

##############################################################################
#
# Load the stored model and evaluate its performance on TREC dataset
#
##############################################################################

# logging.info("Evaluating model on test set")
# test_evaluator = TripletEvaluator.from_input_examples(test_set, name='test')
# model.evaluate(test_evaluator)
