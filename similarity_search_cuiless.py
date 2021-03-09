import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import read_files as read


def main_cuiless(query_path, documents_path, file_name, array_output):

    query = np.load(query_path)
    documents = np.load(documents_path)

    similarity_matrix = cosine_similarity(query, documents)

    idx = np.argsort(similarity_matrix)
    idx_output = idx[:, ::-1]
    idx_output = idx_output[:, :30]
    score_array = [
        row[idx_output[i]] for i, row in enumerate(similarity_matrix)
    ]

    # cui_path = "/xdisk/hongcui/mig2020/extra/dongfangxu9/n2c2/n2c2_triplet/sentence_search/cuiless/train_dev_cuiless/input.tsv"
    # gold_label = read.read_from_tsv(cui_path)
    # dev_pre = [[gold_label[int(item)][0] for item in row[-30:]] for row in idx ]
    # read.save_in_json(file_name,dev_pre)
    np.save(array_output, score_array)
    # read.save_in_pickle(array_output,score_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Generate sentence embedding for each sentence in the sentence corpus '
    )

    parser.add_argument('--query_path',
                        help='the direcotory of the model',
                        required=True)

    parser.add_argument(
        '--documents_path',
        help='the type of the model, sentence_bert or just bert',
        required=True)

    parser.add_argument('--file_name',
                        help='the direcotory of the sentence corpus',
                        required=True)

    parser.add_argument('--array',
                        help='the direcotory of the sentence corpus',
                        required=False,
                        default="")

    args = parser.parse_args()
    query_path = args.query_path
    documents_path = args.documents_path
    file_name = args.file_name
    array_output = args.array
    main_cuiless(query_path, documents_path, file_name, array_output)
