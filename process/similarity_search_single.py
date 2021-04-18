import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import read_files as read


def main_single(query_path, documents_path, file_name, array_output, cui_path):

    query = np.load(query_path)
    documents = np.load(documents_path)

    gold_label = read.read_from_json(cui_path)
    gold_label = {
        int(i): cui
        for cui, item in gold_label.items()
        for i in range(int(item[0]), int(item[1]))
    }

    similarity_matrix = cosine_similarity(query, documents)
    similarity_matrix = similarity_matrix.astype(np.float16)
    idx = np.argsort(similarity_matrix)
    idx = idx.astype(np.int32)
    idx = idx[:, ::-1][:, :30]

    score_array = [row[idx[i]] for i, row in enumerate(similarity_matrix)]
    dev_pre = [[gold_label[item] for item in row] for row in idx]

    read.save_in_json(file_name, dev_pre)
    np.save(array_output, score_array)


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

    parser.add_argument(
        '--cui_path',
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
    cui_path = args.cui_path
    main_single(query_path, documents_path, file_name, array_output, cui_path)
