import read_files as read
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import os

def main_umls(query_path,documents_path,file_name):
    
    query = read.read_from_pickle(query_path)
    documents = read.read_from_pickle(documents_path)
    for i in range(73):
        index_start = 20000*i
        index_end = 20000*(i+1)
        similarity_matrix = cosine_similarity(query[index_start:index_end],documents)
        idx = np.argsort(similarity_matrix)
        idx_output = idx[:,-100:]
        
        cui_path = "/extra/dongfangxu9/umls/processed/cui_snomed_rxnorm_all"
        gold_label = read.read_from_json(cui_path)
        dev_pre = [[gold_label[int(item)] for item in row[-50:]] for row in idx ]
        read.save_in_json(file_name+ "_" +str(i),dev_pre)

def main(query_path,documents_path,file_name,array_output):
    
    query = read.read_from_pickle(query_path)
    documents = read.read_from_pickle(documents_path)

    similarity_matrix = cosine_similarity(query,documents)


    idx = np.argsort(similarity_matrix)
    idx_output = idx[:,-30:]
    score_array = [row[idx_output[i]] for i,row in enumerate(similarity_matrix)]
        
    # cui_path = "/xdisk/hongcui/mig2020/extra/dongfangxu9/umls/processed/cui_snomed_rxnorm_all"
    cui_path = "/xdisk/hongcui/mig2020/extra/dongfangxu9/MedRa/label"
    
    gold_label = read.read_from_json(cui_path)
    dev_pre = [[gold_label[int(item)] for item in row[-30:]] for row in idx ]
    read.save_in_json(file_name,dev_pre)
    read.save_in_pickle(array_output,score_array)
    
def main_cuiless(query_path,documents_path,file_name,array_output):
    
    query = read.read_from_pickle(query_path)
    documents = read.read_from_pickle(documents_path)

    similarity_matrix = cosine_similarity(query,documents)


    idx = np.argsort(similarity_matrix)
    idx_output = idx[:,-30:]
    score_array = [row[idx_output[i]] for i,row in enumerate(similarity_matrix)]
        
    cui_path = "/xdisk/hongcui/mig2020/extra/dongfangxu9/n2c2/n2c2_triplet/sentence_search/cuiless/train_dev_cuiless/input.tsv"
    gold_label = read.read_from_tsv(cui_path)
    dev_pre = [[gold_label[int(item)][0] for item in row[-30:]] for row in idx ]
    read.save_in_json(file_name,dev_pre)
    read.save_in_pickle(array_output,score_array)
    
def main_avg_cuiless(query_path,documents_path,file_name,array_output):
    
    query = read.read_from_pickle(query_path)
    documents = read.read_from_pickle(documents_path)

    similarity_matrix = cosine_similarity(query,[documents])



    score_array = [row[0] for i,row in enumerate(similarity_matrix)]
    read.save_in_pickle(array_output,score_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentence embedding for each sentence in the sentence corpus ')

    parser.add_argument('--query_path',
                        help='the direcotory of the model',required= True)

    parser.add_argument('--documents_path',
                        help='the type of the model, sentence_bert or just bert',required= True)

    parser.add_argument('--file_name',
                        help='the direcotory of the sentence corpus',required=True)
                        
    parser.add_argument('--array',
                        help='the direcotory of the sentence corpus',required=True)


    args = parser.parse_args()
    query_path = args.query_path
    documents_path = args.documents_path
    file_name = args.file_name
    array_output = args.array
    main(query_path,documents_path,file_name,array_output)
