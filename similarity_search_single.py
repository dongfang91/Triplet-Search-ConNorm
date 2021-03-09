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

def main(query_path,documents_path,file_name,array_output,cui_path):
    
    query = np.load(query_path)
    documents = np.load(documents_path)

    similarity_matrix = cosine_similarity(query,documents)


    idx = np.argsort(similarity_matrix)
    # idx_output = idx[:,-30:]
    # score_array = [row[idx_output[i]] for i,row in enumerate(similarity_matrix)]
        
    # cui_path = "/xdisk/hongcui/mig2020/extra/dongfangxu9/umls/processed/cui_snomed_rxnorm_all"
    # cui_path = "/xdisk/bethard/mig2020/extra/dongfangxu9/resources/MedRa/label"
    
    gold_label = read.read_from_json(cui_path)
    dev_pre = [[gold_label[int(item)] for item in row[::-1][:15]] for row in idx ]
    read.save_in_json(file_name,dev_pre)
    np.save(array_output,score_array)

def main_single(query_path,documents_path,file_name,array_output,cui_path):

    query = np.load(query_path)
    documents = np.load(documents_path)
    

    gold_label = read.read_from_json(cui_path)
    gold_label = {int(i):cui for cui,item in gold_label.items() for i in range(int(item[0]), int(item[1]))}
    
    similarity_matrix = cosine_similarity(query,documents)
    similarity_matrix = similarity_matrix.astype(np.float16)
    idx = np.argsort(similarity_matrix)
    idx = idx.astype(np.int32)
    idx = idx[:,::-1][:,:30]

    score_array = [row[idx[i]] for i,row in enumerate(similarity_matrix)]
    dev_pre = [[gold_label[item] for item in row] for row in idx ]

    read.save_in_json(file_name,dev_pre)
    np.save(array_output,score_array)

def main_single_split(query_path,documents_path,file_name,array_output,cui_path):
    import math
    query = np.load(query_path)
    documents = np.load(documents_path)
    
    batch = math.floor(len(documents)/200000)
    batch = [i*200000 for i in range(batch)]
    batch.append(len(documents))

    score_arrays = []
    dev_pres = []
    gold_label = read.read_from_json(cui_path)
    gold_label = {int(i):cui for cui,item in gold_label.items() for i in range(int(item[0]), int(item[1]))}
    
    for i in range(len(batch)-1):
        documents_single =documents[batch[i]:batch[i+1]]
        
        similarity_matrix = cosine_similarity(query,documents_single)
        similarity_matrix = similarity_matrix.astype(np.float16)
        idx = np.argsort(similarity_matrix)
        idx = idx.astype(np.int32)
        idx = idx[:,::-1][:,:30]

        score_array = [row[idx[i]] for i,row in enumerate(similarity_matrix)]
        idx = idx + 200000*i
        dev_pre = [[gold_label[item] for item in row] for row in idx ]

        read.save_in_json(file_name+str(i),dev_pre)
        np.save(array_output+str(i),score_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentence embedding for each sentence in the sentence corpus ')

    parser.add_argument('--query_path',
                        help='the direcotory of the model',required= True)

    parser.add_argument('--documents_path',
                        help='the type of the model, sentence_bert or just bert',required= True)
    
    parser.add_argument('--cui_path',
                        help='the type of the model, sentence_bert or just bert',required= True)

    parser.add_argument('--file_name',
                        help='the direcotory of the sentence corpus',required=True)
                        
    parser.add_argument('--array',
                        help='the direcotory of the sentence corpus',required=False, default="")


    args = parser.parse_args()
    query_path = args.query_path
    documents_path = args.documents_path
    file_name = args.file_name
    array_output = args.array
    cui_path = args.cui_path
    main_single(query_path,documents_path,file_name,array_output, cui_path)
