#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True
### Specify a name for the job
#PBS -N encoder_triplet
### Specify the group name
#PBS -W group_list=hongcui
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=112:00:00
### Walltime is how long your job will run
#PBS -l walltime=4:00:00
#PBS -e log/error/umls_medmention_similarity_search_292104
#PBS -o log/output/umls_medmention_similarity_search_292104


module load singularity/3/3.6.4

cd $PBS_O_WORKDIR

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/train_con_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/data/train_cui \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_train_con \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_train_con

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology_con_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology/ontology_cui \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology_con \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology_con

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology+train_con_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology/ontology_cui \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology+train_con \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology+train_con





singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/train+dev_con_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/data/train+dev_cui \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_train+dev_con \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_train+dev_con

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology_con_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology/ontology_cui \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology_con \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology_con

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology+train+dev_con_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology/ontology_cui \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology+train+dev_con \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology+train+dev_con

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_cuiless.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/train_cuiless_embeddings.npy \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_train_cuiless \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_train_cuiless

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_cuiless.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/train+dev_cuiless_embeddings.npy \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_train+dev_cuiless \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_train+dev_cuiless




singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_single.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/train_syn_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/data/cui_mentions_train \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_train_syn \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_train_syn

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_single.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology_syn_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology/cui_mentions_ontology \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology_syn \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology_syn

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_single.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology+train_syn_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology+data/cui_mentions_ontology+train \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology+train_syn \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/dev_ontology+train_syn



singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_single.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/train+dev_syn_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/data/cui_mentions_train+dev \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_train+dev_syn \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_train+dev_syn

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_single.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology_syn_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology/cui_mentions_ontology \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology_syn \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology_syn

singularity exec --nv /groups/bethard/image/hpc-ml_centos7-python3.7-transformers3.0.2.sif python3.7 similarity_search_single.py \
--query_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_embeddings.npy \
--documents_path /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/ontology+train+dev_syn_embeddings.npy \
--cui_path /xdisk/bethard/dongfangxu9/resources/n2c2_st/semantic_group/ontology+data/cui_mentions_ontology+train+dev \
--file_name /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology+train+dev_syn \
--array /xdisk/bethard/dongfangxu9/triplet/n2c2_st/semantic_group/output/ontology+data*100/test_ontology+train+dev_syn

