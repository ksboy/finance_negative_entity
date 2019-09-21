# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: process_ner_data.py
@time: 2019-08-08 21:53

这一行开始写关于本文件的说明与解释
"""
import json, csv, os
import pandas as pd
import random

def reformat_csv(input_file, oputput_file):
    """
    """
    import pandas as pd
    f=pd.read_csv(input_file)
    keep_col = ['id','AspectTerms','A_start','A_end','OpinionTerms','O_start','O_end']
    new_f = f[keep_col]
    new_f.to_csv(oputput_file, index=False)

def shuffle_file(input_file, output_file):
    reviews = open(input_file, encoding='utf-8').readlines()
    random.shuffle(reviews)
    with open(output_file, 'w', encoding='utf-8') as f:
        for review in reviews:
            f.write(review)

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
            
def gen_ensemble_data(input_file = "../processed_data/ner_train_laptop.jsonl", 
                               output_dir='../processed_data/ensemble_data/ner_data/', 
                               num_split=7, if_shuffle =False):
    datas = open(input_file, encoding='utf-8').read().splitlines()
    for i in range(num_split):
        globals()["train_data" + str(i + 1)] = []
        globals()["dev_data" + str(i + 1)] = []
    for i, data in enumerate(datas):
        cur = i % num_split + 1
        for j in range(num_split):
            if cur == j + 1:
                globals()["dev_data" + str(j + 1)].append(json.loads(data))
            else:
                globals()["train_data" + str(j + 1)].append(json.loads(data))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(num_split):
        if if_shuffle:
            random.shuffle(globals()["train_data" + str(i + 1)])
            random.shuffle(globals()["dev_data" + str(i + 1)])
        write_file(globals()["train_data" + str(i + 1)], os.path.join(output_dir, "train_" + str(i + 1) + ".jsonl"))
        write_file(globals()["dev_data" + str(i + 1)], os.path.join(output_dir, "dev_" + str(i + 1) + ".jsonl"))

def mix_file(file1,file2,output_file):

    lines_1 = open(file1, encoding='utf-8').readlines()
    lines_2 = open(file2, encoding='utf-8').readlines()
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines_1:
            f.write(line)
        for line in lines_2:
            f.write(line)


def func():

    # reformat_csv(input_file=   "../processed_data/laptop_corpus_relation_roberta_tri_predict/Train_laptop_labels.csv",
    #              oputput_file= "../processed_data/laptop_corpus_relation_roberta_tri_predict/cls_relation_laptop.csv")
    
    # shuffle_file('../data/processed_data/laptop_corpus_relation_roberta_tri_predict/cls_relation_predict1.jsonl',
    #              '../data/processed_data/laptop_corpus_relation_roberta_tri_predict/cls_relation_predict1_shuffle.jsonl')
    
    in_file_1 = "../data/processed_data/ensemble_data/relation_laptop_5_shuffle/train_"
    in_file_2 = "../data/processed_data/laptop_corpus_relation_roberta_tri_predict/relation_tri_training_first_5/train_"
    out_file = "../data/processed_data/laptop_corpus_relation_roberta_tri_predict/relation_tri_training_first_5/train_"
    for i in range(5):
        mix_file(in_file_1+str(i+1)+".jsonl",
                 in_file_2+str(i+1)+".jsonl",
                 out_file+str(i+1)+"_mix.jsonl")
    pass


if __name__ == '__main__':
    func()
