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
from typing import List
from enum import Enum, unique
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import copy
import random


def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
            
def gen_ensemble_train_data(input_file = "../processed_data/Train_Data.csv", 
                               output_dir="../processed_data/ensemble_data/cls_entity/", 
                               num_split=5, if_shuffle =False, level="sentence"):
    items= []
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            if (len(row['text']) >512):
                print(row['\ufeffid'], len(row['text']))
            row['text'] =row['text'][:450]
            if level == 'sentence':
                item ={}
                item['passage'] = row['text']
                item['question'] = ''
                if row['negative'] == "0":
                    item['label'] = '负类'
                else:
                    item['label'] = '正类'
                items.append(item)
            else:
                for entity in row['entity'].split(";"):
                    item ={}
                    item['passage'] = row['text']
                    item['question'] = entity + "不好"
                    if entity in row['key_entity']:
                        item['label'] = '正类'
                    else:
                        item['label'] = '负类'
                    items.append(item)
    
    if if_shuffle:
        random.shuffle(items)
    
    for i in range(num_split):
        globals()["train_data" + str(i + 1)] = []
        globals()["dev_data" + str(i + 1)] = []
    for i, item in enumerate(items):
        cur = i % num_split + 1
        for j in range(num_split):
            if cur == j + 1:
                globals()["dev_data" + str(j + 1)].append(item)
            else:
                globals()["train_data" + str(j + 1)].append(item)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(num_split):
        write_file(globals()["train_data" + str(i + 1)], os.path.join(output_dir, "train_" + str(i + 1) + ".jsonl"))
        write_file(globals()["dev_data" + str(i + 1)], os.path.join(output_dir, "dev_" + str(i + 1) + ".jsonl"))


def gen_test_data(input_file = "../processed_data/Test_Data.csv" ,
                  output_file="../processed_data/test.jsonl"):
    items= []
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            item ={}
            item['id']=row['\ufeffid']
            item['passage'] = row['text'][:450]
            item['question_list'] =[]
            for entity in row['entity'].split(";"):
                item['question_list'].append(entity+"不好")
                # item['label'] = '正类'
            items.append(item)
    write_file(items, output_file)

def func():
    # gen_ensemble_train_data(input_file="../processed_data/Train_Data.csv", 
    #               output_dir="../processed_data/ensemble_data/cls_sentence/",
    #               num_split=5, if_shuffle= False,level='sentence')
    
    # gen_ensemble_train_data(input_file="../processed_data/Train_Data.csv", 
    #               output_dir="../processed_data/ensemble_data/cls_entity/",
    #               num_split=5, if_shuffle= False,level='entity')

    gen_test_data(input_file="../processed_data/Test_Data.csv", 
                  output_file="../processed_data/test.jsonl")

    pass


if __name__ == '__main__':
    func()
