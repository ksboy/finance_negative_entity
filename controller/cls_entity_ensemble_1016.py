# encoding: utf-8
"""
@author: whou
@contact: whou@126.com

@version: 1.0
@file: viewpoint_mining.py
@time: 2019-08-09 19:39

这一行开始写关于本文件的说明与解释
"""
import sys

sys.path.append("../")

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from typing import List
# import finance_negative_entity  # 是必要的，虽然没有显式用到，但是它会告诉allennlp一个搜索路径。
import finance_negative_entity
import time
from tqdm import tqdm
import os
import json, csv
from scripts.ensemble import cls_entity_ensemble
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(filename='finance_negative_entity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BertSeqPairClsEngine(object):
    def __init__(self, model_path: str, predictor_type: str, cuda_device: int = -1):
        self.predictor_type = predictor_type
        self.predictor = Predictor.from_archive(load_archive(model_path, cuda_device=cuda_device), predictor_type)

    def predict(self, passage: str, question):
        result = self.predictor.predict(passage=passage.rstrip(), question=question)
        return result['label']


CLS_ENTITY_MODEL_DIR = "/home/mhxia/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_ReLU_1536_"
CLS_ENTITY_CUDA_DEVICE = 1
NUM_CLS_ENTITY_MODELS = 8


class ClsEntity(object):
    def __init__(self):

        self.cls_entity_predictor_pool = []
        for i in range(NUM_CLS_ENTITY_MODELS):
            model_path = os.path.join(CLS_ENTITY_MODEL_DIR + str(i + 1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CLS_ENTITY_CUDA_DEVICE)
            self.cls_entity_predictor_pool.append(model)
        # model_path = os.path.join(CLS_ENTITY_MODEL_DIR, "model.tar.gz")
        # model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CLS_ENTITY_CUDA_DEVICE)
        # self.cls_entity_predictor_pool.append(model)


    def predict(self, input_file, output_file, test_file="../data/results/result_log2.csv"):
        """
        输入必须有序号，包含在passage中，
        例如：`1,凑单买的面膜，很好用，买给妈妈的`
        :param passage:
        :return:
        """
        with open(input_file, encoding='utf-8') as infile:
            items = []
            for row in infile:
                row = json.loads(row)
                item = {}
                item['id'] = row['id']
                item['passage'] = row['passage']
                item['negative'] = row['negative']
                item['entity'] = []
                for entity in row['entity']:
                    item['entity'].append(entity)
                # print(item)
                items.append(item)

        outputs = []
        print_items = []
        count = 0
        reverse_count1 = 0
        reverse_count2 = 0
        for item in tqdm(items, desc="正在预测", ncols=80):
            # item =json.loads(item)
            passage = item['passage']
            entity_list = item['entity']
            # 去除重复 entity
            pop_index = []
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    if entity_list[i] in entity_list[j] or entity_list[j] in entity_list[i]:
                        if passage.count(entity_list[i]) == passage.count(entity_list[j]):
                            # print(item['id'], passage,
                            #       entity_list[i], entity_list[j],
                            #       passage.count(entity_list[i]),passage.count(entity_list[j]))
                            count += 1
                            if entity_list[i] in entity_list[j]:
                                pop_index.append(i)
                            else:
                                pop_index.append(j)

            entity_list = [entity_list[i] for i in range(len(entity_list)) if (i not in pop_index)]

            output_item = {}
            output_item['id'] = item['id']
            output_item['entity'] = []
            output_item['negative'] = 0

            if item['negative'] == 1:
                # 判断 negative_entity
                negative_entity_list = []
                for i, entity in enumerate(entity_list):
                    entity_label_ensemble = []
                    for model in self.cls_entity_predictor_pool:
                        entity_label_ensemble.append(model.predict(passage[:512 - 3 - 2 - len(entity)], entity))
                    entity_label = cls_entity_ensemble(entity_label_ensemble)
                    if entity_label == "正类":
                        negative_entity_list.append(entity)
                # print(negative_entity_list)

                output_item['entity'] = negative_entity_list
                output_item['negative'] = 0 if len(negative_entity_list) == 0 else 1
                if len(negative_entity_list) == 0:
                    output_item['negative'] = 0
                    # print_items.append(item)
                    # reverse_count1 += 1
                else:
                    output_item['negative'] = 1
            # else:
            #     negative_entity_list = []
            #     for i, entity in enumerate(entity_list):
            #         entity_label_ensemble = []
            #         for model in self.cls_entity_predictor_pool:
            #             entity_label_ensemble.append(model.predict(passage[:512 - 3 - 2 - len(entity)], entity))
            #         entity_label = cls_entity_ensemble(entity_label_ensemble)
            #         if entity_label == "正类":
            #             negative_entity_list.append(entity)
            #     if len(negative_entity_list) != 0:
            #         print_items.append(item)
            #         reverse_count2 += 1

            outputs.append(output_item)
        print("去重 entity 的个数", count)
        # print("空entity_list个数", reverse_count1)
        # print("entity_list个数", reverse_count2)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("id,negative,key_entity\n")
            for item in outputs:
                f.write(item['id'] + ',' + str(item['negative']) + ',' + ';'.join(item['entity']))
                f.write("\n")

        with open(test_file, 'w', encoding='utf-8') as f:
            for item in print_items:
                f.write(item['id'] + ',' + str(item['passage']) + str(item['negative']) + ',' + ';'.join(item['entity']))
                f.write("\n")


def func():
    cls_entity = ClsEntity()
    input_file = '../data/results/result_sentence_roberta_logits6_1024.jsonl'
    output_file = '../data/results/result_test_cleansed1_roberta_logits6_1024.csv'

    # input_file  = '../data/processed_data/test_some.jsonl'
    # output_file = '../data/results/result_some.csv'

    cls_entity.predict(input_file, output_file)
    pass


if __name__ == '__main__':
    func()