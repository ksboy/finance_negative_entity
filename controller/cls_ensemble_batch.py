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
from typing import List, Tuple
# import finance_negative_entity 是必要的，虽然没有显式用到，但是它会告诉allennlp一个搜索路径。
import finance_negative_entity
import time
from tqdm import tqdm
import os
import json, csv
from scripts.ensemble import cls_entity_ensemble
import logging
import numpy as np
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
    
    def batch_predict(self, inputs: List[Tuple[str, str]]):
        inputs_jsons = [{"passage": passage, "question": question} for passage, question in inputs]
        assert len(inputs_jsons) != 0
        outputs = self.predictor.predict_batch_json(inputs_jsons)
        results = []
        for o in outputs:
            results.append(o['label'])
        return results

CLS_ENTITY_MODEL_DIR = "/home/whou/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_"
CLS_SENTENCE_MODEL_DIR = "/home/whou/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_model_"
CLS_ENTITY_CUDA_DEVICE = 0
CLS_SENTENCE_CUDA_DEVICE = 1
NUM_CLS_ENTITY_MODELS = 5
NUM_CLS_SENTENCE_MODELS = 5
batch_size = 48

class ClsEntity(object):
    def __init__(self):

        self.cls_entity_predictor_pool = []
        for i in range(NUM_CLS_ENTITY_MODELS):
            model_path = os.path.join(CLS_ENTITY_MODEL_DIR+str(i+1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CLS_ENTITY_CUDA_DEVICE)
            self.cls_entity_predictor_pool.append(model)
        
        self.cls_sentence_predictor_pool = []
        for i in range(NUM_CLS_SENTENCE_MODELS):
            model_path = os.path.join(CLS_SENTENCE_MODEL_DIR+str(i+1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CLS_SENTENCE_CUDA_DEVICE)
            self.cls_sentence_predictor_pool.append(model)

    def predict(self, input_file, output_file):
        """
        输入必须有序号，包含在passage中，
        例如：`1,凑单买的面膜，很好用，买给妈妈的`
        :param passage:
        :return:
        """
        with open(input_file, encoding='utf-8') as infile:
            items=[]
            for row in infile:
                row = json.loads(row)
                item ={}
                item['id']=row['id']
                item['passage'] = row['passage']
                item['entity'] =[]
                for entity in row['entity']:
                    item['entity'].append(entity)
                # print(item)
                items.append(item)
                
        batch_items_list = []
        i = 0
        while i < len(items):
            batch_items_list.append(items[i:i+batch_size])
            i += batch_size

        outputs=[]
        count = 0
        for batch_items in tqdm(batch_items_list, desc="正在预测", ncols=80):
            # item =json.loads(item)
            batch_inputs_sentence = []
            batch_inputs_entity = []
            batch_output_item = []
            batch_entity_list = []
            for item in batch_items:
                passage=item['passage']
                batch_inputs_sentence.append((passage[:512-3-2],""))
                entity_list = item['entity']
                ## 去除重复 entity
                pop_index = []
                for i in range(len(entity_list)):
                    for j in range(i+1, len(entity_list)):
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
                batch_entity_list.append(entity_list) 

                inputs_entity =[]
                for entity in entity_list:
                    inputs_entity.append((passage[:512-3-2-len(entity)], entity))
                batch_inputs_entity.append(inputs_entity)

                output_item ={}
                output_item['id'] = item['id']
                output_item['entity'] = []
                output_item['negative'] = 0 
                batch_output_item.append(output_item)
            
            batch_sentence_label_ensemble = []
            for model in self.cls_sentence_predictor_pool:
                batch_sentence_label_ensemble.append(model.batch_predict(batch_inputs_sentence))
            batch_sentence_label_ensemble = np.array(batch_sentence_label_ensemble).T.tolist()

            for i, sentence_label_ensemble in enumerate(batch_sentence_label_ensemble):
                sentence_label = cls_entity_ensemble(sentence_label_ensemble)
                
                if sentence_label == "正类":
                    negative_entity_list=[]
                    if len(batch_inputs_entity[i])==0:
                        print(sentence_label,batch_inputs_sentence[i] )
                        break
                    batch_entity_label_ensemble = []
                    for model in self.cls_entity_predictor_pool:
                        batch_entity_label_ensemble.append(model.batch_predict(batch_inputs_entity[i]))
                    batch_entity_label_ensemble = np.array(batch_entity_label_ensemble).T.tolist()
                    for entity_label_ensemble in batch_entity_label_ensemble:
                        entity_label = cls_entity_ensemble(entity_label_ensemble)
                        if entity_label=="正类":
                            negative_entity_list.append(entity)
                    # print(negative_entity_list)
               
                batch_output_item[i]['entity'] = negative_entity_list
                batch_output_item[i]['negative'] = 0 if len(negative_entity_list)==0 else 1
            outputs.extend(batch_output_item)
        print("去重 entity 的个数", count)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("id,negative,key_entity\n")
            for item in outputs:
                f.write(item['id'] + ',' + str(item['negative'])+ ',' + ';'.join(item['entity']))
                f.write("\n")
    

def func():
    cls_entity = ClsEntity()
    input_file  = '../data/processed_data/test.jsonl'
    output_file = '../data/results/result.csv'
    cls_entity.predict(input_file, output_file)
    pass


if __name__ == '__main__':
    func()