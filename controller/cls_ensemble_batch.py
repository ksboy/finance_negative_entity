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
# import finance_negative_entity 是必要的，虽然没有显式用到，但是它会告诉allennlp一个搜索路径。
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
        batch_size = 64
        with open(input_file, encoding='utf-8') as infile:
            items=[]
            for row in infile:
                row = json.loads(row)
                item ={}
                item['id']=row['id']
                item['passage'] = row['passage']
                item['entity_list'] =[]
                for entity in row['entity']:
                    item['entity_list'].append(entity)
                # print(item)
                items.append(item)
                
        batch_items_list = []
        i = 0
        while i < len(items):
            batch_items_list.append(reviews[i:i+batch_size])
            i += batch_size

        outputs=[]
        for batch_items in tqdm(batch_items_list, desc="正在预测", ncols=80):
            # item =json.loads(item)
            batch_inputs = []
            batch_output_item = []
            for item in batch_items:
                passage=item['passage']
                negative_entity_list=[]
                output_item ={}
                output_item['id'] = item['id']
                # {"passage": "很好，遮暇功能差一些，总体很好，总体还不错", "question": "遮暇功能很好", "label": "负类"}
                batch_inputs.append([passage,""])
                batch_output_item.append(output_item)
            
            for i in range(len(batch_inputs)):
                batch_inputs[i][0] = batch_inputs[i][0][:512-3-2]
            batch_sentence_label_ensemble = []
            for model in self.cls_sentence_predictor_pool:
                batch_sentence_label_ensemble.append(model.batch_predict())
            sentence_label = cls_entity_ensemble(sentence_label_ensemble)
            if sentence_label == "负类":
                output_item['entity'] = []
                output_item['negative'] = 0 
            else:
                print("cls_entity")
                for i, entity in enumerate(item['entity_list']):
                    entity_label_ensemble = []
                    for model in self.cls_entity_predictor_pool:
                        entity_label_ensemble.append(model.predict(passage[:512-3-2-len(entity)], entity))
                    entity_label = cls_entity_ensemble(entity_label_ensemble)
                    if entity_label=="正类":
                        negative_entity_list.append(entity)
                print(negative_entity_list)
               
            output_item['entity'] = negative_entity_list
            output_item['negative'] = 0 if len(negative_entity_list)==0 else 1
            outputs.append(output_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("id,negative,key_entity\n")
            for item in outputs:
                f.write(item['id'] + ',' + str(item['negative'])+ ',' + ';'.join(item['entity']))
                f.write("\n")
    

def func():
    cls_entity = ClsEntity()
    input_file  = '../data/processed_data/Test_Data.csv'
    output_file = '../data/results/result_.csv'
    cls_entity.predict(input_file, output_file)
    pass


if __name__ == '__main__':
    func()