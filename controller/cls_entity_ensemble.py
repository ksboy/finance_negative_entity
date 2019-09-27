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
from scripts.check_file_format import check_submit
from scripts.ensemble import cls_entity_ensemble
from scripts.evaluate import Evaluator
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

CLS_ENTITY_MODEL_DIR = "~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_RoBERTa_singleEntity_"
CUDA_DEVICE = 1
NUM_CLS_ENTITY_MODELS = 5

class ClsEntity(object):
    def __init__(self):
        # logger.info("model init begin...")
        # t1 = time.time()
        # self.ner_predictor = CharBertCrfEngine(NER_MODEL_DIR, "bert_crf_predictor", CUDA_DEVICE)
        # logger.info("ner model init end.")
        # self.cls_labels_predictor = BertSeqPairClsEngine(CLS_LABELS_MODEL_DIR, "bert_seq_pair_clf", CUDA_DEVICE)
        # logger.info("cls_labels model init end.")
        # self.cls_categories_predictor = BertSeqPairClsEngine(CLS_CATEGORIES_MODEL_DIR, "bert_seq_pair_clf", CUDA_DEVICE)
        # logger.info("cls_categories model init end.")
        # self.cls_polarities_predictor = BertSeqPairClsEngine(CLS_POLARITIES_MODEL_DIR,"bert_seq_pair_clf", CUDA_DEVICE)
        # logger.info("cls_polarities model init end.")
        # t2 = time.time()
        # logger.info("model init complete, cost time %.2f s" % (t2 - t1))

        self.cls_entity_predictor_pool = []
        for i in range(NUM_CLS_ENTITY_MODELS):
            model_path = os.path.join(CLS_ENTITY_MODEL_DIR+str(i+1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CUDA_DEVICE)
            self.cls_entity_predictor_pool.append(model)


    def predict(self, input_file, output_file):
        """
        输入必须有序号，包含在passage中，
        例如：`1,凑单买的面膜，很好用，买给妈妈的`
        :param passage:
        :return:
        """
        with open(input_file, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            title = reader.fieldnames
            items=[]
            for row in reader:
                item ={}
                item['id']=row['\ufeffid']
                context = row['title'] + row['text']
                item['passage'] = context[:450]
                item['entity_list'] =[]
                item['question_list'] =[]
                for entity in row['entity'].split(";"):
                    item['entity_list'].append(entity)
                    item['question_list'].append(entity)
                items.append(item)
        
        outputs=[]
        for item in tqdm(items, desc="正在预测", ncols=80):
            # item =json.loads(item)
            passage=item['passage']
            negative_entity_list=[]
            for i, question in enumerate(item['question_list']):
                label_ensemble = []
                for model in self.cls_entity_predictor_pool:
                    # print(passage, question)
                    label_ensemble.append(model.predict(passage, question))
                label = cls_entity_ensemble(label_ensemble)
                if label=="正类":
                    negative_entity_list.append(item['entity_list'][i])
            output_item ={}
            output_item['id'] = item['id']
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
    output_file = '../data/results/result_RoBERTa.csv'
    cls_entity.predict(input_file, output_file)
    pass


if __name__ == '__main__':
    func()