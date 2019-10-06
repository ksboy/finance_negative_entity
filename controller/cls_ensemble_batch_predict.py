# encoding: utf-8
"""
@author: whou 
@contact: whou@126.com

@version: 1.0
@file: viewpoint_mining.py
@time: 2019-08-09 19:39

这一行开始写关于本文件的说明与解释
"""

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from typing import List, Tuple
from data.preprocess.gen_relation_data import gen_one_review_relation_data
# import finance_negative_entity 是必要的，虽然没有显式用到，但是它会告诉allennlp一个搜索路径。
import finance_negative_entity
import time
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict
from scripts.check_file_format import check_submit
from scripts.ensemble import ner_tag_ensemble, cls_ensemble, cls_relations_ensemble
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

class CharBertCrfEngine(object):
    def __init__(self, model_path: str, predictor_type: str, cuda_device: int = -1):
        self.predictor_type = predictor_type
        self.predictor = Predictor.from_archive(load_archive(model_path, cuda_device=cuda_device), predictor_type)

    def predict(self, content: str):
        announcement_type = self.predictor.predict_json({'passage': content.rstrip()})
        return announcement_type['tags'][1:]

    def batch_predict(self, contents: List[str]):
        json_lists = [{"passage":content} for content in contents]
        outputs = self.predictor.predict_batch_json(json_lists)
        # 将CLS位置对应的tag去掉
        results = []
        for o in outputs:
            results.append(o['tags'][1:])
        return results

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

NER_MODEL_DIR = "/home/whou/workspace/my_models/ner_roberta_tri_training_second/ner_model_"
CLS_RELATIONSS_MODEL_DIR = "/home/banifeng/workspace/my_models/cls_relations_model/cls_relations_"
CLS_CATEGORIES_MODEL_DIR = "/home/banifeng/workspace/my_models/cls_categories_model/cls_categories_model_"
CLS_POLARITIES_MODEL_DIR = "/home/banifeng/workspace/my_models/cls_polarities_model/cls_polarities_model_"
# CUDA_DEVICE = 1
NER_CUDA_DEVICE = 0
CLS_CUDA_DEVICE = 1
NUM_NER_MODELS = 5
NUM_CLS_RELATIONS_MODELS = 5
NUM_CLS_CATEGORIES_MODELS = 5
NUM_CLS_POLARITIES_MODELS = 5

class ExtractViewPoint(object):
    def __init__(self):
        logger.info("model init begin...")
        self.ner_predictor_pool = []
        self.num_reviews = -1
        for i in range(NUM_NER_MODELS):
            model_path = os.path.join(NER_MODEL_DIR+str(i+1), "model.tar.gz")
            model = CharBertCrfEngine(model_path, "bert_crf_predictor", NER_CUDA_DEVICE)
            self.ner_predictor_pool.append(model)

        self.cls_realations_predictor_pool = []
        for i in range(NUM_CLS_RELATIONS_MODELS):
            model_path = os.path.join(CLS_RELATIONSS_MODEL_DIR+str(i+1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", NER_CUDA_DEVICE)
            self.cls_realations_predictor_pool.append(model)

        self.cls_polarities_predictor_pool = []
        for i in range(NUM_CLS_POLARITIES_MODELS):
            model_path = os.path.join(CLS_POLARITIES_MODEL_DIR + str(i + 1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CLS_CUDA_DEVICE)
            self.cls_polarities_predictor_pool.append(model)

        self.cls_categories_predictor_pool = []
        for i in range(NUM_CLS_CATEGORIES_MODELS):
            model_path = os.path.join(CLS_CATEGORIES_MODEL_DIR + str(i + 1), "model.tar.gz")
            model = BertSeqPairClsEngine(model_path, "bert_seq_pair_clf", CLS_CUDA_DEVICE)
            self.cls_categories_predictor_pool.append(model)


    def construct_item(self, passage_id, passage: str, labels: List):
        """
        根据输入的passage和labels构建item，
        我在巴拉巴拉...
        ['B-ASP', 'I-ASP', 'I-ASP', 'I-ASP', ..., 'I-OPI', 'I-OPI', 'O']
        构造结果示例如下：
        {
            'passage': '使用一段时间才来评价，淡淡的香味，喜欢！',
            'aspect': [['香味', 14, 16]],
            'opinion': [['喜欢', 17, 19]]
        }
        :return:
        """
        assert len(passage) == len(labels)
        aspects, opinions = [], []
        for i, char, label in zip(range(len(passage)), passage, labels):
            if label == "O":
                continue
            elif label.startswith("B"):
                if label.endswith("ASP"):
                    aspects.append([char, i])
                elif label.endswith("OPI"):
                    opinions.append([char, i])
                else:
                    raise Exception("label must be in set {'B-ASP', 'I-ASP', 'B-OPI', 'I-OPI', 'O'}.")
            elif label.endswith("ASP"):
                if (i==0 or not labels[i-1].endswith("ASP")):
                    aspects.append([char, i])
                else:
                    aspects[-1][0] += char
            elif label.endswith("OPI"):
                if (i==0 or not labels[i-1].endswith("OPI")):
                    opinions.append([char, i])
                else:
                    opinions[-1][0] += char
            else:
                raise Exception("label must be in set {'B-ASP', 'I-ASP', 'B-OPI', 'I-OPI', 'O'}.")
        aspects = [[aspect[0], aspect[1], aspect[1]+len(aspect[0])] for aspect in aspects]
        opinions = [[opinion[0], opinion[1], opinion[1] + len(opinion[0])] for opinion in opinions]
        result = {
            "passage_id": passage_id,
            "passage": passage,
            "aspects": aspects,
            "opinions": opinions
        }
        return result

    def gen_one_cls_data(self, review_str, split_label_list: str):
        """
        很好，超值，很好用
        2,_, , ,很好,0,2,整体,正面
        :param review:
        :param label:
        :return:
        """
        result = dict()
        review_str = review_str.replace(" ", "，")
        result["passage"] = review_str.rstrip()
        question = ""
        assert len(split_label_list) == 7
        question += split_label_list[1] + "，" + split_label_list[4] + "。"
        result["question"] = question
        return result

    def gen_one_cls_data_passage_based(self, split_label_list: str):
        """
        很好，超值，很好用
        2,_, , ,很好,0,2,整体,正面
        :param review:
        :param label:
        :return:
        """
        result = dict()
        review_str = split_label_list[1]
        review_str = review_str.replace(" ", "，")
        result["passage"] = review_str.rstrip()
        question = ""
        assert len(split_label_list) == 8
        question += split_label_list[2] + "，" + split_label_list[5] + "。"
        result["question"] = question
        return (review_str.rstrip(), question)
        # return results

    def gen_batch_review_relation(self, items):

        # {"passage": "很好，遮暇功能差一些，功能很好，总体还不错",
        # "aspects": [["遮暇功能", 3, 7]，["遮暇功能", 11, 15]],
        # "opinions": [["很好", 0, 2],["差一些", 7, 10] , ["很好", 15, 17], ["还不错", 20, 23]]}
        batch_review_data =[]
        for item in items:
            one_review_data = gen_one_review_relation_data(item)
            batch_review_data.append(one_review_data)
        """
        
        [[1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错', '遮暇功能', 3, 7, '很好', 0, 2], 
         [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错', '遮暇功能', 3, 7, '差一些', 7, 10], 
         [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错', '遮暇功能', 11, 15, '还不错', 20, 23],
         [2, ...] ,
         [2, ...] ,
          ...   ]]
        """

        # 判断 relation_candidate 的类别
        batch_inputs = []
        inputs_index =[]
        for one_review_data in batch_review_data:
            one_inputs = []
            for relation_candidate in one_review_data:
                passage = relation_candidate[1]
                question = relation_candidate[2] + relation_candidate[5]
                # {"passage": "很好，遮暇功能差一些，总体很好，总体还不错", "question": "遮暇功能很好", "label": "负类"}
                one_inputs.append((passage, question))
            batch_inputs.extend(one_inputs)
            inputs_index.append(len(one_inputs))
        if batch_inputs[0] == ('质量非常好价格不贵是正品', '质量非常好'):
            print()
        batch_labels = self.cls_relations_predictor.batch_predict(batch_inputs)
        batch_labels_reformat = []
        start_index = 0
        for index in inputs_index:
            batch_labels_reformat.append(batch_labels[start_index: start_index + index])
            start_index += index

        batch_outputs =[]
        for labels_index in range(len(batch_labels_reformat)):
            labels = batch_labels_reformat[labels_index]
            outputs = []
            aspect_start_index_in_output = []
            opinion_start_index_in_output = []
            for label_index in range(len(labels)):
                label= labels[label_index]
                if label == '正类':
                    # print(label)
                    # [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 3, 7, '很好', 0, 2]
                    relation_candidate= batch_review_data[labels_index][label_index]
                    outputs.append(relation_candidate)
                    aspect_start_index_in_output.append(relation_candidate[3])
                    opinion_start_index_in_output.append(relation_candidate[6])
            """
            [[1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 3, 7, '很好', 0, 2], 
             [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 3, 7, '差一些', 7, 10], 
             [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 3, 7, '很好', 15, 17], 
             [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 11, 15, '差一些', 7, 10], 
             [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 11, 15, '很好', 0, 2], 
             [1, '很好，遮暇功能差一些，遮暇功能很好，总体还不错',  '遮暇功能', 11, 15, '很好', 15, 17]] 
            """

            # 去重: 同一index的aspect对应内容相同的opinion
            output_pop_index = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    if outputs[i][0] == outputs[j][0] and outputs[i][3] == outputs[j][3] and outputs[i][5] == outputs[j][5]:
                        d_i = outputs[i][6] - outputs[i][3]
                        d_j = outputs[j][6] - outputs[i][3]
                        if d_i > d_j:
                            output_pop_index.append(j)
                            opinion_start_index_in_output.remove(outputs[j][6])
                        else:
                            output_pop_index.append(i)
                            opinion_start_index_in_output.remove(outputs[i][6])
            outputs = [outputs[i] for i in range(len(outputs)) if (i not in output_pop_index)]

            # 某aspect或opinion在outputs没有出现，需要加上"_"
            item = items[labels_index]
            for aspect in item['aspects']:
                if aspect[1] not in aspect_start_index_in_output:
                    outputs.append([item["passage_id"]] +[item["passage"]] + aspect + ['_', '_', '_'])
            for opinion in item['opinions']:
                if opinion[1] not in opinion_start_index_in_output:
                    outputs.append([item["passage_id"]] +[item["passage"]] + ['_', '_', '_'] + opinion)
            batch_outputs.extend(outputs)
        if batch_outputs == []:
            print()
        return batch_outputs


    def batch_e2e_extract(self, passages: List[str]):
        """
        输入必须有序号，包含在passage中，
        例如：`1,凑单买的面膜，很好用，买给妈妈的`
        :param passage:
        :return:
        """
        split_passages = [passage.split(",", 1) for passage in passages]
        sent_tags = []
        for model in self.ner_predictor_pool:
            sent_tags.append(model.batch_predict([s[1].rstrip() for s in split_passages]))
        output_sent_tags = []
        for index in range(len(sent_tags[0])):
            output_sent_tags.append([model[index] for model in sent_tags])
        items = []
        for i in range(len(passages)):
            tags = ner_tag_ensemble(output_sent_tags[i])
            items.append(self.construct_item(split_passages[i][0], split_passages[i][1].rstrip(),tags))
        relation_labels_ensemble = []
        for model in self.cls_realations_predictor_pool:
            self.cls_relations_predictor = model
            relation_labels_ensemble.extend(self.gen_batch_review_relation(items))
        passage_labels_map = defaultdict(list)
        for label in relation_labels_ensemble:
            passage_labels_map[label[0]].append(label)
        output_labels = []
        for key, values in passage_labels_map.items():
            output_labels.extend(cls_relations_ensemble([values], model_num=NUM_CLS_RELATIONS_MODELS))

        # relation_labels = cls_relations_ensemble(relation_labels_ensemble, model_num=NUM_CLS_RELATIONS_MODELS)
        relation_labels = output_labels
        cls_datas = [self.gen_one_cls_data_passage_based(label) for label in relation_labels]
        categories = []
        if cls_datas ==[]:
            print()
        for model in self.cls_categories_predictor_pool:
            categories.append(model.batch_predict(cls_datas))
        np_categories = np.array(categories)
        np_categories = np_categories.T
        categories = np_categories.tolist()
        for i, label in enumerate(relation_labels):
            label.append(cls_ensemble(categories[i]))

        polarities = []
        for model in self.cls_polarities_predictor_pool:
            polarities.append(model.batch_predict(cls_datas))
        np_polarities = np.array(polarities)
        np_polarities = np_polarities.T
        polarities = np_polarities.tolist()
        for i, label in enumerate(relation_labels):
            label.append(cls_ensemble(polarities[i]))

        for r in relation_labels:
            del r[1]
        return relation_labels

    def e2e_extract_file(self, reviews_file, output_file, submit_file,  jump_first_line=True):
        reviews = open(reviews_file, encoding='utf-8').readlines()
        if jump_first_line:
            reviews = reviews[1:]
        self.num_reviews = len(reviews)
        batch_size = 64
        i = 0
        batch_reviews = []
        while i < len(reviews):
            batch_reviews.append(reviews[i:i+batch_size])
            i += batch_size
        outputs = []
        for batch_review in tqdm(batch_reviews, desc="正在预测", ncols=80):
            labels = self.batch_e2e_extract(batch_review)
            outputs.extend(labels)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities\n")
            for obj in outputs:
                for i in [2, 3, 5, 6]:
                    if isinstance(obj[i], int):
                        obj[i] = str(obj[i])
                f.write(','.join(obj))
                f.write("\n")
        self.gen_submit_file(output_file, submit_file)

    def gen_submit_file(self, result_file, submit_file):
        datas = open(result_file,  encoding='utf-8').read().splitlines()[1:]
        output = []
        for data in datas:
            split_data = data.split(',')
            assert len(split_data) == 9
            output.append([split_data[i] for i in range(len(split_data)) if i in [0, 1, 4, 7, 8]])
        index = 0
        result = []
        for i in range(1, self.num_reviews+1):
            flag = False
            while index < len(output) and output[index][0] == str(i):
                flag = True
                result.append(output[index])
                index += 1
            if not flag:
                result.append([str(i), '_', '_', '_', '_'])
        outputs = []
        for i, data in enumerate(result):
            if i < 1 or data != result[i - 1]:
                outputs.append(data)
        result = outputs
        with open(submit_file, 'w',  encoding='utf-8') as f:
            for i, obj in enumerate(result):
                f.write(','.join(obj))
                if i != len(result) - 1:
                    f.write("\n")
        pass

def func():
    e2e = ExtractViewPoint()
    reviews_file = '../data/TEST/Test_reviews.csv'
    # reviews_file = '../data/TRAIN_2/Dev_reviews.csv'
    result_file = '../data/results/result_test.csv'
    submit_file = '../data/results/Result.csv'
    e2e.e2e_extract_file(reviews_file, result_file, submit_file)
    check_submit()
    # evaluator = Evaluator(reviews_file.replace('reviews', 'labels'), result_file)
    # evaluator.cal_ner_f1()
    # evaluator.cal_gen_label_item_f1()
    # evaluator.cal_e2e_f1()
    pass


if __name__ == '__main__':
    func()


test = [[1,2,3,4], [2,3,4,5],[1,2,3,4], [2,3,4,5]]
for i in range(len(test)):
    if i == 1:
        del test[i]