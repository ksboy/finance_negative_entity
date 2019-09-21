# encoding: utf-8
"""
@author: banifeng 
@contact: ben_feng@shannonai.com

@version: 1.0
@file: predictor_test.py
@time: 2019-08-03 19:24

这一行开始写关于本文件的说明与解释
"""
import json

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import finance_negative_entity


class CharBertCrfEngine(object):
    def __init__(self, model_path: str, predictor_type: str, cuda_device: int = -1):
        self.predictor_type = predictor_type
        self.predictor = Predictor.from_archive(load_archive(model_path, cuda_device=cuda_device), predictor_type)

    def predict(self, content: str):
        announcement_type = self.predictor.predict_json({'passage': content})
        # print(announcement_type)
        return announcement_type['tags'][1:]




def construct_item(passage: str, labels):
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
        "passage": passage,
        "aspects": aspects,
        "opinions": opinions
    }
    return result

if __name__ == '__main__':
    model = "/home/banifeng/workspace/my_models/ner_model_2/model.tar.gz"
    type = "bert_crf_predictor"
    charBertCrfEngine = CharBertCrfEngine(model, type, 1)
    text = "体验不错"
    datas = open('bert_crf_sl_data.json').read().splitlines()

    for d in datas:
        data = json.loads(d)
        text = data["passage"].replace(" ", ",")
        text = '大品牌就是大品牌，我信赖，好，值得推荐^_^'
        labels = charBertCrfEngine.predict(text)
        item = construct_item(text, labels)

        gold_set = set([l[0] for l in data["aspects"]]+[l[0] for l in data["opinions"]])
        predict_set = set([l[0] for l in item["aspects"]]+[l[0] for l in item["opinions"]])
        if gold_set != predict_set:
            print(data["passage"])
            print('gold:', gold_set)
            print('predict:', predict_set)
            print()
        char_labels = []
        for char, label in zip(text, labels):
            char_labels.append(char+"/"+label)
        print(" ".join(char_labels))
        break