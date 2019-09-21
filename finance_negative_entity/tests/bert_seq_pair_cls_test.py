# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: bert_seq_pair_cls_test.py
@time: 2019-08-09 19:56

这一行开始写关于本文件的说明与解释
"""



from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import finance_negative_entity


class BertSeqPairClsEngine(object):
    def __init__(self, model_path: str, predictor_type: str, cuda_device: int = -1):
        self.predictor_type = predictor_type
        self.predictor = Predictor.from_archive(load_archive(model_path, cuda_device=cuda_device), predictor_type)

    def predict(self, passage: str, question):
        result = self.predictor.predict(passage=passage, question=question)
        # print(announcement_type)
        return result

if __name__ == '__main__':
    model = "/home/whou/workspace/my_models/view_rematch/laptop/cls_relations/cls_relations_model_4/model.tar.gz"
    type = "bert_seq_pair_clf"
    charBertCrfEngine = BertSeqPairClsEngine(model, type, 1)
    # {"passage": "很好，遮暇功能差一些，总体还不错", "question": "遮暇功能很好", "label": "负类"}
    passage = "不论是cpu，显卡，还是屏幕，都堪称完美，散热也不错，玩lol时显卡都懒得跑，感觉用5年都不落伍！"
    question = "显卡堪称完美"
    labels = charBertCrfEngine.predict(passage, question)
    print(labels)