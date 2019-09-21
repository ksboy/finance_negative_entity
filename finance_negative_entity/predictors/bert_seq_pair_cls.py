# encoding: utf-8
"""
@author: banifeng
@contact: banifeng@126.com

@version: 1.0
@file: bert_seq_pair_cls.py
@time: 2019-08-09 16:39
"""
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('bert_seq_pair_clf')
class BertSeqPairClsPredictor(Predictor):
    def predict(self, passage, question) -> JsonDict:
        return self.predict_json({"passage": passage, "question": question})

    def predict_batch(self, json_list):
        return self.predict_batch_json(json_list)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        passage = json_dict['passage']
        question = json_dict['question']
        return self._dataset_reader.text_to_instance(passage=passage, question=question)
