# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: evaluate.py
@time: 2019-08-09 14:44

对预测的结果与真实结果进行对比评价，评价指标采用recall、precision、f1
"""
import logging
logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, truth_file=None, predict_file=None):
        '''
        本Evaluator为端到端的评价，并且分别计算出各个步骤的结果。
        TODO:读文件的过程放在__init__()函数中
        :param truth_file: 真实的标注文件（带表头）
        :param predict_file: 预测标注的文件（带表头）
        '''
        self.truth_file = truth_file
        self.predict_file = predict_file

    def cal_e2e_f1(self):
        '''计算端到端的f1'''
        truth_data = open(self.truth_file).readlines()[1:]
        predict_data = open(self.predict_file).readlines()[1:]
        t_set = set(truth_data)
        p_set = set(predict_data)
        t_num = len(truth_data)
        p_num = len(predict_data)
        p_right_num = len(t_set & p_set)
        if t_num == 0 or p_num == 0 or p_right_num==0:
            return 0.0
        precision = p_right_num/p_num
        recall = p_right_num/t_num
        f1 = 2*precision*recall/(precision+recall)
        print("e2e precision:{%.2f}" % precision, "          recall:{%.2f}" % (recall), "    f1:{%.2f}" % (f1))
        return f1

    def cal_ner_f1(self):
        '''计算ner的f1'''
        truth_data = open(self.truth_file).readlines()[1:]
        predict_data = open(self.predict_file).readlines()[1:]
        truth_entities = []
        for data in truth_data:
            data_list = data.split(",")
            assert len(data_list) == 9
            if data_list[1] != "_":
                cur = data_list[0] + data_list[1] + data_list[2]
                truth_entities.append(cur)
            if data_list[4] != "_":
                cur = data_list[0] + data_list[4] + data_list[5]
                truth_entities.append(cur)
        predict_entities = []
        for data in predict_data:
            data_list = data.split(",")
            if len(data_list) !=9:
                logger.error("len(data_list) !=9, data_list:{0}".format(data_list))
                data_list = [t for t in data_list if t!='']
                if len(data_list) != 9:
                    continue
            assert len(data_list) == 9
            if data_list[1] != "_":
                cur = data_list[0] + data_list[1] + data_list[2]
                predict_entities.append(cur)
            if data_list[4] != "_":
                cur = data_list[0] + data_list[4] + data_list[5]
                predict_entities.append(cur)
        t_set = set(truth_entities)
        p_set = set(predict_entities)
        t_num = len(truth_entities)
        p_num = len(predict_entities)
        p_right_num = len(t_set & p_set)
        if t_num == 0 or p_num == 0 or p_right_num==0:
            return 0.0
        precision = p_right_num / p_num
        recall = p_right_num / len(t_set)
        f1 = 2 * precision * recall / (precision + recall)
        print(p_set-t_set)
        print("ner precision:{%.2f}" % precision, "          recall:{%.2f}" % (recall), "    f1:{%.2f}" % (f1))
        return f1

    def cal_gen_label_item_f1(self):
        '''计算生成label的f1'''
        truth_data_with_cls = open(self.truth_file).readlines()[1:]
        predict_data_with_cls = open(self.predict_file).readlines()[1:]
        truth_data = []
        for data in truth_data_with_cls:
            assert len(data.split(",")) == 9
            split_line = data.split(",")
            cur = "".join([split_line[i] for i in [0,1,4]])
            truth_data.append(cur)
        predict_data = []
        for data in predict_data_with_cls:
            if len(data.split(",")) != 9:
                continue
            assert len(data.split(",")) == 9
            split_line = data.split(",")
            cur = "".join([split_line[i] for i in [0,1,4]])
            predict_data.append(cur)
        t_set = set(truth_data)
        p_set = set(predict_data)
        t_num = len(truth_data)
        p_num = len(predict_data)
        p_right_num = len(t_set & p_set)
        if t_num == 0 or p_num == 0 or p_right_num==0:
            return 0.0
        precision = p_right_num/p_num
        recall = p_right_num/t_num
        f1 = 2*precision*recall/(precision+recall)
        print("gen_label precision:{%.2f}" % precision, "    recall:{%.2f}" % (recall), "    f1:{%.2f}" % (f1))
        return f1

def func():
    pass


if __name__ == '__main__':
    func()