# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: help.py
@time: 2019-08-18 09:22

这一行开始写关于本文件的说明与解释
"""
from collections import defaultdict

def func():
    datas = open('result_test.csv').read().splitlines()[1:]
    output = []
    for data in datas:
        split_data = data.split(',')
        assert len(split_data) == 9
        output.append([split_data[i] for i in range(len(split_data)) if i in [0, 1, 4, 7, 8]])
    index = 0
    result = []
    for i in range(1, 2238):
        flag = False
        while index < len(output) and output[index][0] == str(i):
            flag = True
            output[index][3], output[index][4] = output[index][4], output[index][3]
            result.append(output[index])
            index += 1
        if not flag:
            result.append([str(i), '_', '_', '_', '_'])
    outputs = []
    for i, data in enumerate(result):
        if i < 1 or data != result[i-1]:
            outputs.append(data)
    result = outputs
    with open('Result.csv', 'w') as f:
        # f.write("id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities\n")
        for i, obj in enumerate(result):
            # for i in [2, 3, 5, 6]:
            #     if isinstance(obj[i], int):
            #         obj[i] = str(obj[i])
            f.write(','.join(obj))
            if i != len(result)-1:
                f.write("\n")
    pass

def check_result():
    results = open('Result.csv').read().splitlines()
    id_labels_map = defaultdict(list)
    for r in results:
        id, label = r.split(',',1)
        id_labels_map[id].append(label)
    for id, values in id_labels_map.items():
        aspect_opinion_map = defaultdict(list)
        for v in values:
            split_label = v.split(',')
            aspect_opinion_map[split_label[0]].append(split_label[1])
        for key,opinions in aspect_opinion_map.items():
            if len(opinions)>1 and '_' in opinions:
                print(id, values)

def check_gold_labels():
    results = open('../TRAIN/Train_labels.csv').read().splitlines()[1:]
    id_labels_map = defaultdict(list)
    for r in results:
        id, label = r.split(',',1)
        id_labels_map[id].append(label)
    for id, values in id_labels_map.items():
        aspect_opinion_map = defaultdict(list)
        for v in values:
            split_label = v.split(',')
            aspect_opinion_map[split_label[0]].append(split_label[3])
        for key,opinions in aspect_opinion_map.items():
            if len(opinions)>1 and '_' in opinions:
                print(id, values)
    for id, values in id_labels_map.items():
        opinion_aspect_map = defaultdict(list)
        for v in values:
            split_label = v.split(',')
            opinion_aspect_map[split_label[3]].append(split_label[0])
            # print(opinion_aspect_map)
        for key, aspects in opinion_aspect_map.items():
            if len(aspects) > 1 and '_' in aspects:
                print(id, values)

def compare_result_file():
    hit_data = open('Result.csv', encoding='utf-8').readlines()
    zju_data = open('Result_zju.csv', encoding='utf-8').readlines()
    h_set = set(hit_data)
    z_set = set(zju_data)
    print("h_set - z_set")
    print(h_set-z_set)
    print("z_set - h_set")
    print(z_set - h_set)
    print("\n\n\n")
    print("len(h_set&z_set)", len(h_set&z_set))
    print("len(h_set)==", len(h_set))
    print("len(z_set)==", len(z_set))

if __name__ == '__main__':
    compare_result_file()
    # func()
    # check_result()
    # check_gold_labels()