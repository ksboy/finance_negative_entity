# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com
@version: 1.0
@file: process_ner_data.py
@time: 2019-08-08 21:53
这一行开始写关于本文件的说明与解释
"""
import json, csv, os
from typing import List
from enum import Enum, unique
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import copy
import random
import re
import jieba
import jieba.posseg as pseg


def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


def in_passage(entity_list, passage):
    for entity in entity_list:
        if entity not in passage[:len(passage) - len(entity)]:
            return False
    return True


def gen_cleansed_data(text, entityList):
    # 去掉 不包含 实体 的 分句
    newtext = ""
    for sentence in text.split("。"):
        for entity in entityList:
            if entity in sentence and sentence not in newtext:
                newtext = newtext + sentence + "。"
    text = newtext

    # 把括号中的内容提取出来，看是否与实体有关，若无关，则删除
    pickedList = []
    pickedList.extend(["[" + _ + "]" for _ in re.findall(r"[[](.*?)[]]", text)])
    pickedList.extend(["（" + _ + "）" for _ in re.findall(r"[（](.*?)[）]", text)])
    pickedList.extend(["{" + _ + "}" for _ in re.findall(r"[{](.*?)[}]", text)])
    pickedList.extend(["【" + _ + "】" for _ in re.findall(r"[【](.*?)[】]", text)])
    pickedList.extend(["(" + _ + ")" for _ in re.findall(r"[(](.*?)[)]", text)])
    pickedList.extend(["#" + _ + "#" for _ in re.findall(r"[#](.*?)[#]", text)])
    for pick in pickedList:
        flag = False
        for entity in entityList:
            if entity in pick or pick in entity:
                flag = True
                break
        if not flag:
            text = text.replace(pick, "")

    # res = re.sub(u"\\(.*?\\)|{.*?}|\\[.*?]|【.*?】|（.*?）", "", res)  # 去掉括号中内容

    # 分词 删除 字符
    newtext = ""
    words = pseg.cut(text)
    valid_wordList = ["，", "。", "：", ",", "、", "？", "@", '#']
    for word, flag in words:
        # print(word, flag)
        # if flag == "m" and word not in "".join(entityList):
        #     continue  # 去掉量词
        if flag == "eng" and word not in "".join(entityList):
            continue  # 去掉英文
        if flag == "x" and word not in "".join(entityList):
            flag_2 = False
            for entity in entityList:
                if entity in word or word in entity:
                    newtext = newtext + word
                    flag_2 = True
                    break
            if flag_2:
                continue
            if word not in valid_wordList:
                continue  # 去掉特殊符号
        newtext = newtext + word
    text = newtext

    # # 去掉句子中的英文字符、数字以及特殊符号
    # pattern1 = re.compile(r"[A-Za-z]+")
    # pattern2 = re.compile(r"[0-9]+")
    # pattern3 = re.compile(r'[^A-Z^a-z^0-9^\u4e00-\u9fa5^,^，^。^；^：^、^#^@^?]')
    # char_list = ["年", "月", "月份", "日", "千", "万", "亿"]

    # match_list_letter = pattern1.findall(text)
    # for match_string in match_list_letter:
    #     if match_string not in entityList:
    #         text = text.replace(match_string, "")
    # match_list_digit = pattern2.findall(text)
    # for match_string in match_list_digit:
    #     if match_string not in entityList:
    #         text = text.replace(match_string, "")
    # text = pattern3.sub("", text)
    # for c in text:
    #     if c in char_list:
    #         text = text.replace(c, "")
    # newtext = text

    return text


def get_entityList(input_file="../processed_data/Train_Data.csv"):
    entityList = []
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for entity in row['entity'].split(";"):
                if entity not in entityList and entity != "":
                    entityList.append(entity)
                    jieba.add_word(entity, freq=None, tag=None)  # 向jieba字典中加入实体
    return entityList


def gen_ensemble_train_data(input_file="../processed_data/Train_Data.csv",
                            output_dir="../processed_data/ensemble_data/cls_entity/",
                            num_split=5, if_shuffle=False, type="cls_entity"):
    items = []
    entity_length = []
    count = 0
    entity_not_in_passage = 0
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            text = row['text']
            if row['title'] not in row['text']:
                text = row['title'] + row['text']
            entity_list = row['entity'].split(";")
            while '' in entity_list:
                entity_list.remove('')
            key_entity_list = row['key_entity'].split(";")
            while '' in key_entity_list:
                key_entity_list.remove('')

            text = gen_cleansed_data(text, entity_list)  # 清洗数据
            entity_length.extend([len(entity) for entity in entity_list])
            if type == 'cls_sentence':
                if len(text) > 512 - 3 - 2:
                    count += 1
                item = {}
                item['id'] = row['\ufeffid']
                item['passage'] = text[:512 - 3 - 2]
                item['question'] = ''
                if row['negative'] == "0":
                    item['label'] = '负类'
                else:
                    item['label'] = '正类'
                items.append(item)
            elif type == "cls_entity":
                for entity in entity_list:
                    if len(text) + len(entity) > 512 - 3 - 2:
                        count += 1
                    item = {}
                    item['id'] = row['\ufeffid']
                    item['passage'] = text[:512 - 3 - 2 - len(entity)]
                    item['question'] = entity
                    if entity in text and entity not in item['passage']:
                        entity_not_in_passage += 1
                    if entity in row['key_entity']:
                        item['label'] = '正类'  # 正类
                    else:
                        item['label'] = '负类'  # 负类
                    items.append(item)
            elif type == "ner":
                if len(text) > 512 - 3 - 2:
                    count += 1
                item = {}
                item['id'] = row['\ufeffid']
                item['passage'] = text[:512 - 3 - 2]
                item['key_entity'] = []
                item['entity'] = []
                for key_entity in key_entity_list:
                    # print(key_entity, item['passage'])
                    for m in re.finditer(re.escape(key_entity), item['passage']):
                        item['key_entity'].append([key_entity, m.start(), m.end()])
                for entity in entity_list:
                    # print(key_entity, item['passage'])
                    for m in re.finditer(re.escape(entity), item['passage']):
                        item['entity'].append([entity, m.start(), m.end()])
                items.append(item)

    print("max(entity_length)", max(entity_length))
    print("count(out_of_max_length)", count)
    print("entity not in passage", entity_not_in_passage)
    if if_shuffle:
        random.shuffle(items)

    for i in range(num_split):
        globals()["train_data" + str(i + 1)] = []
        globals()["dev_data" + str(i + 1)] = []
    for i, item in enumerate(items):
        cur = i % num_split + 1
        for j in range(num_split):
            if cur == j + 1:
                globals()["dev_data" + str(j + 1)].append(item)
            else:
                globals()["train_data" + str(j + 1)].append(item)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(num_split):
        write_file(globals()["train_data" + str(i + 1)], os.path.join(output_dir, "train_" + str(i + 1) + ".jsonl"))
        write_file(globals()["dev_data" + str(i + 1)], os.path.join(output_dir, "dev_" + str(i + 1) + ".jsonl"))


def gen_test_data(input_file="../processed_data/Test_Data.csv",
                  output_file="../processed_data/test_entity.jsonl",
                  type="cls_entity"):
    items = []
    entity_length = []
    count = 0
    empty_entity = 0
    redundant_entity = 0
    dup_entity = 0
    entity_not_in_passage = 0

    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            text = row['text']
            if row['title'] not in row['text']:
                text = row['title'] + row['text']
            entity_list = row['entity'].split(";")
            while '' in entity_list:
                entity_list.remove('')
            text = gen_cleansed_data(text, entity_list)  # 清洗数据

            # if row['\ufeffid'] == "04c73542":
            #     print("a")

            # temp_list = []
            # for entity in entity_list:
            #     if entity in text:
            #         temp_list.append(entity)  # 去掉text中没有的entity

            # redundant_entity += (len(entity_list) - len(temp_list))
            # entity_list = temp_list
            # temp_list = []
            # flag = True
            #
            # for curr in entity_list:
            #     for entity in entity_list:
            #         if curr in entity and curr != entity:
            #             flag = False
            #             break
            #     if flag:
            #         temp_list.append(curr)
            #     flag = True
            # dup_entity += (len(entity_list) - len(temp_list))
            # entity_list = temp_list

            entity_length.extend([len(entity) for entity in entity_list])

            if len(text) > 512 - 3 - 2:
                count += 1
            if not in_passage(entity_list, text[:512-3-2]):
                cut_text = text[:256] + text[len(text)-256+3+2:]  # 对于长度>512的字符串，取前256和后256
            else:
                cut_text = text[:512-3-2]
            item = {}
            item['id'] = row['\ufeffid']
            item['passage'] = cut_text
            item['entity'] = []
            for entity in entity_list:
                item['entity'].append(entity)
                if entity in text and entity not in cut_text:
                    entity_not_in_passage += 1
            items.append(item)

    print("max(entity_length)", max(entity_length))
    print("count(out_of_max_length)", count)
    print("entity empty", empty_entity)
    print("redundant_entity", redundant_entity)
    print("dup_entity", dup_entity)
    print("entity_not_in_passage", entity_not_in_passage)
    write_file(items, output_file)


def inspect_data(input_file="../processed_data/Train_Data.csv"):
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        count = 0
        for row in reader:
            # print(row)
            entity_list = row["key_entity"].split(";")
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    if entity_list[i] in entity_list[j]:
                        count += 1
                        print(row["id"], entity_list[i], entity_list[j])
                        # print(row["\ufeffid"], entity_list[i] , entity_list[j])
    print(count)


# def postprocess(input_file, output_file):

#     outfile = open(output_file, 'w', encoding='utf-8')

#     with open(input_file, encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         title = reader.fieldnames
#         writer= csv.DictWriter(outfile, title)

#         count = 0
#         for row in reader:
#             print(row)
#             pop_index=[]
#             entity_list = row["key_entity"].split(";")
#             for i in range(len(entity_list)):
#                 for j in range(i+1, len(entity_list)):
#                     if entity_list[i] in entity_list[j]:
#                         if row["text"].count(entity_list[i]) == row["text"].count(entity_list[j]):
#                             count +=1
#                             pop_index.append(i)
#             entity_list = [entity_list[i] for i in range(len(entity_list)) if (i not in pop_index)]
#             row["key_entity"]=  ";".join(entity_list)
#             writer.writerow(row)

#     print(count)

def func():
    # gen_ensemble_train_data(input_file="../processed_data/Train_Data.csv",
    #               output_dir="../processed_data/ensemble_data/cls_sentence_11/",
    #               num_split=5, if_shuffle=True, type='cls_sentence')
    #
    gen_ensemble_train_data(input_file="../processed_data/Train_Data.csv",
                            output_dir="../processed_data/ensemble_data/cls_entity_10/",
                            num_split=5, if_shuffle=True, type='cls_entity')

    # gen_ensemble_train_data(input_file="../processed_data/Train_Data.csv", 
    #               output_dir="../processed_data/ensemble_data/ner/",
    #               num_split=5, if_shuffle= False,type='ner')

    # gen_test_data(input_file="../processed_data/Test_Data.csv",
    #               output_file="../processed_data/test_cleansed_1015.jsonl")

    pass


def test():
    # inspect_data(input_file="../processed_data/Train_Data.csv")

    # inspect_data(input_file="../results/result_init.csv")
    # 380

    # inspect_data(input_file="../results/result.csv")
    # 112

    # postprocess(input_file="../results/result.csv",
    #             output_file="../results/result_processed.csv",)

    text = "上海宜贷网，成都易捷金融，大股东任海华已逃亡美国，在任海华的遥控指挥下，龚卓、杨帆、冯涛（彭州致和镇）等人的诈骗团伙，打着良性退出的幌子，抢劫出借人的血汗钱，建立互助金资金池，平台再用自设的马甲号掏空，三万多受害人投诉无门，有人因此重病不起，抑郁，跪求公检法立案将犯罪分子绳之以法 ?,上海宜贷网，成都易捷金融，大股东任海华已逃亡美国，在任海华的遥控指挥下，龚卓、杨帆、冯涛（彭州致和镇）等人的诈骗团伙，打着良性退出的幌子，抢劫出借人的血汗钱，建立互助金资金池，平台再用自设的马甲号掏空，三万多受害人投诉无门，有人因此重病不起，抑郁，跪求公检法立案将犯罪分子绳之以法 ?"
    entityList = "宜贷网(沪);易捷金融;宜贷网".split(";")
    # text= "其中，钱宝网以高额收益为诱饵，持续采用吸收新用户资金—兑付老用户本金及收益的方式向不特定社会公众大量非法吸储，涉及的集资参与人遍及全国各地，截至案发，未兑付的本金数额仍高达300亿元左右"
    # entityList=["钱宝"]
    # text = "【钱宝系非法集资超千亿元，张小雷被移送起诉】-新闻联播-余姚新闻网-余姚综合性门户网站"
    # entityList=["钱宝"]
    # text=",青岛中腾生物技术有限公司涉嫌虚假宣传【揭秘】借口中概股回A天人果汁疑玩股权投资骗局【案件】浙江警方查获呼之即来传销案卖袜子的老板娘竟转行挖金矿【头条】我国有消费返利风险的公司超过180家，尚有100多家未被查处【专论】查处直销挂靠切勿雷声大雨点小【头条】易商通掌门人高志华操控下的万亿财富帝国【案件】广州法院公开开庭审理一组织、领导传销活动案涉案金额达7亿多元【曝光】神奇动力水自称地铁和高铁都会用"
    # entityList= ["返利","头条","易商通"]
    res = gen_cleansed_data(text, entityList)
    print(res)


if __name__ == '__main__':
    get_entityList(input_file="../processed_data/Train_Data.csv")
    get_entityList(input_file="../processed_data/Test_Data.csv")

    func()
    # test()
