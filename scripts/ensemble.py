# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: ensemble.py
@time: 2019-08-26 20:56

这一行开始写关于本文件的说明与解释
"""
from typing import List, Union
from collections import Counter



def cls_entity_ensemble(cls_tags: List[str]):
    if len(cls_tags) == 0:
        return None
    return Counter(cls_tags).most_common(1)[0][0]

def cls_entity_ensemble_tri(cls_tags: List[List[List[Union[str, int]]]], num_threshold=4) -> List[List[Union[str, int]]]:
    if len(cls_tags) == 0:
        return None
    new_cls_tags = []
    for model_tag in cls_tags:
        for item in model_tag:
            for index in range(len(item)):
                item[index] = str(item[index])
            new_cls_tags.append("✨".join(item))
    tag_count_map = dict()
    for tag in new_cls_tags:
        tag_count_map[tag] = tag_count_map.get(tag, 0) + 1
    outputs = []
    for key, value in tag_count_map.items():
        if value >= num_threshold:
            cur_tags = key.split("✨")
            outputs.append(cur_tags)
    return outputs

if __name__ == '__main__':
    case = [[['1', '_', '_', '_', '是正品', 18, 21], ['1', '_', '_', '_', '白嫩', 28, 30], ['1', '_', '_', '_', '不错', 33, 35]],
            [ ['1', '_', '_', '_', '白嫩', 28, 30], ['1', '_', '_', '_', '不错', 33, 35]],
            [ ['1', '_', '_', '_', '白嫩', 28, 30], ['1', '_', '_', '_', '不错', 33, 35]]]
    print(cls_relations_ensemble(case))

