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
    tag, count = Counter(cls_tags).most_common(1)[0]
    if tag == "正类" and count >= len(cls_tags)//2 + 1:
        return tag
    else:
        return "负类"
