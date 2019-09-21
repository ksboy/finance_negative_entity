# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: check_file_format.py
@time: 2019-08-25 11:12

这一行开始写关于本文件的说明与解释
"""


def check_submit():
    results = open('../data/results/Result.csv', encoding='utf-8').read().splitlines()
    last = 1
    for r in results:
        split_labels = r.split(',')
        assert len(split_labels) == 5
        assert split_labels[-1] in ['正面', '负面', '中性', '_']
        cur = int(split_labels[0])
        assert cur - last <= 1
        last = cur
    print("file is right, passed the check!")
    pass

