#!/usr/bin/python
# coding: utf-8

'''
Created on 2017-12-26
Update  on 2017-12-26
Author: xiaomingnio
Github: https://github.com/apachecn/kaggle
Project: https://www.kaggle.com/c/word2vec-nlp-tutorial
'''
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup


def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words


root_dir = "/opt/data/kaggle/getting-started/word2vec-nlp-tutorial"
# 载入数据集
train = pd.read_csv('%s/%s' % (root_dir, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
test = pd.read_csv('%s/%s' % (root_dir, 'testData.tsv'), header=0, delimiter="\t", quoting=3)
print(train.head())
print(test.head())
