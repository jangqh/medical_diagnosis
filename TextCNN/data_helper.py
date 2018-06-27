# *************************************
# *Author     : jqh
# *File_name  : data_helper.py
# *Decription :  数据集的加载
# ************************************

import numpy as np
import re
import string
import collections
from zhou.hanzi import punctuation
from sklearn.processing import LabelEncoder
from sklearn.processing import OneHotEncoder

def clean_str(s):
    """分词，字符清理，except for SSI
    :param s:
    :return:
    """
    s = re.sub(r"[^A-Za-z0-9(), !?\'\`]", " ", s) 


def load_data_and_labels(positive_data_file, negative_data_file):
    """加载MR polarity 数据，返回分词和标签
    :param 训练测试数据
    :return 词和标签
    """
    # 加载数据
    positive_examples = list(open(postive_data_file, 'r', encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, 'r', encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # split by words

    # 生成标签


def load_data_and_labels_chinese(train_data_file, test_data_file):
    """加载分诊数据，分词和分标签
    参数：训练和测试数据集
    返回：切分的词和标签
    """
    # 加载数据
    word = []
    contents = []
    test_data = []
    train_data = []
    labels = []

    # 生成训练数据
    with open(train_data_file, 'r', encoding='utf-8') as f:
        for line in f:






if __name__ == '__main__':
    train_data_file = '../data/train.txt'
    test_data_file = '../data/test.txt'
    load_data_and_labels_chinese(train_data_file, test_data_file)

