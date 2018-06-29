# *****************************************
# *Author       ：   jqh
# *Time         :    2018/06/25
# *File         :    text_cnn.py
# *Description  :    卷积神经网络模型
# *****************************************

import numpy as np
import tensorflow as tf

class TextCNN(object):
    """
    文本卷积神经网络模型
    使用嵌入层词向量，卷积层，池化层和softmax层
    """
    def __init__(self, sequence_length, num_classes, vocab_size, 
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        :param sequence_length: 句子的长度，固定同意长度
        :param num_classes: 输出层的神经元个数
        :param vocab_size: 词典的总数用于输入，[vocabulary大小, embedding大小]
        :param embedding_size: 嵌入层的维度
        :param filter_sizes: 卷积层需要覆盖的单次数[2,3,4]
        :param num_filters: 每个尺寸的滤波器的个数， 例如100个
        :param l2_reg_lambda: L2正则化
        """
        # 输入输出和dropout的占位符

