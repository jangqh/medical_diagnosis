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
        self.input_x = tf.placeholder(tf.int32, shape=[None, sqeuence_length],
                name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes],
                name='input_y')
        self.droupout_keep_prob = tf.placeholder(tf.float32, 
                name='dropout_keep_prob')

        # L2正则化
        l2_loss = tf.constant(0.0)

        # 嵌入层
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, 
                embedding_size],-1.0, 1.0), name='W')
            # 输入[输入张量， 张量对应的索引]
            # 输出[None, sequence_length，embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 卷积需要四维tensor
            # expand_dim和reshape都可以改变维度，但是在构建具体的图时，如果没有具体的值，使用reshape则会报错
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,
                    -1)

        # 为每一个filter size构建卷积层和最大池化层
        pooled_outputs = []
        for i,filter_size in enumerate(filter_size):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # 卷积层
                # 卷积核，[卷积核的高度和宽度，通道个数，卷积核个数]
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                        name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]),name='b')
                # padding: SAME 表示用0来填充，VALID用来表示不填充
                # strides: [batch, height,width,channels]，batch和channels为1
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W,
                        strides=[1,1,1,1], padding='VALID',name='relu')
                # relu
                h = tf.nn.relu(tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],  strides=[1,1,1,1],padding='VALID', name='pool'))
                pooled_outputs.append(pooled)

        #  组合所有的池化层特征
        num_filter_total = num_filters * len(filter_sizes)
        # 在pooled_outputs 的最后一个维度上连接
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

        # 增加dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 最后unnormalized scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[num_filter_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.prediction = tf.argmax(self.scores, 1, name='predictions')

        # 计算 平均 cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entopy_with_logit_v2(
                    logits=self.scores,
                    labels=self.input_y)
            #l2 正则化
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(
                self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                'float'),name='accuracy')

            

            

