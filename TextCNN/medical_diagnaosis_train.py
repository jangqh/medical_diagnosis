## ***********************************************
## * Author     :   jqh
## * File_name  :   medical_diagnosis_train.py
## * Description:   CNN模型 对症状分类
## ***********************************************
import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import pickle
import data_helper
from tensorflow.contrib import learn
from text_cnn import TextCNN

###############################################
# 参数解析

# 加载数据
tf.flags.DEFINE_string("train_data_file", "../data/train.txt", "训练数据")
tf.flags.DEFINE_string("test_data_file", "../data/test.txt", "测试数据")

# 选择模型
tf.flags.DEFINE_string("model", "TextCNN", "选择模型")

# 模型超参数
tf.flags.DEFINE_integer("embedding_dim", "128", "词向量维度，不只是词向量")
tf.flags.DEFINE_integer("filter_sizes", "2,3,4", "comma-separated滤波器大小")
tf.flags.DEFINE_integer("num_filters", 128, "每种尺寸滤波器的个数")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 正则化")

# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epoch", 20000, "训练次数")
tf.flags.DEFINE_integer("evaluate_every", 100, "多少步后评估模型")
tf.flags.DEFINE_interger("num_checkpoints", 100, "Num of check to store")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow devices soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters: ")
for attr, value in sorted(FLAGS.flags_value_dcit().items()):
    print("{}={}".format(attr.upper(), value))
print('''''''''''')


# 加载数据
print("Loading data...")
print(FLAGS.train_data_file, FLAGS.test_data_file)
vocabulary, train_data, train_label, test_data, test_label = \
        data.helper.load_data_and_labels_chinese(FLAGS.train_data_file,
                FLAGS.test_data_file)
shuffle_indices = np.random.permutation(np.arange(len(train_data)))
train_data = np.array(train_data)
x_train, y_train = train_data[shuffle_indices], train_label[shuffle_indices]
x_dev, y_dev = np.array(test_data), test_label
print("Train/Dev split:{:d}/{:d}".format(len(y_train), len(y_dev)))

################################################
# 训练
with tf.Graph().as_default():
    """设置允许TensorFlow在首设备不存在时执行特定操作时回落到设备上
    如果代码在GPU上放置操作，然并卵木有GPU，使用了allow_soft_placement
    系统会自动选择CPU来训练，若果不适用allow_soft_placement则会出错
    设置log_device_placement会记录在GPU或CPU上的操作log
    """
    seesion_conf = rf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_devices_placement = FLAGS.log_device_placement)
    sess = tf.Session(configs = session_conf)
    with sess.as_default():
        if FLAGS.model == 'MF-TextCNN':
            cnn = MFTextCNN(
                    Sequence_length=x_train.shape[1],
                    num_classes = y_train.shape[1],
                    vocab_size = len(vocabulary),
                    embedding_size = FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                    num_filters = FLAGS.num_filter,
                    l2_reg_lambda = FLAGS.l2_reg_lambda
                    )
            print("Using the MF-TextCNN model.")
        elif FLAGS.model == 'AFC-TextCNN':
            cnn = AFCTextCNN(
                    Sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size =len(vocabulary),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(',')))
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                    )
            print("Using the AFC-TextCNN model.")
        else::
            cnn = TextCNN(
                    Sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size =len(vocabulary),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(',')))
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                    )
            print("Using the TextCNN model.")





