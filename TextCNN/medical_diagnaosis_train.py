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
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                    )
            print("Using the AFC-TextCNN model.")
        else:
            cnn = TextCNN(
                    Sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size =len(vocabulary),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                    )
            print("Using the TextCNN model.")

        #定义 训练步骤
        """train-op在这里是一个新创建的操作，可以运行它来对参数执行梯度更新，
        train-op的每一次执行都是一个训练步骤
        TensorFlow会自动计算出哪些变量是可训练的可计算梯度的
        通过定义global_step变量并将其传递给优化器，允许Tensorflow为我们处理训练
        步骤的计数
        每次执行train-op时，全局变量递增1
        """
        global_op = tf.Variable(0, name="global_dtep", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, 
                global_step=global_step)

        # Keep trick of gradients values and sparsity 可选
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                        '{}/grad/hist'.format(v.name, g))
                sparsity_summary = tf.summary.scalar(
                        '{}/grad/sparsity'.format(v.name), 
                        tf.nn.zero_fraction(g))
                summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 输出目录
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 
            'runs', timestamp))
        print('Writing to {}\n'.format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accurary', cnn.accurary)
        
        # 训练可视化
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, 
            grad_summaries_mergerd])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        trian_summary_writer = tf.summary.FileWriter(dev_summary_dir, 
                sess.graph)

        # Checkpoint 目录 Tensorflow 默认此目录已经存在，如果没有没有需要创建
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints/'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), 
                max_to_keep=FLAGS.num_checkpoints)

        # Writer vocubarary
        with open(os.path.join(os.path.curdir, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(vocabulary, f)
        
        # 初始化所有的变量
        sess.run(tf.global_variables_initializer())

        '''
        def train_step(x_batch, y_batch):
            """
            单步训练
            :param x_batch,y_batch
            :return 
            """





        # 生成 batch
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)),
                FLAGS.batch_size, FlAGS.num_epoches)

        # 统计参数的数量
        print("The number of paramaters is :{}".format(
            np.sum([np.prob(v.get_shape().as_list()) \
            for v in tf.trainable_variables()])))
        
        # 训练epoches次
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_train, y_train)  # 调用类内函数 train_step()
            current_step = tf.train.glabal_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print('\nEvaluation:')
                step, loss_total, accuracy_total = 0, 0.0, 0.0
                for dev_batch in data_helper.batch_iter(list(zip(x_dev,y_dev)),
                        FLAGS.batch_size, 1):
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    step, loss, accuracy = dev_step(x_dev_batch, y_dev_batch,
                            writer=dev_summary_writer)
                    loss_total += loss*FLAGS.batch_size
                    accuracy_total += accuracy * FLAGS.batch_size
                time_str = datetime.datetime.now().isoformat()
                print('{}:step {}, loss:{:g}, acc:{:g}'.format(time_str,
                    step, loss_total / len(x_dev), 
                    accuracy_total / len(x_dev)))
            
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_dir, 
                        global_step = current_step)
                print('Saved model checkpoint to {}\n'.format(path))
        '''
