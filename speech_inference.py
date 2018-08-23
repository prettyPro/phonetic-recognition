import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import random

# 参数
lr = 0.001                  # 学习率
batch_size = 10             # batch大小
n_inputs = 40               # MFCC数据输入
num_epochs = 100            # 迭代周期
n_hidden = 128              # 隐藏单元个数,太少了的话预测效果不好,太多了会overfitting,这里普遍取128
n_classes = 21              # 分类类别个数
dropout_keep_prob = 0.5     # dropout参数,为了减轻过拟合的影响，我们用dropout,它可以随机地关闭一些神经元
evaluate_every = 10         # 多少step测试一次
checkpoint_every = 500      # 多少step保存一次模型
num_checkpoints = 2         # 最多保存多少个模型

#通过tf.get_variable来获取变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    #当给出正则化函数时，将当前变量的正则化损失加入losses的集合中
    #add_to_collection函数讲一个张量加入一个集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights