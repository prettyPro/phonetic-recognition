
# coding: utf-8

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import random

# 超参数
lr = 0.001                  # 学习率
batch_size = 10             # batch大小
n_inputs = 40               # MFCC数据输入
num_epochs = 100            # 迭代周期
n_hidden = 128              # 隐藏单元个数,太少了的话预测效果不好,太多了会overfitting,这里普遍取128
n_classes = 21              # 分类类别个数
dropout_keep_prob = 0.5     # dropout参数,为了减轻过拟合的影响，我们用dropout,它可以随机地关闭一些神经元
evaluate_every = 10         # 多少step测试一次
checkpoint_every = 500      # 多少step保存一次模型
num_checkpoints = 1         # 最多保存多少个模型

# 加载训练用的特征和标签
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')

# 计算最长的step,分为step帧
wav_max_len = max([len(feature) for feature in train_features])
print("max_len:", wav_max_len)

# 填充0
tr_data = []
for mfccs in train_features:
    while len(mfccs) < wav_max_len:
        mfccs.append([0] * n_inputs)
    tr_data.append(mfccs)
tr_data = np.array(tr_data)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(tr_data)))
x_shuffled = tr_data[shuffle_indices]
y_shuffled = train_labels[shuffle_indices]

# 数据集切分为两部分
dev_sample_index = -1 * int(0.2 * float(len(y_shuffled)))
train_x,  test_x = x_shuffled[:dev_sample_index],  x_shuffled[dev_sample_index:]
train_y,  test_y = y_shuffled[:dev_sample_index],  y_shuffled[dev_sample_index:]


#参数：批次，序列号（分帧的数量），每个序列的数据(batch, step, input)
x = tf.placeholder("float",  [None,  wav_max_len, n_inputs])
y = tf.placeholder("float",  [None])
dropout = tf.placeholder(tf.float32)

# labels转one_hot格式
one_hot_labels = tf.one_hot(indices=tf.cast(y,  tf.int32),  depth=n_classes)

# 定义RNN网络
# 初始化权值和偏置
weights = tf.Variable(tf.truncated_normal([n_hidden,  n_classes],  stddev=0.1))
biases = tf.Variable(tf.constant(0.1,  shape=[n_classes]))

# 网络层数
cell_stack = []
cell1 = tf.contrib.rnn.GRUCell(n_hidden)
cell_stack.append(cell1)
cell2 = tf.contrib.rnn.GRUCell(n_hidden)
cell_stack.append(cell2)
cell3 = tf.contrib.rnn.GRUCell(n_hidden)
cell_stack.append(cell3)

cell = tf.contrib.rnn.MultiRNNCell(cell_stack)
outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
# final_state[0]:cell state
# final_state[1]:hidden_state
# 预测值
prediction = tf.matmul(final_state[1], weights) + biases

# loss
# cross_entropy = tf.reduce_mean(tf.square(tf.subtract(prediction,one_hot_labels)))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=one_hot_labels))

# optimizer
lr = tf.Variable(lr,  dtype=tf.float32,  trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,  tf.float32))


def batch_iter(data,  batch_size,  num_epochs,  shuffle=True):
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:", num_batches_per_epoch)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size,  data_size)
            yield shuffled_data[start_index:end_index]


# Initializing the variables
init = tf.global_variables_initializer()
# 定义saver
saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     batches = batch_iter(list(zip(train_x,  train_y)), batch_size,  num_epochs)
#     batches = list(batches)
#
#     for i, batch in enumerate(batches):
#         i = i + 1
#         x_batch,  y_batch = zip(*batch)
#         # sess.run([optimizer],  feed_dict={x: x_batch,  y: y_batch,  dropout: dropout_keep_prob})
#         # print(x_batch, y_batch)
#         _, loss_value, pred = sess.run([optimizer,cross_entropy, prediction], feed_dict={x: x_batch, y: y_batch})
#         # print("loss:{}".format(loss_value))
#         # 测试
#         # if i %   == 0:
#         if i % 50 == 0:
#             # sess.run(tf.assign(lr, lr * (0.90 ** (i // evaluate_every))))
#             # learning_rate = sess.run(lr)
#             tr_acc, _loss = sess.run([accuracy, cross_entropy], feed_dict={x: train_x, y: train_y})
#             ts_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
#             # tr_acc,  _loss = sess.run([accuracy,  cross_entropy],  feed_dict={x: train_x,  y: train_y,  dropout: 1.0})
#             # ts_acc = sess.run(accuracy,  feed_dict={x: test_x,  y: test_y,  dropout: 1.0})
#             print("Iter {}, loss {:.5f}, tr_acc {:.5f}, ts_acc {:.5f}".format(i, _loss, tr_acc, ts_acc))
#
#         # 保存模型
#         if i % checkpoint_every == 0 or i == 6400:
#             path = saver.save(sess, "sounds_models/model", global_step=i)
#             print("Saved model checkpoint to {}\n".format(path))

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batches = batch_iter(list(zip(train_x, train_y)), batch_size, num_epochs)
        batches = list(batches)

        for i, batch in enumerate(batches):
            i = i + 1
            x_batch, y_batch = zip(*batch)
            # sess.run([optimizer],  feed_dict={x: x_batch,  y: y_batch,  dropout: dropout_keep_prob})
            _, loss_value, pred = sess.run([optimizer, cross_entropy, prediction], feed_dict={x: x_batch, y: y_batch})

            if i % 50 == 0:
                # sess.run(tf.assign(lr, lr * (0.90 ** (i // evaluate_every))))
                # learning_rate = sess.run(lr)
                tr_acc, _loss = sess.run([accuracy, cross_entropy], feed_dict={x: train_x, y: train_y})
                # ts_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
                # tr_acc,  _loss = sess.run([accuracy,  cross_entropy],  feed_dict={x: train_x,  y: train_y,  dropout: 1.0})
                # ts_acc = sess.run(accuracy,  feed_dict={x: test_x,  y: test_y,  dropout: 1.0})
                print("Iter {}, loss {:.5f}, tr_acc {:.5f}".format(i, _loss, tr_acc))

            # 保存模型
            if i % checkpoint_every == 0 or i == 6400:
                path = saver.save(sess, "model", global_step= i )
                print("Saved model checkpoint to {}\n".format(path))

def predict():
    with tf.Session() as sess:
        saver.restore(sess, "model")
        output = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        print(output)

if __name__ == "__main__":
    train()
    # predict()