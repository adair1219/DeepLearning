# encoding:utf-8

import numpy as np

wordsList = np.load(r'D:\NLP\learning-nlp\chapter-8\sentiment-analysis\wordsList.npy')
print('载入word列表')
wordsList = wordsList.tolist()
print(wordsList[0: 20])
wordsList = [word.decode('UTF-8')
             for word in wordsList]
wordVectors = np.load(r'D:\NLP\learning-nlp\chapter-8\sentiment-analysis\wordVectors.npy')
VectorList = wordVectors.tolist()[2]
print('单个词向量：%s' % VectorList) 
print('载入文本向量')

print(len(wordsList))
print(wordVectors.shape)

import os
from os.path import isfile, join

pos_file = r'D:\NLP\learning-nlp\chapter-8\sentiment-analysis\pos/'
neg_file = r'D:\NLP\learning-nlp\chapter-8\sentiment-analysis\neg/'
pos_files = [pos_file + f for f in os.listdir(
    pos_file) if isfile(join(pos_file, f))]
neg_files = [neg_file + f for f in os.listdir(
    neg_file) if isfile(join(neg_file, f))]
num_words = []
for pf in pos_files:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完结')

for nf in neg_files:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完结')

num_files = len(num_words)
print('文件总数', num_files)
print('所有的词的数量', sum(num_words))
print('平均文件词的长度', sum(num_words) / len(num_words))

import re

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 300  # Dimensions for each word vector


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


max_seq_num = 250
"""
ids = np.zeros((num_files, max_seq_num), dtype='int32')
file_count = 0
for pf in pos_files:
  with open(pf, "r", encoding='utf-8') as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
      try:
        ids[file_count][indexCounter] = wordsList.index(word)
      except ValueError:
        ids[file_count][indexCounter] = 399999  # 未知的词
      indexCounter = indexCounter + 1
      if indexCounter >= max_seq_num:
        break
    file_count = file_count + 1

for nf in neg_files:
  with open(nf, "r",encoding='utf-8') as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
      try:
        ids[file_count][indexCounter] = wordsList.index(word)
      except ValueError:
        ids[file_count][indexCounter] = 399999  # 未知的词语
      indexCounter = indexCounter + 1
      if indexCounter >= max_seq_num:
        break
    file_count = file_count + 1

np.save('idsMatrix', ids)
"""

from random import randint

batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 100
lr = 0.001
ids = np.load(r'D:\NLP\learning-nlp\chapter-8\sentiment-analysis\idsMatrix.npy')


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]


    return arr, labels

print(get_train_batch())


def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


import tensorflow as tf   

graph = tf.Graph()

with graph.as_default():
  labels = tf.placeholder(tf.float32, [batch_size, num_labels])
  input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])

data = tf.Variable(
    tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)  # embedding_lookup 根据 ids 返回 tensor


lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)  # 防止过赌拟合
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)  # value 就是 output, _ 是 state, return embedding
print(value)

value = tf.transpose(value, [1, 0, 2])  # 将输出 tensor with embedding 转化为 1 or 0
print(value)

#  获得最终的 labels


weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))  # shape=[64, 2]  weight = std - mean (两个均为正太分布) 若分数小于则重新生成
last = tf.gather(value, int(value.get_shape()[0]) - 1)
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
prediction = (tf.matmul(last, weight) + bias)  # matmul 将两个 embeding 相乘, return embeding (24 x 50) logits
print('prediction %s' % prediction)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))  # 1 表示 按行查找 [1, 5, 26, 49, ......, 23]
# accuracy 应该精确为 float32 类型，示例 0.78, correct_pred 为 int 类型
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 将整型转化为浮点型, 及[1.0, 5.0, 26.0, 49.0......., 23.0], 后面为指定参数

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)  # lr 学习率

saver = tf.train.Saver()

with tf.Session() as sess:
    # 初始化参数
    # 保证程序不会重新跑一遍，如果存在 checkpoint 就从最新检查点开始训练，如果没有，就初始化 tf.init
    if os.path.exists("models") and os.path.exists("models/checkpoint"):
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    else:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

    # 开始训练模型
    iterations = 10000
    for step in range(iterations):
        next_batch, next_batch_labels = get_test_batch()  # test, 那是因为预训练已经提前弄好了，这会是在测试
        # 目的是为检测准确度，使用 feed_dict 传入 input_data, labels, 两参数会被分配到相关函数中
        if step % 20 == 0:
            # feed_dict 给 placeholder 赋值，赋值内容该到哪去就哪去,    这里每次提取 64 X 300 X 50
            feed_dict = {
              input_data: next_batch,
              labels: next_batch_labels,
            }
            print("step:", step, " 正确率:", (sess.run(
                accuracy, feed_dict=feed_dict)) * 100)

    # 保存模型
    if not os.path.exists("models"):
        os.mkdir("models")
    save_path = saver.save(sess, "models/model.ckpt")
    print("Model saved in path: %s" % save_path)
