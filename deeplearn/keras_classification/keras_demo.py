
# -*- coding:utf-8 -*-

'''
one hot测试
在GTX960上，约100s一轮
经过90轮迭代，训练集准确率为96.60%，测试集准确率为89.21%
Dropout不能用太多，否则信息损失太严重
'''

import numpy as np
import pandas as pd

pos = pd.read_excel(r'D:\Program Files\qqtxt\1445620939\FileRecv\sushen\pos1.xls', header=None)
pos['label'] = 1
neg = pd.read_excel(r'D:\Program Files\qqtxt\1445620939\FileRecv\sushen\neg1.xls', header=None)
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)

# maxlen = 200  # 截断字数
# min_count = 20  # 出现次数少于该值的字扔掉。这是最简单的降维方法
#
# content = ''.join(all_[0])
# abc = pd.Series(list(content)).value_counts()
# abc = abc[abc >= min_count]
# abc[:] = list(range(1, len(abc)+1))
# abc[''] = 0 #添加空字符串用来补全
# word_set = set(abc.index)
#
# def doc2num(s, maxlen):
#     s = [i for i in s if i in word_set]
#     s = s[:maxlen] + ['']*max(0, maxlen-len(s))
#     return list(abc[s])
#
# all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))
#
# #手动打乱数据
# idx = list(range(len(all_)))
# np.random.shuffle(idx)
# all_ = all_.loc[idx]
#
# #按keras的输入要求来生成数据
# x = np.array(list(all_['doc2num']))
# y = np.array(list(all_['label']))
# y = y.reshape((-1,1)) #调整标签形状

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

# #建立模型
# model = Sequential()
# model.add(Embedding(len(abc), 256, input_length=maxlen))
# model.add(LSTM(128))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# batch_size = 128
# train_num = 15000
#
# model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=30)
#
# model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
#
# def predict_one(s): #单个句子的预测函数
#     s = np.array(doc2num(s, maxlen))
#     s = s.reshape((1, s.shape[0]))
#     return model.predict_classes(s, verbose=0)[0][0]

import jieba
all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词
maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法

content = []
for i in all_['words']:
	content.extend(i)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = list(range(1, len(abc)+1))
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)

def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])

all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))

#手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
print(y)
y = y.reshape((-1,1)) #调整标签形状
print(y)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
train_num = 15000

model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=3)

model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)


def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]

# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# from keras.layers.merge import concatenate
# from keras.models import Sequential, Model
# from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
# from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
# from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
# from keras.utils.np_utils import to_categorical
# from keras import initializers
# from keras import backend as K
# from keras.engine.topology import Layer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
#
# file_name = r'中文文本分类语料1.xlsx'
# sheet_num = "sheet1"
# data = pd.read_excel(file_name, sheet_num)
#
#
# lbl_enc = preprocessing.LabelEncoder()
# label = lbl_enc.fit_transform(data.分类.values)
# # 划分训练/测试集
# X_train, X_test, y_train, y_test = train_test_split(data["正文"], label, test_size=0.1, random_state=42)
# # xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, label,
# #                                                   stratify=y,
# #                                                   random_state=42,
# #                                                   test_size=0.1, shuffle=True)
#
# # 对类别变量进行编码，共10类
# y_labels = list(y_train.value_counts().index)
# le = preprocessing.LabelEncoder()
# le.fit(y_labels)
# num_labels = len(y_labels)
# y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
# y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)
#
# # 分词，构建单词-id词典
# tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ")
# tokenizer.fit_on_texts(data["正文"])
# vocab = tokenizer.word_index
#
# # 将每个词用词典中的数值代替
# X_train_word_ids = tokenizer.texts_to_sequences(X_train)
# X_test_word_ids = tokenizer.texts_to_sequences(X_test)
#
# # One-hot
# x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
# x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')
#
# # 序列模式
# x_train = pad_sequences(X_train_word_ids, maxlen=20)
# x_test = pad_sequences(X_test_word_ids, maxlen=20)