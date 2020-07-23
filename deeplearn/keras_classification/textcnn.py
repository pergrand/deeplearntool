#!/usr/bin/env python
# -*- coding:utf-8
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn import metrics
from keras.layers import Conv1D,Input,Embedding,MaxPooling1D,Flatten,Dropout,Dense,BatchNormalization

from keras.models import Model,Sequential
from keras.backend import concatenate


# def TextCNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
#     main_input = keras.Input(shape=(50,), dtype='float64')
#     # 词嵌入（使用预训练的词向量）
#     embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
#     embed = embedder(main_input)
#     # 词窗大小分别为3,4,5
#     cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
#     cnn1 = MaxPooling1D(pool_size=48)(cnn1)
#     cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
#     cnn2 = MaxPooling1D(pool_size=47)(cnn2)
#     cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
#     cnn3 = MaxPooling1D(pool_size=46)(cnn3)
#     # 合并三个模型的输出向量
#     cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
#     flat = Flatten()(cnn)
#     drop = Dropout(0.2)(flat)
#     main_output = Dense(3, activation='softmax')(drop)
#     model = keras.Model(main_input, main_output)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
#     model.fit(x_train_padded_seqs, one_hot_labels, batch_size=64, epochs=2)
#     # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
#     result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
#     result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
#     y_predict = list(map(str, result_labels))
#     print('准确率', metrics.accuracy_score(y_test, y_predict))
#     print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

#构建CNN分类模型(LeNet-5)
#模型结构：嵌入-卷积池化*2-dropout-BN-全连接-dropout-全连接
def CNN_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50)) #使用Embeeding层将每个词编码转换为词向量
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=2, batch_size=64)
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
    y_predict = list(map(str, y_predict))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))







if __name__=='__main__':
    dataset = pd.read_csv(r'D:\Program Files\qqtxt\1445620939\FileRecv\情感分析数据集/data_train.csv', sep='\t',names=['ID', 'type', 'review', 'label']).astype(str)
    dataset = dataset[:10]
    cw = lambda x: list(jieba.cut(x))
    dataset['words'] = dataset['review'].apply(cw)
    tokenizer=Tokenizer()  #创建一个Tokenizer对象
    #fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    tokenizer.fit_on_texts(dataset['words'])
    vocab=tokenizer.word_index #得到每个词的编号
    x_train, x_test, y_train, y_test = train_test_split(dataset['words'], dataset['label'], test_size=0.1)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids=tokenizer.texts_to_sequences(x_train)

    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    #序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=50) #将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=50)

    # TextCNN_model_1(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test)