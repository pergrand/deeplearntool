
from deeplearn.text_classification.dataprocess import readfile, cutword, stopword,savemodel,loadmodel
from deeplearn.text_classification.loss import multiclass_logloss
from deeplearn.text_classification.vectorizers import tf_idf, wcv
from deeplearn.text_classification.models import *

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# 第一步读取数据
file_name = r'中文文本分类语料1.xlsx'
sheet_num = "sheet1"
data = readfile(file_name, sheet_num)
print(data.head())
print(data.分类.unique())

# 第二步分词
name_head = "正文"  # 文本内容的字段名
data = cutword(data, name_head)
print(data.head())

# 第三步 处理标签；将文本标签（Text Label）转化为数字
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.分类.values)

# 第四步 切分数据集
xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, y,
                                                  stratify=y,
                                                  random_state=42,
                                                  test_size=0.1, shuffle=True)
# print(xtrain.shape)
# print(xvalid.shape)

stop_file = '停用词汇总.txt'
stwlist = stopword(stop_file)
# 第五步 TF-IDF 提取特征 / word count/ word vector
xtrain_v, xvalid_v = tf_idf(xtrain, xvalid, 3, 0.5, stwlist)
# Words Count Vectorizer
# xtrain_v, xvalid_v = wcv(xtrain, xvalid, 3, 0.5, stwlist)

#使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_v)
xtrain_svd = svd.transform(xtrain_v)
xvalid_svd = svd.transform(xvalid_v)

#对从SVD获得的数据进行缩放
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

# 对标签进行binarize处理
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)

#创建1个3层的序列神经网络（Sequential Neural Net）
model = Sequential()

model.add(Dense(120, input_dim=120, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(8))
model.add(Activation('softmax'))

# 模型编译
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(xtrain_svd_scl, y=ytrain_enc, batch_size=64,
          epochs=1, verbose=1,
          validation_data=(xvalid_svd_scl, yvalid_enc))