
from deeplearn.text_classification.dataprocess import readfile, cutword, stopword
from deeplearn.text_classification.loss import multiclass_logloss
from deeplearn.text_classification.vectorizers import tf_idf

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
# 第五步 TF-IDF 提取特征
xtrain_v, xvalid_v = tf_idf(xtrain, xvalid, 3, 0.5, stwlist)

# 第六步 Logistic Regression模型
# 利用提取的TFIDF特征来fit一个简单的Logistic Regression
clf = LogisticRegression(C=1.0,solver='lbfgs',multi_class='multinomial')
clf.fit(xtrain_v, ytrain)

# 第七步 预测
predictions = clf.predict_proba(xvalid_v)
print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

