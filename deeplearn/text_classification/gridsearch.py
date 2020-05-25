
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
# 第五步 TF-IDF 提取特征
xtrain_v, xvalid_v = tf_idf(xtrain, xvalid, 3, 0.5, stwlist)
# Words Count Vectorizer
# xtrain_v, xvalid_v = wcv(xtrain, xvalid, 3, 0.5, stwlist)
# 网格搜索是一种超参数优化的技巧。可以通过获取最优的参数组合来产生良好的文本分类效果。
mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
# SVD初始化
svd = TruncatedSVD()

# Standard Scaler初始化
scl = preprocessing.StandardScaler()

# 再一次使用Logistic Regression
lr_model = LogisticRegression()

# 创建pipeline
clf = pipeline.Pipeline([('svd', svd),
                         ('scl', scl),
                         ('lr', lr_model)])
# 参数网格（A Grid of Parameters）
param_grid = {'svd__n_components' : [120, 180],
              'lr__C': [0.1, 1.0, 10],
              'lr__penalty': ['l1', 'l2']}
# 网格搜索模型（Grid Search Model）初始化
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

#fit网格搜索模型
model.fit(xtrain_v, ytrain)  #为了减少计算量，这里我们仅使用xtrain
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# 朴素贝叶斯进行超参数调优
# nb_model = MultinomialNB()
#
# # 创建pipeline
# clf = pipeline.Pipeline([('nb', nb_model)])
#
# # 搜索参数设置
# param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
#
# # 网格搜索模型（Grid Search Model）初始化
# model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
#                                  verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
#
# # fit网格搜索模型
# model.fit(xtrain_v, ytrain)  # 为了减少计算量，这里我们仅使用xtrain
# print("Best score: %0.3f" % model.best_score_)
# print("Best parameters set:")
# best_parameters = model.best_estimator_.get_params()
# for param_name in sorted(param_grid.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

# # 第六步 Logistic Regression模型
# # 利用提取的TFIDF特征来fit一个简单的Logistic Regression
# print('LR Train classifier...')
# model = LogisticRegression(C=1.0,solver='lbfgs',multi_class='multinomial')
# # model = xgboost()
#
# model.fit(xtrain_v, ytrain)
#
# # 保存模型
# model_path = 'clf.pickle'
# savemodel(model_path, model)
#




# # 第七步 预测
# clf2 = loadmodel(model_path)
# predictions = clf2.predict_proba(xvalid_v)
# print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

