
import pandas as pd
import jieba

def readfile(file_name,sheet_num):
    data = pd.read_excel(file_name, sheet_num)
    # print(data.head())
    # # data.info()
    # print(data.分类.unique())
    # data = data[:10]
    return data

def cutword(data,name_head):
    # jieba.enable_parallel() #并行分词开启 windows不支持
    data['文本分词'] = data[name_head].apply(lambda i: jieba.cut(i))
    data['文本分词'] = [' '.join(i) for i in data['文本分词']]
    return data

def stopword(filename):
    stwlist = [line.strip() for line in open('停用词汇总.txt', 'r', encoding='utf-8').readlines()]
    return stwlist
"""
多线程分词
"""
# def cut(sentence):
# 	if sentence!=None:
# 		sentence = jieba.lcut(sentence,cut_all=False)
# 		return [i for i in sentence]
# 	else :
# 		return None
# if __name__ == '__main__':
# 	pool = Pool(cpu_count())
# 	data = pool.map(cut, data['正文'])
# 	pool.close()
# 	pool.join()