'''
商品评论情感分析

1获取数据
2数据基本处理
	2-1处理数据y
	2-2加载 停用词------无效词
	2-3 处理数据x  把文档分词
	2-4统计词频矩阵作为句子特征
3准备训练集测试集
4模型训练
	4-1实例化贝叶斯添加拉普拉斯平滑参数
	4-2模型预测
 5 模型评估

朴素贝叶斯介绍：
    概述：
        贝叶斯：仅仅依赖概率就可以进行分类的一种机器学习算法.
        朴素：不考虑特征之间的关联性，即：特征间都是相互独立的.
        原始：P（AB)=P（A)*P（B|A)=P（B)*P（A/B)
        加入朴素后：P（AB）=P（A）*P（B)
    细节：
        因为我们分词要用到jieba分词器，记得先装一下，例如：pip install jieba
'''

# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba        #分词包
from sklearn.model_selection import train_test_split
#词频统计包，把评论内容转成词频矩阵.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB   #朴素贝叶斯对象

CSV_File = '../data/Book_Reviews.csv'
STOP_WORDS_FILE = '../data/cn_stopwords.txt'

# 1.定义函数，演示：商品评论情感分析
def bayes_demo():
    # 1 获取数据
    data = pd.read_csv(CSV_File)
    # print(data.head())
    
    # 2 数据基本处理
    # 评价: 好评:1,差评:0
    data['评价'] = data['评价'].map(lambda x: 1 if x == '好评' else 0)
    # 2.1 处理数据y
    y = data['评价']
    
    # 2.2 加载 停用词------无效词
    stopwords =[]
    with open(STOP_WORDS_FILE,'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.strip())
    # 去重, set:不能有重复的元素,--->再转为列表
    stopwords = list(set(stopwords))
    
    # 2.3 处理数据x  把文档分词
    comments = []
    for comment in data['内容']:
        # 分词,去停用词 lcut: 精确分词
        words = jieba.lcut(comment) 
        words = [word for word in words if word not in stopwords]
        # 拼接, 转为字符串 
        comments.append(','.join(words))
    
    # 2.4 统计词频矩阵作为句子特征
    transfer = CountVectorizer(stop_words=stopwords)
    # 2.4 统计词频矩阵，先训练，后转换，在转数组。
    x = transfer.fit_transform(comments)
    # 2.5 看一下我们13条评论，切词，且删除停用词后，一共剩下多少个词了，
    # print(transfer.get_feature_names_out())
    
    
    # 3 准备训练集测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    # 4 模型训练
    # 4.1 实例化贝叶斯添加拉普莱斯平滑参数
    model = MultinomialNB(alpha=1.0)
    # 4.2 模型预测
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print('预测结果为：', y_predict)
    # print('实际结果为：\n', y_test)
    
    # 5 模型评估
    myscore = model.score(x_test, y_test)
    print('准确率：', myscore)
    
    
if __name__ == '__main__':
    bayes_demo()
