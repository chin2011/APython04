# 利用KNN算法实现手写数字识别

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib   

#扩展：忽略警告.
import warnings
warnings.filterwarnings('ignore')

"""
    1.加载数据集
    2.数据集划分
    3.创建模型
    4.模型训练
    5.模型预测
    6.模型评估
    7.模型保存
"""
'''
import joblib:
    导入了joblib库，用于在Python中实现机器学习模型的持久化保存和加载。
    它允许将训练好的模型保存到磁盘文件中，以便后续可以直接加载使用，而无需重新训练。
    这是机器学习项目中的常用操作。
from collections import Counter:
    导入了Counter类，用于计数可哈希对象。它是一个字典的子类，其中元素作为字典的键，
    它们的计数作为字典的值。Counter类在处理计数问题时非常有用，例如统计列表中每个元素出现的次数。
'''

#1，定义函数，接收用户传入的索引，展示该索引对应的图片。
def show_digit(idx):
    #1。读取数据集，获取到源数据.
    df = pd.read_csv('../data/train.csv')
    # 2。判断传入的索引是否越界。
    if idx >= len(df) or idx < 0:
        print('索引超出范围')
        return
    # 3.获取数据
    X = df.iloc[idx, 1:].values  # 像素值, 784列  特征
    y = df.iloc[idx, 0]  # 标签, 第0列
    print('该图片的标签为：', y) 
    # print(X.shape)  # (784,)
    
    # 4.把（784，）转换成（28，28)
    X = X.reshape(28, 28)
    
    # 5.绘制图像 
    plt.imshow(X, cmap='gray') # 设置颜色为灰度
    plt.axis('off') # 关闭坐标轴
    plt.show()

#2，定义函数，训练模型，并保存训练好的模型。
def train_model():
    # 1.加载数据
    df = pd.read_csv('../data/train.csv')
    # 2.获取数据
    X = df.iloc[:, 1:].values  # 特征 像素值, 1-784列
    y = df.iloc[:, 0].values  # 标签, 第0列
    
    # 对特征列（拆分前）进行归一化，
    X = X /255  # 归一化 (x-0)/(255-0)
    
    # 3.数据集划分
    #  stratify=y: 按照标签Y 的分布进行划分，确保训练集和测试集的标签分布相似
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=42, stratify=y))
    
    # 4.创建模型
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 5.训练模型
    estimator.fit(X_train, y_train)
    
    # 6.模型评估 accuracy_score 
    y_pre = estimator.predict(X_test)
    print('模型预测值为：',y_pre)
    print('模型准确率：', accuracy_score(y_test, y_pre))
  
    # 7.模型保存
    # 参1：模型对象。 参2：模型保存的路径。
    # pickle文件：Python（Pandas）独有的文件类型。
    joblib.dump(estimator, '../model/手写数字识别.pkl')
    print('模型保存成功')

# 3.定义函数，读取模型，并预测。
def predict_model():
    # 1.加载模型
    estimator = joblib.load('../model/手写数字识别.pkl')
    
    # 2.加载数据
    df = pd.read_csv('../data/test.csv')
    
    # 3.获取数据
    X = df.values
    X = X / 255
    
    # 4.预测
    y_pred = estimator.predict(X)
    print('预测值为：', y_pred)


def img_r_pre():
    # 2.1读取图片
    X = plt.imread('../data/test.png')   # 读取图片, 返回的是一个数组
    # plt.imshow(X, cmap='gray')  # 设置颜色为灰度
    # plt.axis('off') # 关闭坐标轴
    # plt.show()

    # 2.2加载模型
    estimator = joblib.load('../model/手写数字识别.pkl')
    
    # print(X.shape)  # (28, 28, 3)
    
    # 2.3图片预处理
    # 如果图片是彩色的，需要转换为灰度图
    if len(X.shape) == 3:
        # 将RGB图像转换为灰度图; 计算数组X在第三个轴（axis=2）上的平均值
        X = X.mean(axis=2)  ## (28, 28, 3) ---> (28, 28)
   
    X = X.reshape(1, -1)  # ((1, 784)  <---(28, 28))
    # X = X / 255     # 归一化 不需要了,因数据都为:[0.5555556  0.51633984....

    # 2.4预测
    y_pred = estimator.predict(X)

    # 2.5打印预测结果
    print('预测值为：', y_pred)

    # # 显示处理后的图像，帮助调试
    # plt.imshow(X.reshape(28, 28), cmap='gray')
    # plt.title(f'Predicted: {y_pred[0]}')
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    # show_digit(23)
    # train_model()
    # predict_model()
    img_r_pre()
    