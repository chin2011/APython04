'''
线性回归算法属于有监督学习之 有特征，有标签，且标签是连续的.
    线性回归分类：
        一元线性回归：1个特征列，1个标签列.
        多元线性回归：多个特征列，1个标签列.
线性回归大白话解释：
	它是用线性公式来描述特征和标签之间关系的，方便做预测，公式如下：
	    一元线性回归：y=W*×+b
	    多元线性回归:y=w1*x1+w2*x2+w3*x3+...+wn*xn+b=w的转置*x+b
	    
思路：
	预测值和真实值之间的误差，误差越小，模型越好。
	
具体的方案：
    1.最小二乘.每个（样本）误差平方和
    2.均方误差（MSE）每个（样本）误差平方和／样本总数
    3.均方根误差（RMSE）每个（样本）误差平方和／样本总数的平方根
    4.平均绝对误差（MAE）每个（样本）误差绝对值和／样本总数

如何让损失函数最小？
    思路1：梯度下降法.
    思路2：正规方程法.

机器学习开发流程：
    1.记载数据.
    2.数据的预处理.
    3.特征工程（特征提取，特征预处理...）
    4.模型训练.
    5.模型预测.
    6.模型评估

'''

# 使用推荐的方式加载波士顿房价数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
# data_csv="../data/BostonHousing.csv"
    
# 导包
from sklearn.linear_model import LinearRegression, SGDRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler


def R_Regression():
    # 1.加载数据 
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # 数据处理
    # 特征
    # hstack: 水平拼接, 竖直拼接 vstack, 对角拼接 diag
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # 标签
    target = raw_df.values[1::2, 2].reshape(-1, 1)

    # 数据集切分
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    # 数据预处理
    # 标准化处理
    scaler = StandardScaler()
    # 训练数据标准化
    x_train = scaler.fit_transform(x_train)
    # 测试数据标准化
    x_test = scaler.transform(x_test)

    # 创建模型  fit_intercept: 是否计算截距，默认为True
    estimator = LinearRegression(fit_intercept=True)
    # 模型训练
    estimator.fit(x_train, y_train)
    # 打印模型计算出的 w（权重，weight） 和 b（偏置，bias）。
    print("权重w:", estimator.coef_)
    print("截距b:", estimator.intercept_)

    # 模型预测
    y_pre = estimator.predict(x_test)
    # print("预测值：", y_pre)
    # print("真实值：", y_test)

    # 模型评估 MAE,mse,rmse,R2...
    print("模型评估：")
    print("平均绝对误差MAE:", mean_absolute_error(y_test, y_pre))  #
    print("均方误差MSE:", mean_squared_error(y_test, y_pre))
    # RMSE, 均方根误差，RMSE越小，模型越好
    print("均方根误差RMSE:", root_mean_squared_error(y_test, y_pre))
    # # R2, 决定系数，越接近1，模型越好
    # print("决定系数R2:", r2_score(y_test, y_pre))
    # # 最小二乘误差, 越小，模型越好
    # print("最小二乘误差:", np.sum((y_pre - y_test) ** 2)) 


def SGD_Regressor():
    # 1.加载数据 
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # 数据处理
    # 特征
    # hstack: 水平拼接, 竖直拼接 vstack, 对角拼接 diag
    # [::2, :]- 取偶数行的所有列.  ]   [1::2, :2]- 取奇数行的前两列
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    # 标签  .reshape(-1, 1) - 把x轴（特征）转成多行 1列的形式。
    # reshape(-1, 1): -1表示行数未知，根据列数自动计算行数.
    # reshape(-1, 1)  处理前:[1,2,3]  ----> [[1]],[2],[3]]
    target = target.reshape(-1, 1)

    # 数据集切分
    x_train, x_test, y_train, y_test = (
        train_test_split(data, target, test_size=0.2, random_state=23))

    # 数据预处理
    # 标准化处理
    scaler = StandardScaler()
    # 训练数据标准化
    x_train = scaler.fit_transform(x_train)
    # 测试数据标准化
    x_test = scaler.transform(x_test)

    # 创建模型  fit_intercept: 是否计算截距，默认为True, 
    # alpha: 正则化参数，默认值为0.0001, 
    # max_iter: 最大迭代次数，默认值为1000, 
    # eta0: 初始学习率，默认值为0.01
    estimator = SGDRegressor(fit_intercept=True, max_iter=1000, eta0=0.01)
    # 模型训练
    estimator.fit(x_train, y_train)
    # 打印模型计算出的 w（权重，weight） 和 b（偏置，bias）。
    print("权重w:", estimator.coef_)
    print("截距b:", estimator.intercept_)

    # 模型预测
    y_pre = estimator.predict(x_test)
    # print("预测值：", y_pre)
    # print("真实值：", y_test)

    # 模型评估 MAE,mse,rmse,R2...
    print("模型评估：")
    print("平均绝对误差MAE:", mean_absolute_error(y_test, y_pre))  #
    print("均方误差MSE:", mean_squared_error(y_test, y_pre))
    # RMSE, 均方根误差，RMSE越小，模型越好
    print("均方根误差RMSE:", root_mean_squared_error(y_test, y_pre))
    
    
    
if __name__ == '__main__':
    # 波士顿房价预测 正规方程法
    R_Regression()
    # 波士顿房价预测 梯度下降法
    # SGD_Regressor()

