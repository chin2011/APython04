"""
线性回归介绍（LinearRegressor）：
    分类：
        一元线性回归： y=wx + b   w==coef_    b==intercept_
        多元线性回归：
    线性回归属于：有监督学习，即：有特征，有标签，且标签是连续的.
    
误差=预测值－真实值      --->小--->好
损失函数（LossFunction，也叫成本函数，代价半数，目标函数，CostFunction）：
    用于描述每个样本点和其预测值之间关系的，让损失函数最小，就是让误差和小，线性回归效率，评估就越高，
如何让损失 最小:
    思路1：正规方程法.
    思路2：梯度下降法.    
--------------------
    1.加载数据集
    2.数据集划分
    3.创建模型
    4.模型训练
    5.模型预测
    6.模型评估
    7.模型保存
"""

# 1导入
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

def dm01_lr():
    # 2 获取数据
    x=[[160],[166],[172],[174],[180]]
    y=[56.3,60.6,65.1,68.5,75]
    x_test=[[176]]

    # 2.2。数据的预处理，这里不需要。
    # 2.3。特征工程（特征提取，特征预处理），这里不需要。
    # 2.4.数据集划分
    
    # 3实例化
    estimator = LinearRegression()
    # 4 训练
    estimator.fit(x, y)
    #打印模型参数 coef_: 斜率w  intercept_: 截距b,
    print('模型参数：', estimator.coef_, estimator.intercept_)
    print('模型参数：', estimator.score(x, y))
    
    # 5模型预测
    y_predict = estimator.predict(x_test)
    print('模型预测值：', y_predict)
    # print('模型预测值：', estimator.predict([[180]]))
   
    # 6保存模型
    # joblib.dump(estimator, '../model/lr.pkl')
   
   
    
    
if __name__ == '__main__':
    dm01_lr()







