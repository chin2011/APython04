'''
演示逻辑回归模型实现癌症预测.

逻辑回归模型介绍：
    概述：
        属于有监督学习，即：有特征，有标签，且表示是离散的.主要适用于：二分类.
    原理：
        把线性回归处理后的预测值→通过Sigmoid激活函数，映射到[0，1]概率→基于自定义的阈值，结合概率来分类.
    损失函数：
        极大似然估计函数的负数形式.


'''

#导包
import numpy as np
from sklearn.model_selection import train_test_split #训练集和测试集分割
from sklearn.linear_model import LogisticRegression #逻辑回归模型
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer # 


# 1.获取数据
cancer = load_breast_cancer()

# 2.基本数据处理
X = cancer.data
y = cancer.target
# 切割训练集和测试集
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=22))

# 2.1 缺失值处理
# X_train = np.nan_to_num(X_train)
# X_test = np.nan_to_num(X_test)
# print(X_train.shape)

# # 2.2确定特征值，目标值
# X_train = X_train[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
# X_test = X_test[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

# 2.3分割数据

# 3.特征工程(标准化）
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 4.机器学习（逻辑回归）
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
# print(f'预测结果：{y_predict}')

# 5.模型评估
print('预测前评估 的准确率：\t', model.score(X_test, y_test))
print('逻辑回归的准确率:    \t', accuracy_score(y_test, y_predict))
#测试集的标签，预测值.
print('逻辑回归的精确率：', precision_score(y_test, y_predict))


