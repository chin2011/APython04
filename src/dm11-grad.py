'''
波士顿房价预测正规方程模型
'''

#正规方程模型
'''
波士顿房价预测正规方程模型
'''

# 正规方程模型
from sklearn.linear_model import LinearRegression
import numpy as np
# 修改为推荐的数据加载方式
import pandas as pd


# 函数：正规方程模型
def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


# 函数：数据处理
def data_process():
    # 使用推荐的方式加载波士顿房价数据
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    x = data
    y = target
    x = np.array(x)
    y = np.array(y)
    x = x[y < 50.0]
    y = y[y < 50.0]
    x = x[:, 5]
    x = x.reshape(-1, 1)
    return x, y


# 主函数
if __name__ == '__main__':
    # 处理,训练,预测
    x, y = data_process()
    model = linear_regression(x, y)
    print(model.score(x, y))
    print(model.coef_)
    print(model.intercept_)
    print(model.predict([[5]]))
