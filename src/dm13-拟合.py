import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,Ridge


# 1．定义函数，模拟：欠拟合.
def under_fit():
    np.random.seed(666)
    x=np.random.uniform(-3.0,3.0,size=100)
    y=2*x+3+np.random.randn(100)*10
    
    #实例化线性回归模型
    model = LinearRegression()
    # 训练
    model.fit(x.reshape(-1,1), y)
    # 预测
    y_predict = model.predict(x.reshape(-1,1))
    
    plt.scatter(x, y)   # 绘制数据点
    plt.plot(x, y_predict, color='r')   # 绘制预测曲线
    plt.show()

# 2．定义函数，模拟：拟合.
def just_fit():
    #1.生成数据
    np.random.seed(666)
    x=np.random.uniform(-3.0,3.0,size=100)
    # 生成数据,y=0.5x^2+x+2 + 噪声 normal: 正态分布
    # normal: 参1：平均值，参2：标准差，参3：生成个数
    y=0.5*x**2+x+2 + np.random.normal(0,1,size=100)
   
   #2.实例化线性回归模型
    estimator = LinearRegression()
    
    #3.训练模型
    # 3.1。数据预处理，把x轴（特征）转成多行1列的形式。
    X=x.reshape(-1,1)
    # 3.2。数据预处理，把x轴（特征）转成多行2列的形式。
    # 3.2因为目前特征列只有1列，模型过于简单，会出现欠拟合的问题，我们增加1列特征列，
    # 从而增加模型的复杂度。
    # #即: [[1]，[2],[3]，[4]，[5]] = [[1，1]，[2，4]，[3，9]，[4，16]，[5，25]]
    
    X2=np.hstack([X,X**2]) # 添加一个特征,hstack: 水平拼接,行数不变,列数增加
    '''
    这段代码创建了一个机器学习管道(Pipeline),包含两个步骤:
使用PolynomialFeatures将输入特征转换为2次多项式特征
使用LinearRegression进行线性回归拟合 这样可以实现多项式回归,提高模型对非线性数据的拟合能力。
    '''
    # estimator = Pipeline([
    #     ('poly', PolynomialFeatures(degree=2)),
    #     ('linear', LinearRegression())
    # ])
    
    estimator = LinearRegression()
    estimator.fit(X2, y)
    
    #4.预测
    y_predict = estimator.predict(X2)
    
    #5.计算误差
    myret = mean_squared_error(y, y_predict)
    print('myret-->',myret)
    
    #绘制曲线
    plt.scatter(x, y)   # 绘制数据点
    # 绘制预测曲线, np.argsort(x): 返回排序后x的索引, np.sort(x): 对x排序
    # 例如：排序前x轴是[11,33,22]  对应的索引是[0,1,2]
    # 排序后x轴是[11,22,33]       对应的索引是[0,2,1]
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')   # 绘制预测曲线
    plt.show()
    
    
# 3．定义函数，模拟：过拟合.
def over_fit():
    # 1.生成数据
    np.random.seed(666)
    x = np.random.uniform(-3.0, 3.0, size=100)
    # 生成数据,y=0.5x^2+x+2 + 噪声 normal: 正态分布
    # normal: 参1：平均值，参2：标准差，参3：生成个数
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2.实例化线性回归模型
    estimator = LinearRegression()
    
    # 3.训练模型
    X = x.reshape(-1, 1)
    # 数据增加高次项,
    X2 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    
    estimator.fit(X2, y)
    
    # 4.预测
    y_predict = estimator.predict(X2)
    
    # 5.计算误差
    myret = mean_squared_error(y, y_predict)
    print('myret-->',myret)
    
    # 绘制曲线
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show()
    
    
# 4.定义函数，Lasso回归
def lasso_regression():
     # 1.生成数据
     np.random.seed(666)
     x = np.random.uniform(-3.0, 3.0, size=100)
     y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

      # 2实例化L1正则化模型做实验：alpha惩罚力度越来越大 → k值越来越小，返回会欠拟合
     estimator = Lasso(alpha=0.1)
     
     # 3.训练模型
     X = x.reshape(-1, 1)
     X2 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
     estimator.fit(X2, y)
     
     # 4.预测
     y_predict = estimator.predict(X2)
     
     # 5.计算误差
     myret = mean_squared_error(y, y_predict)
     print('myret-->',myret)
     
     # 绘制曲线
     plt.scatter(x, y)
     plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
     plt.show()
 
# 5.Ridge回归
def ridge_regression():
     # 1.生成数据
     np.random.seed(666)
     x = np.random.uniform(-3.0, 3.0, size=100)
     y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
     
     # 2实例化L2正则化模型做实验：alpha惩罚力度越来越大 → k值越来越小，返回会欠拟合
     estimator = Ridge(alpha=0.1)
     
     # 3.训练模型
     X = x.reshape(-1, 1)
     X2 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
     estimator.fit(X2, y)
     
     # 4.预测
     y_predict = estimator.predict(X2)
     # 5.计算误差
     myret = mean_squared_error(y, y_predict)
     print('myret-->',myret)
     
     # 绘制曲线
     plt.scatter(x, y)
     plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
     plt.show()
     
     
if __name__ == '__main__':
    # under_fit()
    # just_fit()
    # over_fit()
    # lasso_regression()
    ridge_regression()