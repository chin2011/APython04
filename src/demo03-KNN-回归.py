#KNN算法_回归
'''
    1.导包.
    2.准备数据集（测试集和训练集）
    3.创建（KNN分类模型）模型对象.
    4.模型训练
    5.模型预测.
'''
# 1.导包.
from sklearn.neighbors import KNeighborsRegressor
# 2.准备数据集（测试集和训练集）
#开根号：   14.53       14.28       1       2.24
#平方和：   211         204         1       5
#差值：    (3,11,9)	(2,10,10)	(0,1,0)	 (1,0,2)    
X_train = [[0,0,1], [1,1,0], [3,10,10], [4,11,12]]
#训练集的标签数据 标签是离散的,所以是一个一维数组
y_train = [0.1,0.2,0.3,0.4]
X_test= [[3,11,10]]

# 3.创建（KNN分类模型）模型对象.
estimator = KNeighborsRegressor(n_neighbors=2)

# 4.模型训练
estimator.fit(X_train, y_train)

# 5.模型预测.
y_pre = estimator.predict(X_test)

#打印预测结果
print("预测结果：", y_pre)