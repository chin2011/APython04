'''
    1.导包.
    2.准备数据集（测试集和训练集）
    	1．特征提取.
		2.特征预处理（归一化，标准化）
		3.特征降维.
		4.特征选择.
		5.特征组合.
    3.创建（KNN分类模型）模型对象.
    4.模型训练
    5.模型预测.
'''
# 1.导包.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# # 2.准备数据集（测试集和训练集）
# iris = load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)   
# 
# # 3.创建（KNN分类模型）模型对象.
# estimator = KNeighborsRegressor(n_neighbors=2)
# 
# # 4.模型训练
# estimator.fit(X_train, y_train)
# 
# # 5.模型预测.
# y_pre = estimator.predict(X_test)
# 
# print("预测结果：", y_pre)

# 2.准备数据集（归一化之前的原数据）
X = [[0,0,1], [1,1,0], [3,10,10], [4,11,12]]
y = [0.1,0.2,0.3,0.4]

# 归一化, feature_range=(0,1) 时,可以不写feature_range
# scaler = MinMaxScaler(feature_range=(0,1))
scaler = MinMaxScaler()

# 3.归一化处理特征数据（训练集和测试集） fit_transform(X)
X_scaled = scaler.fit_transform(X)

# 3.创建（KNN分类模型）模型对象.
estimator = KNeighborsRegressor(n_neighbors=2)
# 4.模型训练
estimator.fit(X_scaled, y)
# 5.模型预测.
y_pre = estimator.predict(X_scaled)
print("预测结果：", y_pre)


