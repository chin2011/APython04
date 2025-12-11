'''
示例：用 scikit-learn 对鸢尾花种类进行分类
'''

# 1. 导入所需库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. 加载内置数据集（鸢尾花）
iris = datasets.load_iris()
X = iris.data   # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target # 标签：0=Setosa, 1=Versicolor, 2=Virginica

# 3. 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 创建模型（这里用随机森林）
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 在测试集上预测
y_pred = model.predict(X_test)

# 7. 评估准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")