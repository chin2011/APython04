'''
泰坦尼克号乘客生存预测

        字段:
    Passengerld：乘客编号
    Survived：生存状态（0代表未存活，1代表存活）
    Pclass：舱位等级（1等、2等、3等）
    Name：乘客姓名
    Sex:性别
    Age:年龄
    SibSp：在船上的兄弟姐妹或配偶个数
    Parch：在船上的父母或孩子个数''
    Ticket：船票号码
    Fare：票价
    Cabin：客舱
    Embarked:登船港口 (C=Cherbourg,Q=Queenstown,S=Southampton)
    
'''
import numpy as np
from matplotlib import pyplot as plt
# 导包
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression

CSV_FILE = '../data/titanic.csv'

# .定义函数，演示：泰坦尼克号乘客生存预测
# 逻辑回归
def t_survived():
    # 1.数据处理
    data = pd.read_csv(CSV_FILE)
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})

    # 处理Age列的缺失值，用平均年龄填充
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    X = data.drop(['Survived'], axis=1)
    y = data['Survived']

    # 2.数据集划分
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22)

    # 3.数据预处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4.模型训练
    model = LogisticRegression() # 逻辑回归
    model.fit(X_train, y_train)

    # 5.模型预测
    y_predict = model.predict(X_test)

    # 6.模型评估
    print('准确率:', accuracy_score(y_test, y_predict))
    print('精确率:', precision_score(y_test, y_predict))
    print('召回率:', recall_score(y_test, y_predict))
    print('F1-score:', f1_score(y_test, y_predict))


# 2.定义函数，演示：决策树 
def t_tree(): # 决策树
    # 1.数据处理
    data = pd.read_csv(CSV_FILE)
    # data.info()     # 查看数据结构，是否有缺失值
    
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})

    # 处理Age列的缺失值，用平均年龄填充
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    X = data.drop(['Survived'], axis=1)
    y = data['Survived']

    # 2.数据集划分
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22)
    
   # 3.数据预处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 4.模型训练
    model = DecisionTreeClassifier() # 决策树
    model.fit(X_train, y_train)
    
    # 5.模型预测
    y_predict = model.predict(X_test)
    
    # 6.模型评估
    print('准确率:', accuracy_score(y_test, y_predict))
    print('精确率:', precision_score(y_test, y_predict))
    print('召回率:', recall_score(y_test, y_predict))
    print('F1-score:', f1_score(y_test, y_predict))
    
    
# 3.定义函数，演示：提取特征和标签
def t_feature(): # 提取特征和标签
    # 1.数据处理
    data = pd.read_csv(CSV_FILE)
    X = data[['Pclass', 'Sex', 'Age']].copy()  # 使用.copy()创建副本避免警告
    y = data['Survived']
    # X.loc[:, 'Sex'] = X['Sex'].map({'male': 0, 'female': 1})

    # 转换类别值 Sex列: male:0, female:1
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    # 处理Age列的缺失值，用平均年龄填充
    X['Age'].fillna(X['Age'].mean(), inplace=True)
    # X.info()
    
    # 2.数据集划分
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22)
    # 3.数据预处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 4.模型训练
    # max_depth: 树的深度, 默认为None,意思是：绘制的决策树结构，最多10层.
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X_train, y_train)
    # 5.模型预测
    y_predict = model.predict(X_test)

    # 6.模型评估
    print('准确率:', accuracy_score(y_test, y_predict))
    print('精确率:', precision_score(y_test, y_predict))
    print('召回率:', recall_score(y_test, y_predict))
    print('F1-score:', f1_score(y_test, y_predict))
    
    # #分类评估报告
    # print(classification_report(y_test, y_predict))
    
    # 7.可视化
    plt.figure(figsize=(30, 20))
    # 参1: 模型对象，参2: 填充颜色，参3: 特征名称
    plot_tree(model, filled=True, feature_names=X.columns)
    #保存 图
    plt.savefig('../data/titanic_tree.png')
    #显示
    plt.show()
    
    
# 4.定义函数，用自定义数据演示：
def t_custom_data():
    # 1.数据处理
    x_train = np.array(list(range(1, 11))).reshape(-1, 1)
    y_train = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
    # print(x_train)
    # print(y_train)
    
    # 2.模型训练
    model1 = LinearRegression() # 线性回归
    model2= DecisionTreeRegressor(max_depth=1) # 决策树
    model3= DecisionTreeRegressor(max_depth=3) # 决策树
    
    # 3.模型训练
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)
    
    # 4.模型预测
    # 4.1 准备测试数据
    # x_test = np.array(list(range(0.0, 10.0, 0.1))).reshape(-1, 1)
    # arange: 生成等差数列，reshape: 改变数组形状，生成测试数据
    x_test = np.arange(0.0, 10.0, 0.1).reshape(-1, 1) 
    # print(x_test)
    
    # 4.2 模型预测
    y_predict1 = model1.predict(x_test)
    y_predict2 = model2.predict(x_test)
    y_predict3 = model3.predict(x_test)

    # 5.模型评估
    print('线性回归准确率:', model1.score(x_test, y_predict1))
    print('决策树1准确率:', model2.score(x_test, y_predict2))
    print('决策树3准确率:', model3.score(x_test, y_predict3))
    
    # 6.可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='gray', label='Training Data')
    plt.plot(x_test, y_predict1, label='Linear Regression',c='red')
    plt.plot(x_test, y_predict2, label='Decision Tree 1',color='blue')
    plt.plot(x_test, y_predict3, label='Decision Tree 3',color='green')
    plt.legend()
    # 6.2 设备X軕标题, Y轴标题
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Decision Tree Regression') # 设备标题
    plt.show()


    
if __name__ == '__main__':
    # t_survived()
    # t_tree()
    # print('-'*30)
    # t_feature()
    t_custom_data()
