'''
随机森林:
    泰坦尼克号案例
'''

# 导入包
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import  ConfusionMatrixDisplay


CSV_FILE = '../data/titanic.csv'

# 1.定义函数，演示：随机森林, 泰坦尼克号乘客生存预测
def t_survived():
    # 1.数据处理
    data = pd.read_csv(CSV_FILE)
    #obj对象:name,ticket,cabin,embarked, sex 列处理
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
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 5.模型预测
    y_predict = model.predict(X_test)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()
    
    # 6.模型评估
    print('准确率:', accuracy_score(y_test, y_predict))
    print('精确率:', precision_score(y_test, y_predict))
    print('召回率:', recall_score(y_test, y_predict))
    print('F1-score:', f1_score(y_test, y_predict))
    

# 2.定义函数，自己选择属性. 演示：随机森林, 泰坦尼克号乘客生存预测
def t_feature():
    # 1.数据处理
    data = pd.read_csv(CSV_FILE)
    X=data[['Pclass', 'Sex', 'Age']].copy()
    y=data['Survived'].copy()
    
    # 转换类别值 Sex列: male:0, female:1
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    # 处理Age列的缺失值，用平均年龄填充
    X['Age'].fillna(X['Age'].mean(), inplace=True)
    
    # 2.数据集划分
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22)
    
    # 3.数据预处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4.模型训练 RandomForestClassifier: 随机森林, n_estimators: 树的数量
    # max_depth: 树的深度, 默认为None,意思是：绘制的决策树结构，最多10层.
    # random_state: 随机数种子，保证每次运行结果一致
    # 演示：单一的决策树效果。
    model1 = RandomForestClassifier(max_depth=6, random_state=22)
    model1.fit(X_train, y_train)
    
    # 5.模型预测,评估
    y_predict = model1.predict(X_test)
    print('单一的决策树-准确率:', accuracy_score(y_test, y_predict))
    
    
    # 6.随机森林 交叉验证网格搜索 进行模型训练和评估
    # 演示：多个的决策树（Bagging思想）效果。
    param_grid = {'n_estimators': [60,70,90,110], 'max_depth': [2,4,6,8,10]}
    # GridSearchCV: 交叉验证网格搜索, 参数: 模型对象, 参数空间, 交叉验证折数cv=2
    model2 = GridSearchCV(RandomForestClassifier(), param_grid, cv=2)
    model2.fit(X_train, y_train)
    
    # 模型预测,评估
    y_predict2 = model2.predict(X_test)
    
    print('多个的决策树-准确率:', accuracy_score(y_test, y_predict2))
    print('多个的决策树-最佳参数:', model2.best_params_)
    
    
if __name__ == '__main__':
    # t_survived()
    # print('-'*30)
    t_feature()