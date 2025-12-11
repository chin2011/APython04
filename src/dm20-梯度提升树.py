'''
梯度提升树 －案例泰坦尼克号生存预测

'''
#梯度提升树

# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# 1.定义函数，演示：梯度提升树分类器的使用  泰坦尼克号生存预测
def titanic_survival1():
    # 1.数据处理
    data = pd.read_csv('../data/titanic.csv')
    # data = data[['Survived', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']]
    data = data[['Survived', 'Sex', 'Age']]
    data.dropna(inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    X = data.drop(['Survived'], axis=1).copy()
    y = data['Survived']
    
    # 2. 数据集划分
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=22)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print('训练集的形状：', X_train.shape)
    print('测试集的形状：', X_test.shape)
    
    # 3.模型训练
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=22)
    model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test)
    
    # 4.模型评估
    print('准确率：', accuracy_score(y_test, y_pred))
    print('召回率：', recall_score(y_test, y_pred))
    print('精确率：', precision_score(y_test, y_pred))
    print('F1分数：', f1_score(y_test, y_pred))


# 2.定义函数，演示：梯度提升树分类器  泰坦尼克号生存预测
def titanic_survival2():
    # 1.数据处理
    data = pd.read_csv('../data/titanic.csv')
    data = data[['Survived', 'Sex', 'Age']].copy()
    data.dropna(inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    X = data.drop(['Survived'], axis=1).copy()
    y = data['Survived']

    # 2. 数据集划分
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3.模型训练,预测,评估
    # 场景1：单个决策树
    estimator = DecisionTreeClassifier()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    # print(f'单个决策树-预测结果：{y_pred}')
    # print(f'单个决策树-分类评估报告：\n{classification_report(y_test, y_pred)}')
    print('单个决策树-准确率：', accuracy_score(y_test, y_pred))

    # 场景2：梯度提升树对象（GBDT）
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=22)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('梯度提升树-准确率：', accuracy_score(y_test, y_pred))
    print('-'*50)
    

    # 4.GBDT 网格搜索交叉验证训练
    param_grid = {
        'n_estimators': [60,70,90,110]
        , 'max_depth': [2,4,6,8,10]
        # , 'learning_rate': [0.1,0.2,0.3,0.4,0.5]
    }
    #参数1：模型,参数2：参数空间,参数3：交叉验证折数cv=2
    model2 = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=2)
    # 模型训练
    model2.fit(X_train, y_train)
    # 模型评估
    print('网格搜索后的-准确率：', model2.score(X_test, y_test))
    print('网格搜索后的-最佳参数：', model2.best_params_)
   
    
if __name__ == '__main__':
    # titanic_survival1()
    titanic_survival2()