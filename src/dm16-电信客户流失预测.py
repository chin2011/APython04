'''
案例－电信客户流失预测
    已知：用户个人，通话，上网等信息数据
需求：通过分析特征属性确定用户流失的原因，以及哪些因素可能导致用户流失。
    建立预测模型来判断用户是否流失，并提出用户流失预警策略。

'''

CSV_FILE = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
# 导入包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def customer_dm():
    # 1.获取数据  WA_Fn-UseC_-Telco-Customer-Churn.csv
    data = pd.read_csv(CSV_FILE)
    print(data.head())

    # 2.数据处理
    # 正确地分离特征和标签
    X = data.drop(['customerID', 'Churn'], axis=1)
    y = data['Churn']

    # 对分类变量进行独热编码
    X = pd.get_dummies(X)

    # 处理缺失值
    X = X.fillna(0)

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 将标签转换为数值
    y = y.map({'Yes': 1, 'No': 0})

    # 3.数据集切割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 4.机器学习
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    # 5.模型评估
    print('准确率:', precision_score(y_test, y_predict))
    print('召回率:', recall_score(y_test, y_predict))
    print('F1值:', f1_score(y_test, y_predict))

#1.定义函数，演示：数据的预处理.
def dm01_data_preprocess():
    # 1.获取数据
    data = pd.read_csv(CSV_FILE)
    # data.info()
    # print(data.head())

    # 2.数据处理
    # 2.1 因为Churn 和gender列是字符串，所以需要进行one-hot编码（热编码处理）。
    data=pd.get_dummies(data,columns=['Churn','gender'])
    # data.info()
    
    # 2.2 删除one-hot处理后，余的列。
    # 删除customerID列，因为customerID列没有意义，没有特征属性。
    data.drop(['Churn_No', 'gender_Male','customerID'], axis=1, inplace=True)
    # data.info()
    
    # 2.3 修改列名，将Churn_Yes → flag，充当标签列.
    # flag False →不流失，True → 流失
    data.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # print(data.head())
    
    # 2.4 查看数据值的分布。False:5174  True:1869  -->不均衡数据
    print(data['flag'].value_counts())
    


# 2.定义函数，演示：特征筛选
def dm02_feature_select():
    # 1.获取数据
    data = pd.read_csv(CSV_FILE)
    
    #2.处理类别型的数据类别型数据做one-hot编码
    data = pd.get_dummies(data,columns=['Churn','gender'])
    
    # 3.去除列churn_no gender_Male #nplace=True 在原来的数据上进行删除
    data.drop(['Churn_No', 'gender_Male','customerID'], axis=1, inplace=True)
    
    # 4.列标签重命名 打印列名
    data.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    print(data.columns)
    # 5.查看标签的分布情况0.26用户流失
    # value_counts = data['flag'].value_counts()
    
    # 6.查看Contract_Month 是否月签约流失情况
    # 参1：数据集,参2：特征属性，参数3：标签列 默认为False:不流失，True:流失
    sns.countplot(data=data, x='Contract', hue='flag' )
    plt.show()


# 3.定义函数，演示：模型训练与评测
def dm03_model_train_eval():
    # 1.获取数据
    data = pd.read_csv(CSV_FILE)

    # 保存原始数据用于可视化
    original_data = data.copy()

    # 2.数据处理
    # 删除不需要的列
    data.drop(['customerID'], axis=1, inplace=True)

    # 对所有分类变量进行独热编码，包括目标变量Churn
    data = pd.get_dummies(data)

    # 处理缺失值
    data = data.fillna(0)

    # 分离特征和标签
    # 注意：经过one-hot编码后，Churn列变成了Churn_Yes和Churn_No两列
    # 我们只需要其中一列作为目标变量，这里选择Churn_Yes
    X = data.drop(['Churn_Yes', 'Churn_No'], axis=1)
    y = data['Churn_Yes']  # 1表示流失(Yes)，0表示未流失(No)

    # 保存特征名称
    feature_names = X.columns

    # 3.数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 将标准化后的numpy数组转回DataFrame，保持列名
    X_dataframe = pd.DataFrame(X_scaled, columns=feature_names)

    # 4.数据集切割 (使用numpy数组进行训练)
    X_train, X_test, y_train, y_test = (
        train_test_split(X_scaled, y, test_size=0.2, random_state=22))

    # 5.机器学习
    model = LogisticRegression(max_iter=100, solver='liblinear')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    # 6.模型评估
    print('准确率:', precision_score(y_test, y_predict))
    print('召回率:', recall_score(y_test, y_predict))
    print('F1值:', f1_score(y_test, y_predict))

    # 7.可视化 - 使用原始数据而非标准化后的数据
    # 注意：在one-hot编码后，原始的Contract列已不存在，应使用原始数据进行可视化
    # 例如，我们可以绘制原始Contract列与Churn的关系
    sns.countplot(data=original_data, x='Contract', hue='Churn')
    plt.show()


if __name__ == '__main__':
    # customer_dm()
    # dm01_data_preprocess()
    # dm02_feature_select()
    dm03_model_train_eval()