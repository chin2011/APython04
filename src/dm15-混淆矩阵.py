'''
逻辑回归
    概述：
        属于有监督学习，即：有特征，有标签，且标签是离散的.适用于二分类.
    评估：
        精确率，召回率，F1值

混淆矩阵：
    概述：
        用来描述真实值和预测值之间关系的.

    结论：
        1.模拟使用 分类少的 充当 正例.
        2.精确率=真正例在预测正例中的占比，即：tp/（tp+fp）
        3.召回率=真正例在真正例中的占比，即：tp/（tp + fn）
        4.F1值= 2*（精确率 * 召回率）／（精确率 + 召回率）
'''
import numpy as np
import pandas as pd
# 导包
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler  # 添加StandardScaler用于数据标准化

def fun_from_load():
    # 1.获取数据
    cancer = load_breast_cancer()

    # 2.基本数据处理
    X = cancer.data
    y = cancer.target

    # 数据标准化处理，有助于提高模型收敛速度和性能
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3.数据集切割
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=22))

    # 4.机器学习
    # 增加max_iter参数以解决收敛警告，并指定solver为lbfgs
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5.模型评估
    y_predict = model.predict(X_test)

    print(confusion_matrix(y_test, y_predict))
    print(f'精确率：{precision_score(y_test, y_predict)}')
    print(f'召回率：{recall_score(y_test, y_predict)}')
    print(f'F1值：{f1_score(y_test, y_predict)}')


#demo1
def fun_demo1():
    # 1.定义变量，记录：样本数据
    y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
    # 2.定义变量，记录：模型A的预测结果
    y_pred_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
    # 3.定义变量，记录：模型B的预测结果
    y_pred_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '恶性', '恶性', '恶性']

    # 4.用标签标记正例，反例.
    # labels = ['恶性', '良性']
    df_labels=['恶性(正例)', '良性(反例)']
    
    
    # 5.计算混淆矩阵
    # cm_A = confusion_matrix(y_train, y_pred_A, labels=labels)
    cm_A = confusion_matrix(y_train, y_pred_A)
    # print(f'混淆矩阵：\n{cm_A}')
    
    # 6.为了测试结果更好看，把上述的混淆矩阵转换成DataFrame。
    cm_A = pd.DataFrame(cm_A, index=df_labels, columns=df_labels)
    print(f'混淆矩阵A：\n{cm_A}')

    # 争对 y_pred_B
    cm_B = confusion_matrix(y_train, y_pred_B)
    # print(f'混淆矩阵：\n{cm_B}')
    cm_B = pd.DataFrame(cm_B, index=df_labels, columns=df_labels)
    print(f'混淆矩阵B：\n{cm_B}')

    # 7.计算A模型准确率，召回率，F1值 (指定pos_label为'恶性')
    # 参1：真实值，参2：预测值，参3：正例的标签
    print(f'A模型准确率：{precision_score(y_train, y_pred_A, pos_label="恶性")}')
    print(f'A模型召回率：{recall_score(y_train, y_pred_A, pos_label="恶性")}')
    print(f'A模型F1值：{f1_score(y_train, y_pred_A, pos_label="恶性")}')
    
    print('-'*30)
    # 8.计算B模型准确率，召回率，F1值
    print(f'B模型准确率：{precision_score(y_train, y_pred_B, pos_label="恶性")}')
    print(f'B模型召回率：{recall_score(y_train, y_pred_B, pos_label="恶性")}')
    print(f'B模型F1值：{f1_score(y_train, y_pred_B, pos_label="恶性")}')
    




if __name__ == '__main__':
    # fun_from_load()
    fun_demo1()
