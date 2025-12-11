'''
1.导包.
2.准备数据集（测试集和训练集）
    1．特征提取.
    2.特征预处理（标准化）
    3.特征降维.
    4.特征选择.
    5.特征组合.
3.创建（KNN分类模型）模型对象.
4.模型训练
5.模型预测.
6.模型评估.
'''
# 导包.
from sklearn.neighbors import KNeighborsClassifier #KNN算法分类对象
from sklearn.preprocessing import StandardScaler #数据标准化的
from sklearn.datasets import load_iris  #加载鸢尾花测试集的.
# 导入数据集的分割方法 
from sklearn.model_selection import train_test_split   
#模型评估的，计算模型预测的准确率
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def dm01_loadiris():
    # 1.准备数据集（测试集和训练集）
    iris = load_iris()  # 加载数据集, 
        # {'data':array(...),'target':array(...),'frame':array(...)...}
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 
    # 'feature_names', 'filename', 'data_module'])
    # 3.查看数据集所有的键，
    # print(iris.keys())
    
    #4.查看数据集的键对应的值，
    # print(iris.data[:5])  # 查看数据集的特征数据
    # print(f'具体的标签:{iris.target[:5]}')  # 查看数据集的标签数据
    # print(iris.target_names)  # 查看数据集的标签名称
    # print(f'具体的特征:{iris.feature_names}')  # 查看数据集的特征名称
    # print(iris.DESCR)  # 查看数据集的描述信息
    
    
    # X = iris.data  # 特征数据
    # y = iris.target  # 标签数据
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # 2.标准化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # 
    # # 3.创建（KNN分类模型）模型对象.
    # estimator = KNeighborsClassifier(n_neighbors=2)
    # # 4.模型训练
    # estimator.fit(X_train, y_train)
    # 
    # # 5.模型预测.
    # y_pre = estimator.predict(X_test)
    # 
    # print("测试集标签：\t", y_test)
    # print("预测结果： \t", y_pre)
    # 
    # # 6.模型评估.
    # mse = mean_squared_error(y_test, y_pre)
    # print("模型评估：", mse)
    # print("模型评估：", accuracy_score(y_test, y_pre))

#2.定义函数，绘制数据集的散点图.
def demo02_show_iris():
    # 1.准备数据集（测试集和训练集）
    iris = load_iris()  # 加载数据集, 
    
    #2.把数据集转换成DataFrame,设置data, columns, 目标值名称
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # print(df.head(3))

    col1='sepal length (cm)'
    col2='sepal width (cm)'

    # 3.绘制数据集, 设置x,y,数据集,颜色,是否画出回归线
    sns.lmplot(data=df,x=col1, y=col2,  hue='Species', fit_reg=True)
    # plt.xlabel(col1)
    # plt.ylabel(col2)
    plt.title('Iris Dataset', fontsize=16)   # 标题
    plt.tight_layout()   # 自动调整子图参数，使之填充整个图像区域
    plt.show()
    
#3．定义函数，切分训练集和测试集。
def dm03_traintest_split():
    # 1.准备数据集（测试集和训练集）
    iris = load_iris()  # 加载数据集, 

    # 2.数据的预处理：从150个特征和标签中，按照8：2的比例，切分训练集和测试集。
    #random_state 是随机数种子，保证每次运行结果相同
    # 参1：特征数据. 参2：标签数据. 参3：测试集的比例。
    # 返回值：训练集的特征数据，测试集的特征数据，训练集的标签数据，测试集的标签数据。
    X_train, X_test, y_train, y_test = train_test_split(iris.data, 
                                    iris.target, test_size=0.2, random_state=42)
    # 3.打印训切割的数据集
    print(f'训练集的标签数据：{X_train}')  # 打印训练集的标签数据
    print(f'训练集的标签数据：{y_train}')  # 打印训练集的标签数据
    print(f'训练集的样本数量：{X_train.shape[0]}')  # 打印训练集的样本数量
    print(f'测试集的样本数量：{X_test.shape[0]}')  # 打印测试集的样本数量

# 4．定义函数，实现鸢尾花完整案例一加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测。
def dm04_iris_all():
    # 1.准备数据集（测试集和训练集）
    iris = load_iris()
    # 2.数据预处理：从150个特征和标签中，按照8：2的比例，切分训练集和测试集。
    X_train, X_test, y_train, y_test = train_test_split(iris.data, 
                                    iris.target, test_size=0.2, random_state=22)
    # 3.数据预处理：数据标准化处理
    # 3.1.创建StandardScaler对象
    scaler = StandardScaler()
    # 3.2.数据标准化处理   
    # fit_transform：兼具fit和transform的功能，即：训练，转换。
    #       该函数适用于：第一次进行标准化的时候使用. 一般用于处理：训练集。
    # transform：转换，即：使用已经训练的模型进行数据转换。 一般用于处理：测试集。

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4.1.创建（KNN分类模型）模型对象.
    estimator = KNeighborsClassifier(n_neighbors=3)
    
    # 4.2.模型训练
    estimator.fit(X_train, y_train)

    # 5.模型预测.
    # 场景1：对刚才切分的测试集（30条）进行测试。
    # 5.1直接预测即可，获取到：预测结果
    y_pre = estimator.predict(X_test)
    # print("预测结果： \t", y_pre)

    # 场景2：对新的数据集（源数据150条之外的数据）进行测试.
    # 5.1.准备新的数据集
    # X_new = [[7.8,2.1,3.9,1.6], [5.9, 3.0, 5.1, 1.8]]
    X_new = [[7.8,2.1,3.9,1.6]]
    # 5.2.对新的数据集进行数据标准化处理，使用的是训练集的StandardScaler对象
    X_new = scaler.transform(X_new)
    
    # 5.3.对新的数据集进行预测，获取到：预测结果
    y_pre_new = estimator.predict(X_new)
    print("新的数据集预测结果：", y_pre_new)
    
    
    # 5.4查看上述数据集，每种分类的预测概率。
    y_pre_proba = estimator.predict_proba(X_new)
    print("新的数据集预测概率：", y_pre_proba)
    # [0.0,.66666667,0.33333333] 0:概率为0.0, 1:概率为0.66666667, 2:概率为0.33333333
    # 故预测结果为：1


    # 6.模型评估.
    # 方式1：直接评分，基于：训练集的特征 和 训练集的标签
    ess = estimator.score(X_train, y_train)
    print("模型评估1：", ess)
    
    # 方式2：基于测试集的标签 和 预测结果进行评分。
    acs= accuracy_score(y_test, y_pre)
    print("模型评估2：", acs)
    
    
    
if __name__ == '__main__':
    # dm01_loadiris()
    # dm02_show_iris()
    # dm03_traintest_split()
    dm04_iris_all()
