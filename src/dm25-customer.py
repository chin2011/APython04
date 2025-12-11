'''
案例: 顾客数据聚类分析法
已知：
	客户性别、年龄、年收入、消费指数
需求：
	对客户进行分析，找到业务突破口，寻找黄金客户





'''
CSV_FILE = '../data/Mall_Customers.csv'
# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.datasets import make_blobs

# 处理中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# 1.定义函数，演示：
def demo_SC1():
    # 1 导入 数据
    data = pd.read_csv(CSV_FILE)
    x = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    # print(x.info())
    
    # 2 数据预处理
    x = StandardScaler().fit_transform(x)
    # print(x)
    
    # 3 创建模型对象
    kmeans = KMeans(n_clusters=5)
    # 4 训练模型
    kmeans.fit(x)
    # 5 模型评估
    y_predict = kmeans.predict(x)
    # print('类别标签：', y_predict)
    print('类别中心：\n', kmeans.cluster_centers_)
    
    # 绘制图像
    plt.figure(figsize=(8, 5))
    # 绘制数据, x[:, 0]: x[:, 1] 横坐标，纵坐标
    plt.scatter(x[:, 0], x[:, 1], c=y_predict, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x')
    plt.show()
    
    
# 2.定义函数，演示：使用聚类算法完成客户案例分析,求最佳K值
# 年收入、消费指数 的聚类分析  -->求最佳K值
def fn_find_k():
    # 1 导入 数据
    data = pd.read_csv(CSV_FILE)
    
    # 1.2 抽取数据:年收入、消费指数
    x = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    # 1.3 数据预处理
    x = StandardScaler().fit_transform(x)
    
    # 2 定义 sse_list，sc_list，记录：不同k值的 评估效果.
    sse_list = []   #sse：只考虑 簇内，越小越好。
    sc_list = []   #sc：考虑 簇内和簇间，越大越好。

    # 3.定义for训练，测试不同k值的 评估效果。
    for i in range(2, 20):
        # 创建模型对象 max_iter: 迭代次数，random_state：随机数种子
        kmeans = KMeans(n_clusters=i, max_iter=100, random_state=9)
        # 训练模型
        kmeans.fit(x)
        # 模型评估
        # 1. sse
        sse = kmeans.inertia_
        sse_list.append(sse)
        # 2. sc
        y_predict = kmeans.predict(x)
        sc = silhouette_score(x, y_predict)
        sc_list.append(sc)
    # 打印最大 轮廓系数的K值
    print('最大-轮廓系数的K值为：', np.argmax(sc_list) + 2)
    
    # 4.绘制图像
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, 20), sse_list, marker='o', label='SSE')
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 20), sc_list, marker='o', label='SC')
    plt.show()


# 3.定义函数，演示：模型训练，模型预测，模型评估
def fn_model_predict_evaluate():
    # 1 导入 数据
    data = pd.read_csv(CSV_FILE)

    # 1.2 抽取数据:年收入、消费指数
    x = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    # 1.3 数据预处理
    x = StandardScaler().fit_transform(x)
    
    # 2 创建模型对象 max_iter: 迭代次数，K=5 是刚才找到的K值
    kmeans = KMeans(n_clusters=5, max_iter=100, random_state=9)
    # 3 训练模型
    kmeans.fit(x)
    # 4 模型预测
    y_predict = kmeans.predict(x)
    # 5 模型评估
    # print('类别标签：', y_predict)
    # print('类别中心：\n', kmeans.cluster_centers_)
    print('轮廓系数：', silhouette_score(x, y_predict))
    
    # 6.绘制图像
    plt.figure(figsize=(8, 5))
    '''
    x[:, 0]:年收入,Annual Income (k$), 
    x[:, 1]: 消费指数,Spending Score (1-100)
    c=y_predict 根据聚类预测结果为每个点着色
    s=50 设置点的大小
    cmap='viridis' 使用viridis颜色映射方案
    '''
    # plt.scatter(x[:, 0], x[:, 1], c=y_predict, s=50, cmap='viridis')
    # 绘制5个簇的样本点-→散点图 (0,1) (1,1) (2,1) (3,1) (4,1)
    # y_predict == 0: 获取y_predict == 0的样本点，
    # x[y_predict == 0, 0]: 获取这些样本点的x坐标
    plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], label='Standard')
    plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], label='Traditional')
    plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], label='Normal')
    plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], label='Youth')
    plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], label='TA')
    
    # 绘制类别中心 5个 质心的(x,y)
    plt.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1], s=150, label='类别中心')
    #标签 
    plt.xlabel('年收入',fontsize=14)
    plt.ylabel('消费指数',fontsize=14)
    plt.title('K-Means聚类分析',fontsize=20)
    plt.legend(loc='best')  # 图例，位置自动选择
    plt.show()




if __name__ == '__main__':
    # demo_SC1()
    # fn_find_k()
    fn_model_predict_evaluate()
