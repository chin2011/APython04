'''
Kmeans简介：
	它属于无监督学习，即：有特征，无标签，根据样本间的相似性进行划分.
	所谓的相似性可以理解为就是距离，例如：欧式距离，曼哈顿（城市街区）距离，切比雪夫距离，闵式距离.··

    一般大厂，项目初期在没有先备知识（标签）的情况下，可能会用.

    随机创建不同二维数据集作为训练集，并结合k-means算法将其聚类，尝试分别聚类不同数量的簇，
并观察聚类效果：
'''

# 导入包
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
#默认会按照高斯分布（正态分布）生成数据集,
# 参数1：样本数量，参数2：特征数量，参数3：簇数量，参数4：标准差，参数5：随机数种子
from sklearn.datasets import make_blobs
#评价指标，值越大，聚类效果越好。
from sklearn.metrics import calinski_harabasz_score

plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 1.定义函数，演示：使用KMeans模型数据探索聚类
def kmeans_demo1():
    # 1.1 创建数据集
    x = np.random.randn(200, 2)
    x1 = np.random.randn(50, 2) + np.array([2, 2])
    x2 = np.random.randn(50, 2) + np.array([-2, -2])
    
    # 1.2 拼接数据集
    x = np.vstack((x, x1, x2))
    
    # 2. 创建模型对象
    model = KMeans(n_clusters=3)
    # 3. 训练模型
    model.fit(x)
    y_predict = model.predict(x)
    
    # 4. 模型评估
    print('类别标签：', y_predict)
    print('类别中心：', model.cluster_centers_)
    print('聚类个数：', model.n_clusters)
    
    # 5. 绘制图像
    plt.figure(figsize=(8, 5))
    plt.scatter(x[:, 0], x[:, 1], c=y_predict, s=50, cmap='viridis')
    plt.scatter(model.cluster_centers_[:, 0], 
                model.cluster_centers_[:, 1], s=200, c='red', marker='x')
    
    plt.show()


# 2.定义函数，演示:使用KMeans模型数据探索聚类
def kmeans_demo2():
    # 1. 创建数据集1000个样本，每个样本2个特征4个质心族数据标准差[0.4，0.2,0.2，0.2]
    # make_blobs: 创建数据集, 返回数据集和标签
    # 参数1：样本数量，参数2：特征数量，参数3：簇数量，参数4：标准差，参数5：随机数种子
    x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]], 
                      cluster_std=[0.4,0.2,0.2,0.2], random_state=9)
    plt.figure(figsize=(8, 5))
    # 绘制图像, 参数1：特征1，参数2：特征2，参数3：颜色，参数4：大小，参数5：颜色映射
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    
    # 2. 使用k-means进行聚类，并使用cH方法评估
    # 创建模型对象, 参数1：簇数量，参数2：随机数种子
    y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(x)
    # 绘制图像, 参数1：特征1，参数2：特征2，参数3：大小
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()

    # 3. 评估 cH方法, 参数1：数据集，参数2：预测标签  值域越大，聚类效果越好
    print('Calinski-Harabasz指数：', calinski_harabasz_score(x, y_pred))

    
if __name__ == '__main__':
    # kmeans_demo1()
    kmeans_demo2()