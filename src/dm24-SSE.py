'''
演示聚类算法的评估指标，即：SSE+肘部法，SC轮廓系数法，CH轮廓系数法.

思路1：SSE+肘部法
	SSE:
		概述：
			所有簇的所有样本到该簇质心的误差的平方和.
		特点：
			随着K值的增加，SSE值会逐渐减少
		目标：
			SSE值越小，代表簇内样本越聚集，内聚程度越高.
	肘部法：
		K值增大，SSE值会随之减小，下降梯度陡然变缓的得时候，那个K值，就是我们要的最佳值。

'''

# 导入包
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.定义函数，演示：SSE
def demo_SSE1():
    # 1 创建数据
    x = np.random.randn(200, 2)
    x1 = np.random.randn(50, 2) + np.array([2, 2])
    x2 = np.random.randn(50, 2) + np.array([-2, -2])
    x = np.vstack((x, x1, x2))      # 拼接数据集,3个簇
    
    # 2 创建模型对象
    kmeans = KMeans(n_clusters=3)
    # 3 训练模型
    kmeans.fit(x)
    # 4 模型评估
    y_predict = kmeans.predict(x)
    # print('类别标签：', y_predict)
    print('类别中心：\n', kmeans.cluster_centers_)
    # 绘制图像
    plt.figure(figsize=(8, 5))
    plt.scatter(x[:, 0], x[:, 1], c=y_predict, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x')
    plt.show()
    

# 2.定义函数，演示：SSE+肘部法
def demo_SSE2():
    # 1 创建数据
    x,y = make_blobs(
        n_samples=1000
        , n_features=2
        , centers=[[-1,-1],[0,0],[1,1],[2,2]]   # 4个质心
        ,cluster_std=[0.4,0.2,0.2,0.2]
        , random_state=9)
    
    # 2 创建模型对象
    sse_list = []
    for i in range(1, 100):
        # 创建模型对象
        # 参数1：簇数量，参数2：最大迭代次数，参数3：随机数种子
        kmeans = KMeans(n_clusters=i,max_iter=100, random_state=9)
        # 训练模型
        kmeans.fit(x)
        # 模型评估 kmeans.inertia_ : 所有簇的所有样本到该簇质心的误差的平方和.
        sse_list.append(kmeans.inertia_)
    # print(sse_list)
    
    # 3.绘制图像 横坐标：K值，纵坐标：SSE
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 100), sse_list, marker='o')
    plt.xlabel('K值')
    plt.ylabel('SSE')
    plt.title('SSE肘部法')
    # 刻度
    plt.xticks(range(0, 100, 3))
    plt.yticks(range(0, 5000, 500))
    plt.grid(True)
    plt.show()


# 3.定义函数，演示：SC轮廓系数法
def demo_SC():
    # 1 创建数据
    x,y = make_blobs(
        n_samples=1000
        , n_features=2
        , centers=[[-1,-1],[0,0],[1,1],[2,2]]   # 4个质心
        ,cluster_std=[0.4,0.2,0.2,0.2]
        , random_state=9)
    # 2 创建模型对象
    sc_list = []
    # 获取到每个k值，计算其对应的 sc值，并添加到 sc_list列表中
    for i in range(2, 100):     # K值>=2,至少2个簇 
        # 创建模型对象
        kmeans = KMeans(n_clusters=i,max_iter=100, random_state=9)
        # 训练模型
        kmeans.fit(x)
        # 模型评估
        y_predict = kmeans.predict(x)
        # sc 轮廓系数
        sc = metrics.silhouette_score(x, y_predict)
        sc_list.append(sc)
        if sc < 0.1:
            break
            
    # 3.绘制图像 横坐标：K值，纵坐标：SC
    plt.figure(figsize=(8, 5))
    # range(2, 100): K值范围，sc_list: SC值列表
    plt.plot(range(2, 100), sc_list, marker='o')
    plt.xlabel('K值')
    plt.ylabel('SC')
    plt.title('SC轮廓系数法')
    # 刻度
    plt.xticks(range(0, 100, 3))
    #：range(0, 1, 0.1)。  --->np.arange: 创建一个等差数列,支持浮点数步长
    # 错误原因是 range() 函数不接受浮点数作为步长参数，它只接受整数。
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.show()


# 4。定义函数，演示：CH轮廓系数法   越大越好
def demo_CH():
    # 1 创建数据
    x,y = make_blobs(
        n_samples=1000
        , n_features=2
        , centers=[[-1,-1],[0,0],[1,1],[2,2]]   # 4个质心
        ,cluster_std=[0.4,0.2,0.2,0.2]
        , random_state=9
    )
    # 2 创建模型对象
    ch_list = []
    # 获取到每个k值，计算其对应的 ch值，并添加到 ch_list列表中
    for i in range(2, 100):     # K值>=2,至少2个簇
        # 创建模型对象
        kmeans = KMeans(n_clusters=i,max_iter=100, random_state=9)
        # 训练模型
        kmeans.fit(x)
        # 模型评估
        y_predict = kmeans.predict(x)
        ch = metrics.calinski_harabasz_score(x, y_predict)
        ch_list.append(ch)
        # print('K值：', i, 'CH值：', ch)
        if ch < 0.1:
            break

    # 3.绘制图像 横坐标：K值，纵坐标：CH
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 100), ch_list, marker='o')
    plt.xlabel('K值')
    plt.ylabel('CH')
    plt.title('CH轮廓系数法')
    # 刻度
    plt.xticks(range(0, 100, 3))
    plt.grid(True)
    plt.show()


# 5.测试








if __name__ == '__main__':
    # demo_SSE1()
    # demo_SSE2()
    # demo_SC()
    demo_CH()