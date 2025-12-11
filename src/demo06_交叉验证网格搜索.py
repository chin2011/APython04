# 导包.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier #KNN算法分类对象
from sklearn.preprocessing import StandardScaler #数据标准化的
from sklearn.datasets import load_iris  #加载鸢尾花测试集的.
# 导入数据集的分割方法 ,寻找最优超参的（网格搜索十交叉验证）。
from sklearn.model_selection import train_test_split, GridSearchCV
#模型评估的，计算模型预测的准确率
from sklearn.metrics import accuracy_score


# 交叉验证网格搜索函数
def dm06_cross_validation_grid_search():
    # 1.准备数据集（测试集和训练集）
    iris = load_iris()
    # 2.数据预处理：从150个特征和标签中，按照8：2的比例，切分训练集和测试集。
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                            test_size=0.2, random_state=22)
    # 3.数据预处理：数据标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4.模型训练：创建KNN分类器对象，并进行训练. 不使用超参进行训练 在下面写
    estimator = KNeighborsClassifier()
    
    #4.2 使用校验验证风格搜索
    # param_grid = [{'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}]
    param_grid = [{'n_neighbors': [i for i in range(1, 11)]}] # 1到10
    
    # 输出一个estimator, 出来一个新的estimator（功能变的强大）
    # param_grid: 参数网格，是一个列表，列表的元素是一个字典，字典的键是参数名称，值是参数的值
    # cv：指定几折交叉验证, 默认是3折，这里指定4折
    estimator = GridSearchCV(estimator, param_grid, cv=4)

    # 4.3 模型训练
    estimator.fit(X_train, y_train) #4个模型每个模型进行网格搜素找到做好的模型
    
    # 4.4 查看最佳参数
    print("最佳结果为:", estimator.best_score_)  #0.9666666666666668
    print("最佳参数为:", estimator.best_params_) # {'n_neighbors': 3}
    print("最佳估计器为:", estimator.best_estimator_) #
    # print("交叉验证结果为:", estimator.cv_results_)
    
    # #4.5 保存交叉验证结果为:
    # (pd.DataFrame(estimator.cv_results_)
    #  .to_csv('../data/dm06-cvgs.csv', index=False))
    
    
    
    # 5.模型评估
    # {'n_neighbors': 3} 模型评估：0.9666666666666668
    # #方法一：使用关键字参数解包（推荐）
    # # 5.1 获取最优超参的模型对象。
    # # estimator = KNeighborsClassifier(n_neighbors=3)
    # estimator = KNeighborsClassifier(**estimator.best_params_)
    # # 5.2 模型训练
    # estimator.fit(X_train, y_train)
    
    #方法二：直接使用最佳估计器
    estimator = estimator.best_estimator_ 
    
    # 5.3 模型预测
    y_pre = estimator.predict(X_test)
    # 5.4 模型评估
    print("模型评估acs**：", accuracy_score(y_test, y_pre)) # 0.9666666666666668
    print("模型评估ess：", estimator.score(X_test, y_test)) # 0.9666666666666668
    

    
    
if __name__ == '__main__':
     dm06_cross_validation_grid_search()