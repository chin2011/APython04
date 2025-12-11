'''
XGBoost算法API 例子
回顾：XGBoost极限梯度提升树
    ExtremeGradientBoostingTree，底层采用打分函数决定是否分支.
原理：
    Gain值=分枝前的打分 -（分支后左子树打分+分支后右子树打分）
    如果Gain值>0，考虑分枝，否则：不考虑分枝.

'''
from sklearn.utils import compute_sample_weight

CSV_File = '../data/winequality-red.csv'
CSV_TRAIN = '../data/winequality-red_train.csv'
CSV_TEST = '../data/winequality-red_test.csv'
PKL_File = '../data/winequality-red.pkl'
JSON_File = '../data/winequality-red.json'

# 导入XGBoost算法包
import xgboost as xgb   # 极限梯度提升树对象
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import joblib       # 保存和加载模型
from collections import Counter

# 1.定义函数，演示：XGBoost算法API
def XGBoost_dm1():
    # 1. 数据集
    iris = load_iris()

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = \
        train_test_split(iris.data, iris.target, test_size=0.2)
    # 3. 创建数据集并指定数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 3,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'eta': 0.1,
        'nthread': 4,
        'eval_metric': 'auc'
    }
    num_round = 10

    # 4. 训练模型
    # 参数1：模型参数，参数2：训练数据集，参数3：迭代次数num_boost_round，即树的棵数
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    # 5. 预测
    preds = model.predict(dtest)
    # 6. 评估
    # print(preds)
    print('准确率1：', accuracy_score(y_test, preds))


# 2.定义函数，演示：xgb案例：红酒品质分类
def XGBoost_dm2():
    # 1. 数据处理
    # 原文件 ../data/wine.data 不存在，使用现有的 winequality-red.csv 文件
    data = pd.read_csv(CSV_File, delimiter=';')


    # 将质量作为标签列
    X = data.drop(['quality'], axis=1).copy()
    y = data['quality'].copy()
    
    # 确保标签值从0开始，符合XGBoost的要求（标签必须在[0, num_class)范围内）
    # 调整标签值使其从0开始
    y = y - y.min()

    # 2. 数据集划分
    #  stratify=y：按标签y进行划分，以确保训练集和测试集中标签的分布相似
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)

    # 3. 创建数据集并指定数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

    # 根据新数据集调整参数
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',  # 改为 softprob 以适应新的多类别分类
        'num_class': len(np.unique(y)),  # 自动确定类别数
        'gamma': 0.1,
        'max_depth': 6,  # 增加深度以适应更复杂的数据
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'eta': 0.1,
        'nthread': 4,
        'eval_metric': 'mlogloss'  # 更改评估指标
    }

    # 4. 训练模型
    # 参数1：模型参数，参数2：训练数据集，参数3：迭代次数
    # 注意：迭代次数num_boost_round需要根据数据集和模型参数进行调参
    model = xgb.train(params, dtrain, num_boost_round=10)  # 增加迭代次数以获得更好的性能
    # 5. 预测
    preds = model.predict(dtest)
    
    # 6. 评估
    # print(preds)
    print('准确率2：', accuracy_score(y_test, np.argmax(preds, axis=1)))
   
    
# 3.定义函数，对红酒品质分类源数据
def XGBoost_dm3():
    # 1. 加载数据集
    data = pd.read_csv(CSV_File, delimiter=';')
    # data.info()
    
    # 2. 数据处理: 将质量作为标签列
    X = data.drop(['quality'], axis=1).copy()
    y = data['quality'].copy()
    
    # 2.2. 调整标签值使其从0开始
    y = y - y.min()
    
    # 3. 查看数据集的标签分布
    # Counter(y)  # 统计每个标签的出现次数
    # print('查看标签结果的分布情况公均衡：', Counter(y).most_common())  
    
    # 4. 数据集划分
    # stratify=y：按标签y进行划分，以确保训练集和测试集中标签的分布相似
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)
    
    # 5. 把上述的训练集特征和标签数据拼接到一起，测试集特征和标签数据拼接到一起。 最后写到文件中。
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(CSV_TRAIN, index=False)
    
    pd.concat([X_test, y_test], axis=1).to_csv(CSV_TEST, index=False)
    
    
# 4.定义函数， 训练模型并保存 pkl---json
def XGBoost_dm4():
    # 1. 加载数据集
    train_data = pd.read_csv(CSV_TRAIN)
    test_data = pd.read_csv(CSV_TEST)
    
    # 2. 数据处理
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    # 交叉验证时采用分层抽取
    
    # # 3. 创建数据集并指定数据格式
    # '''
    # 。xgb.DMatrix是XGBoost专用的数据结构，用于高效存储和处理数据。参数说明：
    #     X_train/X_test: 特征数据
    #     label=y_train/y_test: 对应的标签数据
    #     missing=np.nan: 指定缺失值用NaN表示
    # 这样转换后的数据对象可以被XGBoost算法直接使用。
    # '''
    # dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    # dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)
    # params = {
    #     'booster': 'gbtree',    # 使用树模型
    #     'objective': 'multi:softprob',  # 多类别分类目标函数
    #     'num_class': len(np.unique(y_train)),   # 自动确定类别数
    #     'gamma': 0.1,  # 改变默认值以避免过拟合,默认0.3，增加模型的非线性能力,
    #     'max_depth': 6,  # 添加深度以适应更复杂数据集，默认:6
    #     'lambda': 2,    # 添加正则化项以防止过拟合,默认1
    #     'subsample': 0.7,  # 控制每棵树使用的训练数据比例，避免过拟合,默认1
    #     'colsample_bytree': 0.7,  # 控制每棵树使用的特征比例，避免过拟合,默认1
    #     'min_child_weight': 3,  # 控制叶子节点的最小权重和，避免过拟合,默认1
    #     'eta': 0.1,  # 学习率，控制每棵树的贡献，避免过拟合,默认0.3
    #     'nthread': 4,  # 使用4个线程进行训练,默认使用CPU核数
    #     'eval_metric': 'mlogloss'   # 评估指标,默认为rmse
    # }
    # num_round = 10
    # 
    # # 加入平衡权重，因为数据集是样本不均衡的。class_weight.compute_sample_weight
    # # 计算样本权重, 参数1：权重类型，参数2：标签列
    # class_weight = compute_sample_weight('balanced', y_train)
    # 
    # # 4. 训练模型
    # model = xgb.train(params, dtrain, num_boost_round=num_round)
    # 
    # # 5. 模型评估
    # preds = model.predict(dtest)
    # print('准确率3：', accuracy_score(y_test, np.argmax(preds, axis=1)))
    # 
    # # 6. 模型保存
    # # XGBoost的新版本默认使用UBJSON格式保存模型，
    # model.save_model(JSON_File)
    

    # 3. XGBoost模型训练
    model = xgb.XGBClassifier(n_estimators=10, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)

    # 4. 模型评估
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred,zero_division=0))

    # 5. 保存模型
    joblib.dump(model, PKL_File)

    
    
# 5.定义函数， 测试模型 PKL模型
def XGBoost_dm5():
    # 1. 加载模型
    model = joblib.load(PKL_File)
    
    # 2. 加载数据集
    train_data = pd.read_csv(CSV_TRAIN)
    test_data = pd.read_csv(CSV_TEST)
    
    # 3. 数据处理
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    # 4。创建网格搜索+交叉验证（结合分层采样数据），找模型最优参数组合。
    # 4.1定义变量，记录：参数组合.
    params = {'n_estimators': [50, 80, 100]
        , 'max_depth': [6, 9]
        , 'learning_rate': [0.1, 0.2]}
    
    # 4.2创建分层采样对象。
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    
    # 4.3创建网格搜索对象。
    estimator = GridSearchCV(model, params, cv=kf)
    
    # 5.模型训练
    estimator.fit(X_train, y_train)
    
    # 6.模型预测
    y_pred = estimator.predict(X_test)
    # print(f'预测结果：', y_pred)
    
    # 7.打印模型评估系数。
    print('最优估计器对象组合：', estimator.best_estimator_)
    print('网格搜索后的-最佳参数：', estimator.best_params_)
    print('网格搜索后的-准确率：', estimator.score(X_test, y_test))
    

# 5.定义函数， 测试模型 Json模型
def XGBoost_dm6():
    # 1. 加载模型 Json
    model = xgb.Booster(model_file=JSON_File)

    # 2. 加载数据集
    test_data = pd.read_csv(CSV_TEST)

    # 3. 数据处理
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    # 3.2. 将数据转换为DMatrix格式（修复关键点）
    X_test = xgb.DMatrix(X_test, label=y_test, missing=np.nan)
    
    # 4. 模型预测
    y_pred = model.predict(X_test)
    
    # 对于multi:softprob目标函数，预测结果是概率矩阵，需要转换为具体类别
    y_pred = np.argmax(y_pred, axis=1)

    # 5. 模型评估
    print('准确率6：', accuracy_score(y_test, y_pred))
    # 添加F1分数评估，需要导入f1_score
    print('F1分数评估：', f1_score(y_test, y_pred, average='macro')) 
    
    # 添加混淆矩阵，需要导入confusion_matrix\n\n
    print('混淆矩阵：\n', confusion_matrix(y_test, y_pred))  
    print('分类报告：\n', classification_report(y_test, y_pred, zero_division=0))
    
    
if __name__ == '__main__':
    # XGBoost_dm1()
    # XGBoost_dm2()
    # XGBoost_dm3()
    # XGBoost_dm4()
    XGBoost_dm5()
    # XGBoost_dm6()
