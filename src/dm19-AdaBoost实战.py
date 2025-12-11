'''
案例AdaBoost实战 葡萄酒数据
    CSV:
        "fixed acidity";"volatile acidity";"citric acid";"residual sugar";
        "chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";
        "pH";"sulphates";"alcohol";"quality"
        
AdaBoost算法介绍：
    它属于Boosting思想，即：串行执行，每次使用全部样本，最后加权投票.
    原理：
        1.使用全部样本，通过决策树模型（第1个弱分类器）进行训练，获取结果.
            思路：
                预测正确--→权重下降
                预测错误--→权重上升
        2.把第1个弱分类器的处理结果，交给第2个弱分类器进行训练，获取结果.
            思路：
                预测正确--→权重下降
                预测错误--→权重上升
        3.依次类推，串行执行，直至获取最终结果。
 

'''
from sklearn.tree import DecisionTreeClassifier

CSV_FILE = '../data/winequality-red.csv'

# 导入所需库
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier # AdaBoost分类器→集成学习Boosting思想
from sklearn.model_selection import train_test_split # 训练集、测试集分割
from sklearn.metrics import  accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score


# 1.定义函数，演示：AdaBoost实战葡萄酒数据
def wine_quality():
    # 1.数据处理
    data = pd.read_csv(CSV_FILE, sep=';')
    # data.info() 
    X = data.drop(['quality'], axis=1).copy()
    y = data['quality']
    # 数据集划分
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=22)

    # 2.模型训练
    # 实例化单决策树实例化Adaboost-由500颗树组成
    # mytree = DecisionTreeClassifier()
    mytree = DecisionTreeClassifier(criterion='entropy',max_depth=1, random_state=22)
    # 参数: estimator: 基学习器, n_estimators: 基学习器的数量, learning_rate: 学习率
    # 注意：在新版本的scikit-learn中，参数名从 base_estimator 改为 estimator
    model1 = AdaBoostClassifier(estimator=mytree, 
                                n_estimators=200,learning_rate=0.1, random_state=22)
    
    # 3.单决策树训练和评估
    mytree.fit(X_train, y_train)
    myscore = mytree.score(X_test, y_test)
    print('单决策树准确率:', myscore)
    
    # 4.AdaBoost训练和评估
    model1.fit(X_train, y_train)
    # 模型预测
    y_pre = model1.predict(X_test)
    #对于多分类问题，precision_score、recall_score和f1_score需要指定average参数。
    # 默认情况下它们假设是二分类问题(average='binary')，但你的葡萄酒质量数据是一个多分类问题。
    # print('AdaBoost-单决策树精确率:', precision_score(y_test, y_pre))
    
    print('AdaBoost-单决策树准确率:', accuracy_score(y_test, y_pre))
    print('AdaBoost-单决策树召回率:', recall_score(y_test, y_pre, average='macro'))
    print('AdaBoost-单决策树F1-score:', f1_score(y_test, y_pre, average='macro'))
    print('AdaBoost-单决策树F1-score:', f1_score(y_test, y_pre, average='micro'))

    
    
    
if __name__ == '__main__':
    wine_quality()
