import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# 1. 加载数据
print("正在加载训练数据...")
train_df = pd.read_csv('../data/train.csv')
X_train = train_df.iloc[:, 1:].values  # 像素值 (784列)
y_train = train_df.iloc[:, 0].values   # 标签 (第0列)

print("正在加载测试数据...")
test_df = pd.read_csv('../data/test.csv')
X_test = test_df.values  # 测试集只有像素值

# （可选）如果你有真实标签用于评估，比如 sample_submission.csv 或带标签的 test_with_labels.csv
# 否则跳过 accuracy 计算，只做预测
# 假设你有一个带标签的验证集或保留一部分训练集做验证
from sklearn.model_selection import train_test_split
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 2. 创建 KNN 模型（建议先用小样本调试）
k = 3
print(f"正在训练 KNN 模型 (k={k})...")
knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  # n_jobs=-1 使用所有 CPU 核心

start_time = time.time()
knn.fit(X_train_part, y_train_part)
print(f"训练完成，耗时: {time.time() - start_time:.2f} 秒")

# 3. 验证模型性能（在验证集上）
y_val_pred = knn.predict(X_val)
print("验证集准确率:", accuracy_score(y_val, y_val_pred))
print("\n分类报告:\n", classification_report(y_val, y_val_pred))

# 4. 在完整训练集上重新训练（可选，提升效果）
print("在完整训练集上重新训练...")
knn_full = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
knn_full.fit(X_train, y_train)

# 5. 对测试集进行预测
print("正在预测测试集...")
test_pred = knn_full.predict(X_test)

# 6. 保存结果（符合 Kaggle 提交格式）
submission = pd.DataFrame({
    'ImageId': range(1, len(test_pred) + 1),
    'Label': test_pred
})
submission.to_csv('../data/submission_knn.csv', index=False)
print("预测结果已保存为 submission_knn.csv")