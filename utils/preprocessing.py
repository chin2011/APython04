import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_and_preprocess_data(data_path, target_col='load', test_size=0.2, random_state=42):
    """
    加载并预处理数据
    :param data_path: 数据路径
    :param target_col: 目标列名
    :return: X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    df = pd.read_csv(data_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False  # 时间序列通常不打乱
    )

    # 标准化特征（可选，XGBoost 对量纲不敏感，但有时有助于提升）
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 标准化目标（用于逆变换，如果需要）
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    return (
        X_train_scaled, X_test_scaled,
        y_train_scaled, y_test_scaled,
        scaler_X, scaler_y,
        X.columns.tolist()
    )