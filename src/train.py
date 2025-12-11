import xgboost as xgb
import joblib
import os
from .preprocessing import load_and_preprocess_data
from utils.logger import setup_logger


def train_model(data_path, model_save_path, log_dir="log"):
    logger = setup_logger(log_dir=log_dir)
    logger.info("开始加载和预处理数据...")

    X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names =\
        load_and_preprocess_data(data_path)

    logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 创建 XGBoost 回归器
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    logger.info("开始训练模型...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    # 保存模型（XGBoost 原生格式）
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)

    # 保存 scaler 用于后续预测反标准化
    joblib.dump(scaler_X, model_save_path.replace('.json', '_scaler_X.pkl'))
    joblib.dump(scaler_y, model_save_path.replace('.json', '_scaler_y.pkl'))
    joblib.dump(feature_names, model_save_path.replace('.json', '_features.pkl'))

    logger.info(f"模型已保存至: {model_save_path}")