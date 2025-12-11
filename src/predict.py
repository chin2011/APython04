import xgboost as xgb
import joblib
import numpy as np
import pandas as pd


def predict(model_path, scaler_x_path, scaler_y_path, features_path, input_data):
    """
    :param input_data: dict or DataFrame，包含与训练时相同的特征
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    feature_names = joblib.load(features_path)

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data

    X = input_df[feature_names]
    X_scaled = scaler_X.transform(X)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    return y_pred[0]  # 单步预测返回标量