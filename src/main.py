from src.train import train_model
from src.predict import predict

if __name__ == "__main__":
    # 训练
    train_model(
        data_path="../data/electric_data.csv",
        model_save_path="../model/xgboost_model.json",
        log_dir="../log"
    )

    # 示例预测（实际应用中应从外部输入）
    pred = predict(
        model_path="../model/xgboost_model.json",
        scaler_x_path="../model/xgboost_model_scaler_X.pkl",
        scaler_y_path="../model/xgboost_model_scaler_y.pkl",
        features_path="../model/xgboost_model_features.pkl",
        input_data={
            "time": "2011-12-01 00:00:00",
            "temperature": 25.0,
            "humidity": 60.0,
            "load": 100.0
        }
    )
    print(f"预测负荷: {pred:.2f} MW")
    

   