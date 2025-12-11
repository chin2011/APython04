import pandas as pd
import numpy as np
from datetime import datetime
import os

# 1. 读取原始数据（修复格式问题）
with open("../data/COMED_hourly.csv", "r") as f:
    content = f.read()

# 原始文件无换行，需按固定模式分割
# 每条记录形如: YYYY-MM-DD HH:MM:SS,float
import re
matches = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),([0-9.]+)', content)

if not matches:
    raise ValueError("未能解析时间与负荷数据，请检查文件格式")

df = pd.DataFrame(matches, columns=["time", "load"])
df["time"] = pd.to_datetime(df["time"])
df["load"] = df["load"].astype(float)

# 按时间排序（原始顺序混乱）
df = df.sort_values("time").reset_index(drop=True)

# 2. 模拟 temperature 和 humidity（基于月份）
def simulate_weather(dt):
    month = dt.month
    # 芝加哥典型月均温（摄氏度）近似值
    avg_temp = {
        1: -5, 2: -4, 3: 2, 4: 9, 5: 15, 6: 21,
        7: 24, 8: 23, 9: 18, 10: 11, 11: 4, 12: -2
    }.get(month, 0)
    # 加入随机波动 ±5°C
    temp = avg_temp + np.random.normal(0, 3)
    # 湿度：冬季高（70~90%），夏季稍低（50~80%）
    if month in [12, 1, 2]:
        humidity = np.random.uniform(75, 90)
    elif month in [6, 7, 8]:
        humidity = np.random.uniform(50, 75)
    else:
        humidity = np.random.uniform(60, 85)
    return round(temp, 1), round(humidity, 1)

# 应用模拟
np.random.seed(42)  # 可复现
df[["temperature", "humidity"]] = df["time"].apply(
    lambda t: pd.Series(simulate_weather(t))
)

# 3. 选择列并保存
output_df = df[["time", "temperature", "humidity", "load"]]
output_df.to_csv("../data/electric_data.csv", index=False)

print(f"✅ 成功生成 load_data.csv，共 {len(output_df)} 条记录。")
print(output_df.head())