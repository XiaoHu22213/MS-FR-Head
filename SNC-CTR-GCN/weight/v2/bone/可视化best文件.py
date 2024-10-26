import pickle
import pandas as pd
import numpy as np
from scipy.special import softmax

# 加载 .pkl 文件
file_path = 'best.pkl'  # 请确保这是正确的文件路径
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 获取 predictions 数据
predictions = data['predictions']

# 应用 softmax 函数将置信度转换为概率
probabilities = softmax(predictions, axis=1)

# 创建一个 DataFrame
df = pd.DataFrame(probabilities)

# 将所有数值四舍五入到 4 位小数
df = df.round(2)

# 生成列名（从 1 到 155）
column_names = [f'Class_{i}' for i in range(155)]

# 设置 DataFrame 的列名
df.columns = column_names

# 保存为 CSV 文件
output_file = 'predictions.csv'
df.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been created with {df.shape[0]} rows and {df.shape[1]} columns.")

# 打印第一行的和，以验证概率和为1
print(f"Sum of probabilities for the first row: {df.iloc[0].sum():.4f}")
