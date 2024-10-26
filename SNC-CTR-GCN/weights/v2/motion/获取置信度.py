import pickle
import pandas as pd
import numpy as np
from scipy.special import softmax
from collections import OrderedDict

# 加载 .pkl 文件
file_path = 'best.pkl'  # 请确保这是正确的文件路径
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 获取 predictions 数据
predictions = data['predictions']

# 创建一个 OrderedDict 来存储结果
result = OrderedDict()

# 填充 result 字典
for i, prob in enumerate(predictions):
    result[i] = prob

# 保存为 best_score.pkl 文件
output_file = 'best_score.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(result, f)

print(f"File '{output_file}' has been created with {len(result)} items.")

# 验证文件
with open(output_file, 'rb') as f:
    loaded_data = pickle.load(f)

print(f"Loaded data contains {len(loaded_data)} items.")
print(f"First item: {list(loaded_data.items())[0]}")
