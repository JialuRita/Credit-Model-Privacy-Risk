import pandas as pd
import numpy as np

# 加载数据集
data = pd.read_csv('./data/split/random_label.csv')

# 随机分配 y_pred 值（0 或 1）
np.random.seed(42)  # 设置随机种子以保证结果可重复
data['y_pred'] = np.random.choice([0, 1], size=len(data))

# 保存预处理后的数据集
data.to_csv('./data/predict/validation_with_random_label.csv', index=False)
print("随机分配 y_pred 完成，保存为 validation_with_random_label.csv")
