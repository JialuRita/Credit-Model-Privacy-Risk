import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.MLP import MLPModel


def model_predict(credit_model_path, validation_data, validation_predictions_data):
    # 加载验证数据
    validation_data = pd.read_csv(validation_data)

    # 分离特征和目标变量
    X_validation = torch.tensor(validation_data.drop(columns=['y']).values, dtype=torch.float32)
    y_validation = torch.tensor(validation_data['y'].values, dtype=torch.long)

    # 定义数据加载器
    batch_size = 64
    validation_dataset = TensorDataset(X_validation, y_validation)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_size = X_validation.shape[1]
    model = MLPModel(input_size)

    # 加载训练好的模型参数
    model.load_state_dict(torch.load(credit_model_path))
    model.eval()

    # 在验证集上进行预测
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in validation_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())

    # 将预测结果保存到 CSV 文件
    validation_data['y_pred'] = y_pred
    validation_data.to_csv(validation_predictions_data, index=False)
    print("预测结果已保存到 validation_with_predictions.csv")
