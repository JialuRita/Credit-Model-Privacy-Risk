import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

# 加载训练集和验证集
train_data = pd.read_csv('./data/split/train.csv')
validation_data = pd.read_csv('./data/split/validation.csv')

# 分离特征和目标变量
X_train = torch.tensor(train_data.drop(columns=['y']).values, dtype=torch.float32)
y_train = torch.tensor(train_data['y'].values, dtype=torch.long)
X_validation = torch.tensor(validation_data.drop(columns=['y']).values, dtype=torch.float32)
y_validation = torch.tensor(validation_data['y'].values, dtype=torch.long)

# 定义数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
validation_dataset = TensorDataset(X_validation, y_validation)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# 定义MLP模型
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 2)  # 二分类问题

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
input_size = X_train.shape[1]
model = MLPModel(input_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

# 在验证集上评估模型
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in validation_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(y_batch.tolist())

# 计算准确性和分类报告
accuracy = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

report = classification_report(y_true, y_pred)
print("Classification Report:\n", report)

# 保存模型
torch.save(model.state_dict(), './model/mlp_model.pth')
print("模型已保存为 mlp_model.pth")
