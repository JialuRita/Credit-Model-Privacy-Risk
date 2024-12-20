import torch

class MLPModel(torch.nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 100)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(100, 50)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(50, 2)  # 二分类问题

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x