import torch
import torch.nn as nn

class FCModel(nn.Module):
    """
    两层多层感知机（MLP）
    输入：BERT pooler_output (768维)
    隐藏层：256维
    输出：1维（sigmoid激活，对应二分类概率）
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1):
        super(FCModel, self).__init__()
        # 定义两层全连接+激活函数
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # 防止过拟合
        self.sigmoid = nn.Sigmoid()     # 输出概率（适配BCELoss）

    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out