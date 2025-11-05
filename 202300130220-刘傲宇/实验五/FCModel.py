import torch
import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self):
        """
        初始化全连接层模型
        """
        super(FCModel, self).__init__()

        # 定义一个包含两个全连接层的简单网络
        self.fc1 = nn.Linear(768, 256)  # BERT输出的维度为768
        self.fc2 = nn.Linear(256, 1)  # 输出一个标量，表示二分类
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数
        :param x: BERT模型的池化输出，形状为(batch_size, 768)
        :return: 预测的概率值
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # 输出一个0到1之间的概率值
        return x
