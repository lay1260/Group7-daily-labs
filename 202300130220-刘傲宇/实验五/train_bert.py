import torch
from torch.utils.data import DataLoader
from FCModel import FCModel  # 自定义全连接层模型
from MRPCDataset import MRPCDataset  # 自定义MRPC数据集
from transformers import BertTokenizer, BertModel  # HuggingFace的BERT组件

# 载入数据预处理模块
mrpcDataset = MRPCDataset(file_path='./msr/msr_paraphrase_train.txt')  # 传入数据路径
train_loader = DataLoader(dataset=mrpcDataset, batch_size=16, shuffle=True)
print("数据载入完成")

# 设置运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("设备配置完成")

# 加载BERT模型
local_model_path = "./bert-base-uncased"  # 本地模型目录
tokenizer = BertTokenizer.from_pretrained(local_model_path)
bert_model = BertModel.from_pretrained(local_model_path)
bert_model.to(device)
print("BERT模型加载完成")

# 创建全连接层模型对象
model = FCModel().to(device)
print("全连接层模型创建完成")

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()

# 计算准确率的公式
def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict)
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


# 训练方法
def train():
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    for i, data in enumerate(train_loader):
        print("当前显存使用情况：", torch.cuda.memory_allocated())

        bert_model.train()
        model.train()

        input_ids = data['input_ids'].to(device)  # 直接使用 tokenized 的 input_ids
        attention_mask = data['attention_mask'].to(device)  # 输入句子的attention_mask
        label = data['label'].to(device)  # 标签

        # 检查 label 是否在 [0, 1] 范围内
        #print(f"标签值的范围: min={label.min().item()}, max={label.max().item()}")

        # 确保标签在 [0, 1] 之间
        label = torch.clamp(label, 0, 1)

        encoding = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        # 传递给 BERT 模型进行前向传播
        bert_output = bert_model(**encoding)
        pooler_output = bert_output.pooler_output  # 获取BERT模型池化后的输出

        # 通过全连接层模型进行预测
        predict = model(pooler_output).squeeze()  # [batch_size, 1] -> [batch_size]

        # 计算损失和准确率
        loss = crit(predict, label.float())
        acc = binary_accuracy(predict, label)

        # 梯度清零、反向传播、优化
        optimizer.zero_grad()
        bert_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bert_optimizer.step()

        epoch_loss += loss.item() * len(label)
        epoch_acc += acc.item() * len(label)
        total_len += len(label)

        # 打印每个batch的loss和准确率
        print(f"Batch {i}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    return epoch_loss / total_len, epoch_acc / total_len

# 开始训练
num_epochs =3
for epoch in range(num_epochs):
    epoch_loss, epoch_acc = train()
    print(f"EPOCH {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
