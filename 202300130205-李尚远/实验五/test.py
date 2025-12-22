import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from torch.utils.data import DataLoader, TensorDataset,Dataset
from transformers import BertTokenizer, BertModel  # HuggingFace BERT组件
import torch.nn as nn
import time
import pandas as pd
import tqdm  # 用于显示训练进度条
#MRPCDataset
class MRPCDataset(Dataset):
    def __init__(self, data_dir="/root/autodl-tmp/data6",split="train"):
        self.data_dir = data_dir
        self.split = split
        if self.split == "train":
            self.file_path=f"{data_dir}/msr_paraphrase_train.txt"
        else:
            self.file_path=f"{data_dir}/msr_paraphrase_test.txt"
        self.data=pd.read_csv(
            self.file_path,
            sep="\t",
            header=0,
            encoding="utf-8",
            on_bad_lines="skip"
            )
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item=self.data.iloc[idx]
        sen1=str(item['#1 String'])
        sen2=str(item['#2 String'])
        label=int(item['Quality'])
        return (sen1,sen2),label
def collate_fn(batch):
    sen1_list=[sample[0][0] for sample in batch]
    sen2_list=[sample[0][1] for sample in batch]

    labels=torch.tensor([sample[1] for sample in batch],dtype=torch.float32)
    return (sen1_list,sen2_list),labels
#全连接层
class FCModel(nn.Module):
    def __init__(self, input_size=768, hidden1_size=512,hidden2_size=256,output_size=1):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, X):
        out = self.fc1(X)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def main():
    # 1. 数据加载
    print("正在加载数据集...")
    mrpc_dataset = MRPCDataset()
    train_loader = DataLoader(
        dataset=mrpc_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,  # 使用多进程加载数据，加快速度
        collate_fn=collate_fn
    )
    print(f"数据载入完成，训练集批次数量: {len(train_loader)}")

    # 2. 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 3. 加载BERT模型和分词器
    print("正在加载BERT模型...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.to(device)
    print("BERT模型加载完成")

    # 4. 创建全连接层模型
    model = FCModel()
    model = model.to(device)
    print("全连接层模型创建完成")

    # 5. 定义优化器和损失函数
    # BERT通常使用较小的学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=2e-5)  # BERT学习率通常较小
    criterion = torch.nn.BCELoss()

    # 6. 计算准确率的函数
    def binary_accuracy(predict, label):
        rounded_predict = torch.round(predict)
        correct = (rounded_predict == label).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

    # 7. 训练函数
    def train_epoch(epoch):
        # 记录统计信息
        epoch_loss, epoch_acc = 0., 0.
        total_len = 0

        # 设置模型为训练模式
        bert_model.train()
        model.train()

        # 使用tqdm显示进度条
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

        for i, data in progress_bar:
            # 内存使用监控（可选）
            # if device.type == 'cuda':
            #     print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            (sen1_list,sen2_list), labels = data  # 假设data包含句子对和标签
            labels = labels.to(device).float()  # 确保标签是float类型，与BCELoss匹配
            # 分词处理
            encoding = tokenizer(
                sen1_list,
                sen2_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128  # 限制最大长度，加速训练并防止内存溢出
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # BERT前向传播
            with torch.set_grad_enabled(True):
                bert_output = bert_model(**encoding)
                pooler_output = bert_output.pooler_output

                # 全连接层前向传播
                predict = model(pooler_output).squeeze()

                # 计算损失和准确率
                loss = criterion(predict, labels)
                acc = binary_accuracy(predict, labels)

                # 梯度清零
                optimizer.zero_grad()
                bert_optimizer.zero_grad()

                # 反向传播和参数更新
                loss.backward()
                optimizer.step()
                bert_optimizer.step()

            # 累积损失和准确率
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            total_len += batch_size

            # 更新进度条信息
            progress_bar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'batch_acc': f'{acc.item():.4f}'
            })

            # 清除不需要的变量，节省内存
            del encoding, bert_output, pooler_output, predict, loss, acc

        # 计算平均损失和准确率
        avg_loss = epoch_loss / total_len
        avg_acc = epoch_acc / total_len

        return avg_loss, avg_acc

    # 8. 开始训练
    num_epochs = 3  # 增加训练轮次
    print(f"开始训练，共 {num_epochs} 轮")

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_loss, epoch_acc = train_epoch(epoch)

        # 计算本轮训练时间
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.4f}, 耗时: {epoch_time:.2f}秒")
        print("-" * 50)

        # 可选：每轮保存模型
        # torch.save({
        #     'bert_model_state_dict': bert_model.state_dict(),
        #     'fc_model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'bert_optimizer_state_dict': bert_optimizer.state_dict(),
        # }, f'mrpc_model_epoch_{epoch+1}.pth')

    print("训练完成")


if __name__ == "__main__":
    main()
