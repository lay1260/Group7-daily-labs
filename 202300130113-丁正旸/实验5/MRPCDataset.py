import torch
from torch.utils.data import Dataset
import os

class MRPCDataset(Dataset):
    """
    MRPC数据集加载器
    处理逻辑：
    1. 读取MRPC的txt文件（train/test）
    2. 解析每一行的标签、句子1、句子2
    3. 适配PyTorch Dataset接口
    """
    def __init__(self, data_dir="./data", mode="train"):
        """
        参数：
        - data_dir: 数据集存放目录
        - mode: "train" 或 "test"，指定加载训练/测试集
        """
        self.data_dir = data_dir
        self.mode = mode
        self.data = self._load_data()

    def _load_data(self):
        # 拼接数据集文件路径
        file_name = "msr_paraphrase_train.txt" if self.mode == "train" else "msr_paraphrase_test.txt"
        file_path = os.path.join(self.data_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MRPC数据集文件未找到：{file_path}\n请从链接下载并解压至data目录：https://www.microsoft.com/en-us/download/details.aspx?id=52398")
        
        data = []
        # 读取并解析数据集（跳过表头，按\t分割）
        with open(file_path, "r", encoding="utf-8") as f:
            next(f)  # 跳过第一行表头（Quality	#1 ID	#2 ID	#1 String	#2 String）
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 分割列（处理句子中含\t的边界情况）
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                label = int(parts[0])       # 标签：0=不同义，1=同义
                sent1 = parts[3].strip()    # 句子1
                sent2 = parts[4].strip()    # 句子2
                data.append((sent1, sent2, label))
        return data

    def __len__(self):
        # 返回数据集总长度
        return len(self.data)

    def __getitem__(self, idx):
        # 按索引返回单条数据（句子1、句子2、标签）
        sent1, sent2, label = self.data[idx]
        return sent1, sent2, label