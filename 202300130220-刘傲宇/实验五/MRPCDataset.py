import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MRPCDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_length=128):
        """
        初始化数据集，加载数据和准备预处理。
        :param file_path: 数据集文件路径
        :param tokenizer: BERT tokenizer
        :param max_length: 最大输入序列长度
        """
        local_model_path = "./bert-base-uncased"
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(local_model_path)
        self.max_length = max_length

        # 加载数据
        self.data = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # 使用 'utf-8-sig' 以跳过 BOM
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')  # 使用制表符（tab）分隔字段
                if len(parts) < 5:  # 防止格式错误的行
                    continue
                label = int(parts[0])  # 标签为第一列
                sentence1 = parts[3]  # 第一个句子为第四列
                sentence2 = parts[4]  # 第二个句子为第五列
                self.data.append((sentence1, sentence2, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取每个样本
        sentence1, sentence2, label = self.data[idx]

        # 确保 sentence1 和 sentence2 都是字符串
        assert isinstance(sentence1, str), f"Expected string but got {type(sentence1)}"
        assert isinstance(sentence2, str), f"Expected string but got {type(sentence2)}"

        # 对句子进行编码
        encoding = self.tokenizer(
            sentence1, sentence2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 获取输入数据并转换为tensor
        input_ids = encoding['input_ids'].squeeze(0)  # 去掉batch维度
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }
