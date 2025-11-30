import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MRPCDataset(Dataset):
    """
    Microsoft Research Paraphrase Corpus (MRPC) Dataset Loader
    用于加载和预处理MRPC数据集，支持BERT tokenization
    """
    def __init__(self, file_path, tokenizer=None, max_length=128):
        """
        构造函数：初始化数据集
        参数说明：
            file_path: 数据文件路径（TSV格式）
            tokenizer: 可选的BERT tokenizer，如果为None则自动加载
            max_length: 文本序列的最大长度限制
        """
        model_path = "./bert-base-uncased"
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer
        self.max_seq_len = max_length
        
        # 读取并解析数据文件
        self.samples = self._load_data(file_path)

    def _load_data(self, file_path):
        """
        内部方法：从文件中加载数据
        返回格式：[(sentence1, sentence2, label), ...]
        """
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                next(f)  # 跳过第一行表头
                for line_num, line in enumerate(f, start=2):
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split('\t')
                    # 验证数据格式：至少需要5列
                    if len(fields) < 5:
                        print(f"警告：第{line_num}行数据格式不正确，已跳过")
                        continue
                    try:
                        label_val = int(fields[0])
                        sent1 = fields[3].strip()
                        sent2 = fields[4].strip()
                        samples.append((sent1, sent2, label_val))
                    except (ValueError, IndexError) as e:
                        print(f"警告：第{line_num}行数据解析失败: {e}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {file_path}")
        return samples

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        返回字典，包含input_ids、attention_mask和label
        """
        sent1, sent2, target_label = self.samples[index]
        
        # 类型检查
        if not isinstance(sent1, str) or not isinstance(sent2, str):
            raise TypeError(f"句子必须是字符串类型，但得到: {type(sent1)}, {type(sent2)}")
        
        # BERT tokenization处理
        encoded = self.tokenizer(
            sent1,
            sent2,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        
        # 移除batch维度（因为DataLoader会自动添加）
        input_ids = encoded['input_ids'].squeeze(0)
        attn_mask = encoded['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'label': torch.tensor(target_label, dtype=torch.float32)
        }
