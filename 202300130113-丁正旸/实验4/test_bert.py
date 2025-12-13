# 先关闭torchvision的Beta警告（消除无关提示）
import torchvision
torchvision.disable_beta_transforms_warning()

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 国内镜像可访问的中文RoBERTa模型（适配BertTokenizer）
model_name = "hfl/chinese-roberta-wwm-ext"  

# 关键修正：换回BertTokenizer（匹配模型的实际Tokenizer类型）
tokenizer = BertTokenizer.from_pretrained(
    model_name,
    mirror="https://hf-mirror.com"  # 强制国内镜像下载
)

# 加载模型：指定二分类，忽略分类头尺寸不匹配警告
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 二分类（消极/积极）
    ignore_mismatched_sizes=True,  # 解决预训练模型无分类头的报错
    mirror="https://hf-mirror.com"
)

# 移动模型到GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 情感预测函数（逻辑不变）
def predict_sentiment(text):
    # 编码输入（适配中文，max_length=512）
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    # 输入移到对应设备
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 推理（无梯度计算）
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 计算概率并判断情感
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()
    
    return "积极" if sentiment == 1 else "消极", probs

# 测试文本
text = "今天的天气真好，我感到非常开心！"
# text = input("请输入测试文本：")  # 可手动输入测试

# 预测并输出结果
sentiment, probability = predict_sentiment(text)
print(f"文本: {text}")
print(f"预测的情感: {sentiment}")
print(f"概率: {probability.cpu().numpy()}")  # 转numpy更易读