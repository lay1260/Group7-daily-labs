from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 本地模型路径（需提前下载到服务器，例如放在./bert-base-uncased目录）
local_model_path = "./bert-base-uncased"  

# 加载本地tokenizer和模型（替换为bert-base-uncased）
tokenizer = BertTokenizer.from_pretrained(local_model_path)
# 此处示例使用基础模型结构，若需分类功能可替换为微调后的模型文件
model = BertForSequenceClassification.from_pretrained(
    local_model_path,
    num_labels=2  # 二分类任务（根据实际微调任务调整）
)

# 移动模型到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 文本预测函数
def predict_sentiment(text):
    # 编码输入
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推理
    with torch.no_grad():
        outputs = model(** inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()
    return "积极" if sentiment == 1 else "消极", probs

# 测试
text = "The weather is great today, I feel very happy!"
sentiment, probability = predict_sentiment(text)
print(f"文本: {text}")
print(f"预测的情感: {sentiment}")
print(f"概率: {probability}")
