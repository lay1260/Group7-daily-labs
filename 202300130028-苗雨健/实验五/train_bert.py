import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from MRPCDataset import MRPCDataset
from transformers import BertTokenizer, BertModel

# ========== 数据加载 ==========
DATA_PATH = './msr/msr_paraphrase_train.txt'
BATCH_SIZE = 16

dataset = MRPCDataset(file_path=DATA_PATH)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
print("[INFO] Dataset loaded successfully")

# ========== 设备配置 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# ========== 模型初始化 ==========
MODEL_DIR = "./bert-base-uncased"

# Load BERT tokenizer and model
print("[INFO] Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
bert_encoder = BertModel.from_pretrained(MODEL_DIR)
bert_encoder.to(DEVICE)
bert_encoder.eval()  # Set to eval mode initially
print("[INFO] BERT model loaded")

# Initialize classification head
classifier = FCModel().to(DEVICE)
print("[INFO] Classification model initialized")

# ========== 训练配置 ==========
LEARNING_RATE = 0.001

# Setup optimizers
classifier_optim = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
bert_optim = torch.optim.Adam(bert_encoder.parameters(), lr=LEARNING_RATE)

# Loss function
criterion = torch.nn.BCELoss()


def compute_accuracy(preds, targets):
    """
    Calculate binary classification accuracy.
    Args:
        preds: predicted probabilities (tensor)
        targets: ground truth labels (tensor)
    Returns:
        accuracy value (float)
    """
    pred_binary = torch.round(preds)
    matches = (pred_binary == targets).float()
    return matches.mean().item()


def train_one_epoch():
    """
    Train the model for one epoch.
    Returns:
        average loss and accuracy for the epoch
    """
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    
    # Set models to training mode
    bert_encoder.train()
    classifier.train()
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Move data to device
        input_ids = batch_data['input_ids'].to(DEVICE)
        attn_mask = batch_data['attention_mask'].to(DEVICE)
        labels = batch_data['label'].to(DEVICE)
        
        # Clamp labels to valid range
        labels = torch.clamp(labels, min=0.0, max=1.0)
        
        # Forward pass through BERT
        bert_inputs = {
            'input_ids': input_ids,
            'attention_mask': attn_mask
        }
        with torch.set_grad_enabled(True):
            bert_results = bert_encoder(**bert_inputs)
            pooled_features = bert_results.pooler_output
            
            # Forward pass through classifier
            logits = classifier(pooled_features)
            predictions = logits.squeeze(-1)  # Remove last dimension
            
            # Compute loss
            batch_loss = criterion(predictions, labels)
            
            # Compute accuracy
            batch_acc = compute_accuracy(predictions, labels)
            
            # Backward pass
            classifier_optim.zero_grad()
            bert_optim.zero_grad()
            batch_loss.backward()
            classifier_optim.step()
            bert_optim.step()
        
        # Accumulate metrics
        batch_size = labels.size(0)
        total_loss += batch_loss.item() * batch_size
        total_acc += batch_acc * batch_size
        num_samples += batch_size
        
        # Log progress
        if torch.cuda.is_available():
            mem_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"Batch {batch_idx:3d} | Loss: {batch_loss.item():.4f} | "
                  f"Acc: {batch_acc:.4f} | GPU Mem: {mem_usage:.1f}MB")
        else:
            print(f"Batch {batch_idx:3d} | Loss: {batch_loss.item():.4f} | Acc: {batch_acc:.4f}")
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_acc = total_acc / num_samples if num_samples > 0 else 0.0
    return avg_loss, avg_acc


# ========== 主训练循环 ==========
EPOCHS = 3

print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    epoch_loss, epoch_acc = train_one_epoch()
    print(f"\n[Epoch {epoch + 1}] Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_acc:.4f}")

print("\n" + "="*50)
print("Training completed!")
print("="*50)
