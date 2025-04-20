import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================
# 1. Thiết lập cấu hình
# ============================
FEATURE_PATH = 'D:/NCKH/speech_detection_project/features/cnn/X_dev.npy'
LABEL_PATH = 'D:/NCKH/speech_detection_project/features/cnn/y_dev.npy'
FEATURE_TEST_PATH = 'D:/NCKH/speech_detection_project/features/cnn/X_eval.npy'
LABEL_TEST_PATH = 'D:/NCKH/speech_detection_project/features/cnn/y_eval.npy'
SEED = 42
BATCH_SIZE = 64

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================
# 2. Đọc dữ liệu đặc trưng
# ============================
print("\u001b[33mĐang tải dữ liệu đặc trưng và nhãn...\u001b[0m")
X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
X_test = np.load(FEATURE_TEST_PATH)
y_test = np.load(LABEL_TEST_PATH) 

# ============================
# 3. Dataset & DataLoader
# ============================
class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # MFCC
        self.y = torch.tensor(y, dtype=torch.long)      # Labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # reshape (100, 13) → (1, 100, 13) để dùng Conv2D
        return self.X[idx].unsqueeze(0), self.y[idx] 
    

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
# Đếm số lượng mỗi lớp
class_counts = np.bincount(y_train)  # [non-speech, speech]
class_weights = 1. / class_counts
weights = class_weights[y_train]

train_dataset = SpeechDataset(X_train, y_train)
val_dataset   = SpeechDataset(X_val, y_val)
test_dataset  = SpeechDataset(X_test, y_test)

# WeightedRandomSampler 
sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Số lượng batch trong train:", len(train_loader))
print("Số lượng batch trong val:", len(val_loader))
print("Số lượng batch trong test:", len(test_loader))

# ============================
# 4. Xây dựng mô hình CNN_LSTM
# ============================
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        
        # Convolution Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # output shape: (32, 50, 6)
        )
        
        # Convolution Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # output shape: (64, 25, 3)
        )
        
        # Convolution Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # (128, 12, 1)
        )

        # Flatten và reshape để đưa vào LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, 2)  # bidirectional nên hidden*2

    def forward(self, x):
        # x: (batch, 1, 100, 13)
        x = self.conv1(x)  # -> (batch, 32, 50, 6)
        x = self.conv2(x)  # -> (batch, 64, 25, 3)
        x = self.conv3(x)  # -> (batch, 128, 12, 1)

        # Chuyển từ (batch, channels, time, feature) sang (batch, time, feature)
        x = x.permute(0, 2, 1, 3)  # (batch, 12, 128, 1)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (batch, 12, 128)
        
        out, _ = self.lstm(x)       # (batch, seq_len, 256)
        out = out[:, -1, :]         # (batch, 256)


        out = self.dropout(out)
        out = self.fc(out)     # -> (batch, 2)
        return out


# === Kỹ thuật Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')  # (batch,)
        pt = torch.exp(-BCE_loss)  # pt = softmax probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# ============================
# 5. Huấn luyện mô hình
# ============================
EPOCHS = 50
PATIENCE = 10  # số epoch không cải thiện trước khi dừng
BEST_MODEL_PATH = r"D:\NCKH\speech_detection_project\models\best_model_cnn_lstm.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Model().to(device)
criterion = FocalLoss(alpha=0.7, gamma=2.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_val_loss = float("inf")
patience_counter = 0

print("\u001b[33mĐang huấn luyện mô hình ...\u001b[0m")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # === Early Stopping Check ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)  # Lưu model tốt nhất
        print("Đã lưu mô hình tốt nhất.")
    else:
        patience_counter += 1
        print(f"Không cải thiện. patience_counter = {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Dừng huấn luyện sớm do không cải thiện.")
            break

# ============================
# 6. Đánh giá mô hình
# ============================
print("\u001b[33mĐang đánh giá mô hình...\u001b[0m")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load(r"D:\NCKH\speech_detection_project\models\best_model_cnn_lstm.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Chuyển về numpy array
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# === In kết quả đánh giá ===
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))