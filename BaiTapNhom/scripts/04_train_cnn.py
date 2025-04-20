import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =======================
# 1. Thiết lập cấu hình
# =======================
# Đường dẫn đến file đặc trưng và nhãn
FEATURE_PATH = 'D:/NCKH/speech_detection_project/features/cnn/X_dev.npy'
LABEL_PATH = 'D:/NCKH/speech_detection_project/features/cnn/y_dev.npy'
FEATURE_TEST_PATH = 'D:/NCKH/speech_detection_project/features/cnn/X_eval.npy'
LABEL_TEST_PATH = 'D:/NCKH/speech_detection_project/features/cnn/y_eval.npy'
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# =======================
# 2. Đọc dữ liệu đặc trưng
# =======================
print("Đang tải dữ liệu đặc trưng và nhãn...")
X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
X_test = np.load(FEATURE_TEST_PATH)
y_test = np.load(LABEL_TEST_PATH)

# =======================
# 3. Dataset & DataLoader
# =======================
class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]  # (1, 100, 13)
    
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

# Tạo đối tượng dataset cho train, validation, và test
train_dataset = SpeechDataset(X_train, y_train)
val_dataset = SpeechDataset(X_val, y_val)
test_dataset = SpeechDataset(X_test, y_test)

# Tạo DataLoader cho từng tập
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Kiểm tra số lượng batch
print(f"Số lượng batch trong train: {len(train_loader)}")
print(f"Số lượng batch trong validation: {len(val_loader)}")
print(f"Số lượng batch trong test: {len(test_loader)}")

# =======================
# 4. Xây dựng mô hình CNN
# =======================
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(64 * 25 * 3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 50, 6)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 25, 3)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =======================
# 5. Huấn luyện mô hình
# =======================
EPOCHS = 50
PATIENCE = 7
BEST_MODEL_PATH = r"D:\NCKH\speech_detection_project\models\best_model_cnn.pth"

# Khởi tạo mô hình và optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
best_val_loss = float('inf')
patience_counter = 0


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Tính loss và cập nhật weights
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Tính độ chính xác
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    
    # Đánh giá trên validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
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

# =======================
# 6. Đánh giá mô hình
# =======================
print("\u001b[33mĐang đánh giá mô hình...\u001b[0m")
model.eval()
test_loss = 0
correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

test_loss /= len(test_loader)
test_acc = correct / total
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# In confusion matrix và classification report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("Classification Report:")
print(classification_report(y_true, y_pred))