import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Đường dẫn đến file đặc trưng và nhãn
FEATURE_PATH = 'D:/NCKH/speech_detection_project/features/rf/X_dev.npy'
LABEL_PATH = 'D:/NCKH/speech_detection_project/features/rf/y_dev.npy'
FEATURE_TEST_PATH = 'D:/NCKH/speech_detection_project/features/rf/X_eval.npy'
LABEL_TEST_PATH = 'D:/NCKH/speech_detection_project/features/rf/y_eval.npy'
RANDOM_STATE = 42

# ========================
# 1. Tải dữ liệu đặc trưng và nhãn
# ========================
print("Đang tải dữ liệu đặc trưng và nhãn...")
X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)
X_test = np.load(FEATURE_TEST_PATH)
y_test = np.load(LABEL_TEST_PATH)

# ========================
# 2. Chia dữ liệu
# ========================
# Tách dữ liệu dev thành train và validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
)

# ========================
# 3. Xây dựng mô hình Random Forest
# ========================
rf_model = RandomForestClassifier(
    n_estimators=100,  # Số lượng cây
    random_state=RANDOM_STATE,   # Để kết quả có thể tái tạo
    class_weight="balanced",  # Cân bằng trọng số lớp
)

# ========================
# 4. Huấn luyện mô hình
# ========================
print("Đang huấn luyện mô hình...")
rf_model.fit(X_train, y_train)

# ========================
# 5. Đánh giá mô hình trên tập validation
# ========================
y_val_pred = rf_model.predict(X_val)

print("Đánh giá trên tập validation:")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# ========================
# 6. Đánh giá mô hình trên tập test (eval)
# ========================
y_test_pred = rf_model.predict(X_test)

print("Đánh giá trên tập test (eval):")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
