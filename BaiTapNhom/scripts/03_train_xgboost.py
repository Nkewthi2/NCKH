import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
# 2. Chia dữ liệu (dev → train + val và eval là test)
# ========================
# Tách dữ liệu dev thành train và validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
)

# ========================
# 3. Tính toán trọng số lớp (class weights) nếu dữ liệu mất cân bằng
# ========================
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# ========================
# 4. Xây dựng mô hình XGBoost
# ========================
params = {
    'objective': 'binary:logistic',    # Mục tiêu là phân loại nhị phân
    'eval_metric': 'logloss',          # Sử dụng logloss cho đánh giá
    'eta': 0.1,                        # Learning rate
    'max_depth': 6,                    # Chiều sâu tối đa của cây
    'subsample': 0.8,                  # Tỉ lệ mẫu con
    'colsample_bytree': 0.8,           # Tỉ lệ đặc trưng cho mỗi cây
    'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],  # Điều chỉnh trọng số lớp
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# ========================
# 5. Huấn luyện mô hình
# ========================
print("Đang huấn luyện mô hình...")
evals = [(dtrain, 'train'), (dval, 'eval')]  # Đánh giá trên train và validation
num_round = 100  # Số vòng huấn luyện

# Huấn luyện mô hình
bst = xgb.train(params, dtrain, num_round, evals, early_stopping_rounds=10)

# ========================
# 6. Đánh giá mô hình trên tập validation
# ========================
y_val_pred = bst.predict(dval)
y_val_pred = (y_val_pred > 0.5).astype(int)  # Chuyển đổi sang nhãn nhị phân

print("Đánh giá trên tập validation:")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# ========================
# 7. Đánh giá mô hình trên tập test (eval)
# ========================
y_test_pred = bst.predict(dtest)
y_test_pred = (y_test_pred > 0.5).astype(int)  # Chuyển đổi sang nhãn nhị phân

print("Đánh giá trên tập test (eval):")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
