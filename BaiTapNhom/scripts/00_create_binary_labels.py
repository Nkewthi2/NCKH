import os
import pandas as pd
from tqdm import tqdm



# ========================
# 1. Thiết lập cấu hình
# ========================
DATA_ROOT = "D:/NCKH/speech_detection_project"
CSV_PATH_DEV = os.path.join(DATA_ROOT, "FSD50K.metadata/collection/collection_dev.csv")
CSV_PATH_EVAL = os.path.join(DATA_ROOT, "FSD50K.metadata/collection/collection_eval.csv")

SAVE_PATH_DEV = os.path.join(DATA_ROOT, "FSD50K.metadata/collection/collection_dev_binary.csv")
SAVE_PATH_EVAL = os.path.join(DATA_ROOT, "FSD50K.metadata/collection/collection_eval_binary.csv")

SPEECH_LABELS = {
    "Speech", 
    "Speech_synthesizer", 
    "Human_voice", 
    "Male_speech_and_man_speaking", 
    "Male_singing", 
    "Female_speech_and_woman_speaking", 
    "Female_singing", 
    "Child_speech_and_kid_speaking", 
    "Child_singing", 
    "Conversation"
}

# ========================
# 2. Hàm đánh nhãn nhị phân với xử lý NaN
# ========================
def label_to_binary(label_str):
    # Kiểm tra nếu giá trị là NaN
    if pd.isna(label_str):
        return 0  # Không có nhãn → non-speech

    # Tách các nhãn bằng dấu phẩy và loại bỏ khoảng trắng
    labels = [label.strip() for label in label_str.split(",")]

    # Kiểm tra xem có nhãn thuộc nhóm speech không
    return int(any(label in SPEECH_LABELS for label in labels))

# ========================
# 3. Đọc và xử lý tập dev
# ========================
print("Đang xử lý tập dev...")
df_dev = pd.read_csv(CSV_PATH_DEV)
df_dev["label_binary"] = df_dev["labels"].apply(label_to_binary)
df_dev.to_csv(SAVE_PATH_DEV, index=False)
print(f"Đã lưu file nhãn nhị phân cho tập dev vào: {SAVE_PATH_DEV}")

# ========================
# 4. Đọc và xử lý tập eval
# ========================
print("Đang xử lý tập eval...")
df_eval = pd.read_csv(CSV_PATH_EVAL)
df_eval["label_binary"] = df_eval["labels"].apply(label_to_binary)
df_eval.to_csv(SAVE_PATH_EVAL, index=False)
print(f"Đã lưu file nhãn nhị phân cho tập eval vào: {SAVE_PATH_EVAL}")