import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ========================
# 1. Thiết lập cấu hình
# ========================
N_MFCC = 13
SAMPLE_RATE = 16000
MAX_LEN = 100

DATA_ROOT = "D:/NCKH/speech_detection_project"
AUDIO_DIR_DEV = os.path.join(DATA_ROOT, "data/FSD50K.dev_audio")
AUDIO_DIR_EVAL = os.path.join(DATA_ROOT, "data/FSD50K.eval_audio")

CSV_PATH_DEV = os.path.join(DATA_ROOT, "FSD50K.metadata/collection/collection_dev_binary.csv")
CSV_PATH_EVAL = os.path.join(DATA_ROOT, "FSD50K.metadata/collection/collection_eval_binary.csv")

SAVE_PATH_DEV_X = os.path.join(DATA_ROOT, "features/rf/X_dev.npy")
SAVE_PATH_DEV_Y = os.path.join(DATA_ROOT, "features/rf/y_dev.npy")
SAVE_PATH_EVAL_X = os.path.join(DATA_ROOT, "features/rf/X_eval.npy")
SAVE_PATH_EVAL_Y = os.path.join(DATA_ROOT, "features/rf/y_eval.npy")

# ========================
# 2. Hàm trích xuất MFCC
# ========================
def extract_mfcc_features(audio_dir, df):
    X = []
    y = []

    for fname, label in tqdm(zip(df["fname"], df["label_binary"]), total=len(df)):
        file_path = os.path.join(audio_dir, str(fname) + ".wav")
        try:
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
            mfcc = mfcc.T

            if mfcc.shape[0] > MAX_LEN:
                mfcc = mfcc[:MAX_LEN]
            else:
                pad_width = MAX_LEN - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

            X.append(mfcc.flatten())  # Flatten để dùng với RF
            y.append(label)
        except Exception as e:
            print(f"Lỗi khi xử lý {file_path}: {e}")

    return np.array(X), np.array(y)

# ========================
# 3. Xử lý tập dev
# ========================
print("Đang xử lý tập dev...")
df_dev = pd.read_csv(CSV_PATH_DEV)
X_dev, y_dev = extract_mfcc_features(AUDIO_DIR_DEV, df_dev)
np.save(SAVE_PATH_DEV_X, X_dev)
np.save(SAVE_PATH_DEV_Y, y_dev)
print("Đã lưu đặc trưng tập dev.")

# ========================
# 4. Xử lý tập eval
# ========================
print("Đang xử lý tập eval...")
df_eval = pd.read_csv(CSV_PATH_EVAL)
X_eval, y_eval = extract_mfcc_features(AUDIO_DIR_EVAL, df_eval)
np.save(SAVE_PATH_EVAL_X, X_eval)
np.save(SAVE_PATH_EVAL_Y, y_eval)
print("Đã lưu đặc trưng tập eval.")
