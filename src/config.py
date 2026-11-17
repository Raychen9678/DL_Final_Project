"""
專案的所有路徑與超參數集中在這裡管理。
其他腳本一律 `from config import ...`，不要在程式裡寫死路徑。
"""

import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
    """Windows 預設 cp950，中文輸出導到檔案或管線時會變亂碼，這裡統一改成 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


_force_utf8_stdout()

# ---------------------------------------------------------------- 路徑
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 在 Kaggle Notebook 上跑時，資料已經掛在 /kaggle/input 底下，不用下載。
# 掛載路徑會變（例如多一層 competitions/），所以用搜尋的。
def _find_kaggle_data() -> Path | None:
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for depth in range(1, 4):
        hits = sorted(root.glob("/".join(["*"] * depth) + "/train-metadata.csv"),
                      key=lambda p: ("isic-2024-challenge" not in str(p), str(p)))
        if hits:
            return hits[0].parent
    return None


RAW_DIR = _find_kaggle_data() or PROJECT_ROOT / "data" / "raw"

TRAIN_META_CSV = RAW_DIR / "train-metadata.csv"
TEST_META_CSV = RAW_DIR / "test-metadata.csv"
TRAIN_HDF5 = RAW_DIR / "train-image.hdf5"
TEST_HDF5 = RAW_DIR / "test-image.hdf5"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSION_CSV = PROJECT_ROOT / "submission.csv"

# ---------------------------------------------------------------- 競賽常數
COMPETITION = "isic-2024-challenge"
MIN_TPR = 0.80          # 評分只算 TPR >= 80% 的 ROC 面積，分數範圍 0 ~ 0.2

# 不能當特徵的欄位：ID、來源、授權
DROP_COLS = {"isic_id", "patient_id", "target", "image_type", "attribution", "copyright_license"}

# 病人內標準化時不處理的欄位（年齡在同一人身上都一樣；x/y/z 是座標）
NO_PATIENT_NORM = {"age_approx", "tbp_lv_x", "tbp_lv_y", "tbp_lv_z"}

# ---------------------------------------------------------------- 驗證設定
SEED = 42
N_FOLDS = 5             # StratifiedGroupKFold，按 patient_id 分組

# ---------------------------------------------------------------- LightGBM 設定
# 實驗結論：正樣本只有 393 個，模型要小、要正則化；scale_pos_weight 反而會嚴重過擬合
NEG_RATIO = 50          # 每個 fold 的訓練資料，良性只抽惡性數量的 50 倍
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=5,
    n_estimators=200,
    random_state=SEED,
    verbose=-1,
)


def ensure_dirs() -> None:
    """建立所有輸出資料夾（已存在就略過）。"""
    for d in (PROJECT_ROOT / "data" / "raw", FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"RAW_DIR      = {RAW_DIR}")
    print(f"資料是否存在 = {TRAIN_META_CSV.exists()}")
