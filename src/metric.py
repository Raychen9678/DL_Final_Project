"""
競賽官方評分：pAUC above 80% TPR。

只算 ROC 曲線在 TPR >= 0.8 那一段的面積，分數範圍 0 ~ 0.2：
- 完美模型 = 0.2
- 隨機亂猜 ≈ 0.02

公式照 Kaggle 官方 metric 實作：把正負標籤和分數都反過來，
TPR >= 0.8 就變成 FPR <= 0.2，再用 sklearn 的 max_fpr 算部分 AUC，最後換回原本的尺度。
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from config import MIN_TPR


def pauc_above_tpr(y_true, y_pred, min_tpr: float = MIN_TPR) -> float:
    v_gt = np.abs(np.asarray(y_true) - 1)
    v_pred = 1.0 - np.asarray(y_pred, dtype=float)
    max_fpr = 1 - min_tpr
    scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # sklearn 回傳的是 McClish 校正後的值，換回原始面積
    return 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (scaled - 0.5)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100_000)
    print(f"完美預測：{pauc_above_tpr(y, y):.4f}  （應為 0.2）")
    print(f"隨機預測：{pauc_above_tpr(y, rng.random(len(y))):.4f}  （應約 0.02）")
