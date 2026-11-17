"""
表格資料的深度學習模型（PyTorch）。

    python src/train_mlp.py               # Logistic Regression + MLP，5 折 CV
    python src/train_mlp.py --ablation    # 另外跑優化器比較、Dropout/BatchNorm 消融實驗

對應上課內容：
- Logistic Regression = 沒有隱藏層的網路（單層感知器 + sigmoid）           W1
- MLP：Linear → BatchNorm → ReLU → Dropout                                 W1、W6
- 損失函數：BCEWithLogitsLoss（sigmoid + binary cross-entropy）            W4
- 優化器：SGD+Momentum / Adam / AdamW                                       W3、W6
- mini-batch、DataLoader、Early Stopping、model.eval()                      W3、W4、W6

不平衡處理：每個 epoch 用全部惡性 + 重新抽樣的良性（惡性的 NEG_RATIO 倍）。
Early stopping：每個 epoch 在驗證 fold 上算 pAUC，連續 PATIENCE 次沒進步就停，用最好的權重。

輸出：
    reports/cv_tabular.csv                    各模型每折與整體 pAUC
    reports/oof_mlp.csv、reports/oof_lr.csv   OOF 預測
    reports/history_*.csv                     每個 epoch 的 loss 與 pAUC
    reports/figures/05_*.png ~ 07_*.png       消融實驗曲線、ROC 比較
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import FIGURES_DIR, REPORTS_DIR, SEED, ensure_dirs
from features import build_features, load_data, make_folds
from metric import pauc_above_tpr

NEG_RATIO = 20
BATCH_SIZE = 256
MAX_EPOCHS = 40
PATIENCE = 6


# ---------------------------------------------------------------- 資料前處理
def to_matrix(train: pd.DataFrame, features: list[str]):
    """數值欄位保留、類別欄位轉 one-hot。回傳 (X, 數值欄位數)。"""
    cat_cols = [c for c in features if isinstance(train[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in features if c not in cat_cols]
    num = train[num_cols].astype("float32").to_numpy()
    num[~np.isfinite(num)] = np.nan
    onehot = pd.get_dummies(train[cat_cols], dummy_na=True).astype("float32").to_numpy()
    return np.hstack([num, onehot]), len(num_cols)


def standardize(X: np.ndarray, n_num: int, tr: np.ndarray) -> np.ndarray:
    """只用訓練 fold 的統計量：缺失值補中位數、標準化、截斷極端值。"""
    X = X.copy()
    num = X[:, :n_num]
    med = np.nanmedian(num[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(np.isnan(num))
    num[idx] = np.take(med, idx[1])
    mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-6
    X[:, :n_num] = np.clip((num - mu) / sd, -5, 5)
    return X


# ---------------------------------------------------------------- 模型
class TabularNet(nn.Module):
    """hidden=[] 就是 Logistic Regression；hidden=[256, 128] 是兩層 MLP。"""

    def __init__(self, n_in: int, hidden: list[int], dropout: float = 0.3, batchnorm: bool = True):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_optimizer(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    raise ValueError(name)


@torch.no_grad()
def predict(model, X: torch.Tensor) -> np.ndarray:
    model.eval()  # BatchNorm 用訓練時累積的統計量、Dropout 關閉
    return torch.sigmoid(torch.cat([model(X[i:i + 8192]) for i in range(0, len(X), 8192)])).numpy()


def train_fold(X, y, tr, va, cfg, fold, early_stop=True):
    torch.manual_seed(SEED + fold)
    rng = np.random.default_rng(SEED + fold)
    Xt = torch.from_numpy(X)
    pos = tr[y[tr] == 1]
    neg = tr[y[tr] == 0]
    model = TabularNet(X.shape[1], cfg["hidden"], cfg["dropout"], cfg["batchnorm"])
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(cfg["epochs"]):
        idx = np.concatenate([pos, rng.choice(neg, len(pos) * NEG_RATIO, replace=False)])
        rng.shuffle(idx)
        model.train()
        tot = 0.0
        for i in range(0, len(idx), BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]
            if len(b) < 2:
                continue
            xb, yb = Xt[b], torch.from_numpy(y[b])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        p_va = predict(model, Xt[va])
        score = pauc_above_tpr(y[va], p_va)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(p_va).clamp(1e-7, 1 - 1e-7),
                                                            torch.from_numpy(y[va])))
        hist.append(dict(fold=fold, epoch=ep, train_loss=tot / len(idx), val_loss=val_loss, val_pauc=score))
        if score > best:
            best, wait = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if early_stop and wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return predict(model, Xt[va]), hist


def run_cv(X, y, folds, cfg, name, early_stop=True, verbose=True):
    oof = np.zeros(len(y), dtype="float32")
    hist, scores = [], []
    t0 = time.time()
    for fold, (tr, va) in enumerate(folds):
        Xf = standardize(X, cfg["n_num"], tr)
        oof[va], h = train_fold(Xf, y, tr, va, cfg, fold, early_stop)
        hist += h
        scores.append(pauc_above_tpr(y[va], oof[va]))
    total = pauc_above_tpr(y, oof)
    if verbose:
        print(f"{name:32s} OOF pAUC = {total:.4f}  folds = {' '.join(f'{s:.3f}' for s in scores)}  ({time.time() - t0:.0f}s)")
    return oof, pd.DataFrame(hist), total, scores


# ---------------------------------------------------------------- 圖
def plot_curves(hists: dict, path, title, key="val_pauc", fold=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in hists.items():
        d = h[h.fold == fold]
        ax.plot(d.epoch, d[key], label=name, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel({"val_pauc": "validation pAUC", "train_loss": "train loss"}[key])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  圖：{path}")


def plot_overfit(h: pd.DataFrame, path, fold=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = h[h.fold == fold]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(d.epoch, d.train_loss, color="#12343B", lw=2, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot(d.epoch, d.val_pauc, color="#C4553E", lw=2, label="validation pAUC")
    ax2.set_ylabel("validation pAUC")
    best = d.val_pauc.idxmax()
    ax2.axvline(d.loc[best, "epoch"], color="gray", ls="--", lw=1)
    ax1.set_title("MLP without regularization: train loss keeps falling, val pAUC peaks early")
    fig.legend(loc="center right", bbox_to_anchor=(0.86, 0.45), framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  圖：{path}")


def plot_roc(curves: dict, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (pAUC {pauc_above_tpr(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.axhspan(0.8, 1.0, color="#C4553E", alpha=0.08)
    ax.text(0.55, 0.83, "scored region: TPR >= 0.8", color="#C4553E")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (5-fold OOF)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  圖：{path}")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="另外跑優化器與正則化消融實驗")
    args = ap.parse_args()

    ensure_dirs()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train, _ = load_data()
    train, _, features = build_features(train)
    X, n_num = to_matrix(train, features)
    y = train["target"].to_numpy("float32")
    folds = make_folds(train)
    print(f"{X.shape[0]:,} 筆、{X.shape[1]} 維輸入（{n_num} 數值 + {X.shape[1] - n_num} one-hot）")

    base = dict(n_num=n_num, hidden=[256, 128], dropout=0.3, batchnorm=True,
                optimizer="adamw", lr=1e-3, epochs=MAX_EPOCHS)
    lr_cfg = {**base, "hidden": [], "dropout": 0.0, "batchnorm": False}

    rows, hists, oofs = [], {}, {}
    for name, cfg, key in [("Logistic Regression", lr_cfg, "lr"), ("MLP (256-128, BN, Dropout)", base, "mlp")]:
        oof, h, total, scores = run_cv(X, y, folds, cfg, name)
        rows.append(dict(model=name, oof_pauc=total, **{f"fold{i}": s for i, s in enumerate(scores)}))
        pd.DataFrame({"isic_id": train.isic_id, "target": y, "oof": oof}).to_csv(REPORTS_DIR / f"oof_{key}.csv", index=False)
        h.to_csv(REPORTS_DIR / f"history_{key}.csv", index=False)
        oofs[name] = oof

    if args.ablation:
        print("\n===== 消融實驗：優化器（MLP，固定 40 epochs，不 early stop）=====")
        for opt, lr in [("sgd", 1e-2), ("adam", 1e-3), ("adamw", 1e-3)]:
            name = {"sgd": "SGD + Momentum", "adam": "Adam", "adamw": "AdamW"}[opt]
            _, h, total, scores = run_cv(X, y, folds, {**base, "optimizer": opt, "lr": lr}, f"MLP / {name}", early_stop=False)
            rows.append(dict(model=f"MLP / {name}（40 epochs）", oof_pauc=total, **{f"fold{i}": s for i, s in enumerate(scores)}))
            hists[name] = h
        plot_curves(hists, FIGURES_DIR / "05_optimizer_comparison.png", "Optimizer comparison (MLP, fold 0)")

        print("\n===== 消融實驗：正則化（MLP，AdamW，固定 40 epochs）=====")
        reg_h = {}
        for name, bn, dp in [("no BN, no Dropout", False, 0.0), ("Dropout only", False, 0.3),
                             ("BN only", True, 0.0), ("BN + Dropout", True, 0.3)]:
            _, h, total, scores = run_cv(X, y, folds, {**base, "batchnorm": bn, "dropout": dp}, f"MLP / {name}", early_stop=False)
            rows.append(dict(model=f"MLP / {name}（40 epochs）", oof_pauc=total, **{f"fold{i}": s for i, s in enumerate(scores)}))
            reg_h[name] = h
        plot_curves(reg_h, FIGURES_DIR / "06_regularization_comparison.png", "BatchNorm / Dropout ablation (MLP, fold 0)")
        plot_overfit(reg_h["no BN, no Dropout"], FIGURES_DIR / "07_overfitting_curve.png")

    lgb_oof = REPORTS_DIR / "oof_lgbm.csv"
    if lgb_oof.exists():
        oofs["LightGBM"] = pd.read_csv(lgb_oof)["oof"].to_numpy()
    plot_roc(oofs, y, FIGURES_DIR / "08_roc_comparison.png")

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "cv_tabular.csv", index=False, float_format="%.4f")
    print(f"\n結果表：{REPORTS_DIR / 'cv_tabular.csv'}")


if __name__ == "__main__":
    main()
