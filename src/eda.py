"""
資料分析：W12 簡報「資料分析」頁的所有數字和圖表。

用法：
    python src/eda.py              # 統計 + 出圖
    python src/eda.py --no-plots   # 只算統計（沒有 matplotlib 或出圖失敗時）

輸出：
    reports/eda_summary.txt
    reports/figures/eda_01_class_balance.png
    reports/figures/eda_02_images_per_patient.png
    reports/figures/eda_03_malignant_rate_by_site.png
    reports/figures/eda_04_malignant_rate_by_age.png
    reports/figures/eda_05_correlation_heatmap.png
"""

import argparse

import pandas as pd

from config import FIGURES_DIR, REPORTS_DIR, ensure_dirs
from features import load_data


def summarize(train: pd.DataFrame) -> tuple[str, dict]:
    per_patient = train.groupby("patient_id").size()
    site = train.groupby("anatom_site_general")["target"].agg(["count", "sum", "mean"])
    site["mean"] *= 100
    age = train.groupby(pd.cut(train["age_approx"], [0, 40, 50, 60, 70, 100]), observed=False)["target"].mean() * 100
    sex = train.groupby("sex")["target"].agg(["count", "sum", "mean"])
    missing = train.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)

    lines = [
        "===== 基本資訊 =====",
        f"圖片數：{len(train):,}",
        f"欄位數：{train.shape[1]}",
        f"惡性：{int(train.target.sum()):,}（{train.target.mean() * 100:.3f}%）",
        f"病人數：{train.patient_id.nunique():,}，其中有惡性病灶的病人：{int(train.groupby('patient_id').target.max().sum())}",
        f"每位病人圖片數：中位數 {per_patient.median():.0f}，最少 {per_patient.min()}，最多 {per_patient.max()}",
        "",
        "===== 各部位惡性比例（%）=====",
        site.round(3).to_string(),
        "",
        "===== 各年齡層惡性比例（%）=====",
        age.round(3).to_string(),
        "",
        "===== 性別 =====",
        sex.to_string(),
        "",
        "===== 缺失值比例 =====",
        missing.round(4).to_string() if len(missing) else "無",
    ]
    return "\n".join(lines), {"per_patient": per_patient, "site": site, "age": age}


def plot(train: pd.DataFrame, s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / name, dpi=150)
        plt.close(fig)
        print(f"  {FIGURES_DIR / name}")

    fig, ax = plt.subplots(figsize=(5, 4))
    train["target"].value_counts().sort_index().plot.bar(ax=ax, logy=True, color=["#8FB3B0", "#C4553E"])
    ax.set_xticklabels(["benign (0)", "malignant (1)"], rotation=0)
    ax.set_title("Class balance (log scale)")
    save(fig, "eda_01_class_balance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s["per_patient"].plot.hist(bins=60, ax=ax, color="#12343B")
    ax.set_xlabel("images per patient")
    ax.set_title("Images per patient")
    save(fig, "eda_02_images_per_patient.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s["site"]["mean"].sort_values().plot.barh(ax=ax, color="#C4553E")
    ax.set_xlabel("malignant rate (%)")
    ax.set_title("Malignant rate by anatomic site")
    save(fig, "eda_03_malignant_rate_by_site.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s["age"].plot.bar(ax=ax, color="#12343B", rot=0)
    ax.set_xlabel("age")
    ax.set_ylabel("malignant rate (%)")
    ax.set_title("Malignant rate by age")
    save(fig, "eda_04_malignant_rate_by_age.png")

    # 相關性熱力圖：跟 target 相關最高的 7 個數值特徵
    num = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c != "target"]
    top = train[num].corrwith(train["target"]).abs().sort_values(ascending=False).head(7).index.tolist()
    corr = train[top + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap (top 7 features vs target)")
    save(fig, "eda_05_correlation_heatmap.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    train, _ = load_data()
    text, stats = summarize(train)
    print(text)
    (REPORTS_DIR / "eda_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n已寫出 {REPORTS_DIR / 'eda_summary.txt'}")
    if not args.no_plots:
        print("出圖：")
        plot(train, stats)


if __name__ == "__main__":
    main()
