"""
資料分析：W12 簡報「資料分析」頁的所有數字和圖表。

用法：
    python src/eda.py              # 統計 + 出圖
    python src/eda.py --no-plots   # 只算統計（沒有 matplotlib 或出圖失敗時）

輸出：
    reports/eda_summary.txt
    reports/figures/01_class_balance.png
    reports/figures/02_images_per_patient.png
    reports/figures/03_malignant_rate_by_site.png
    reports/figures/04_malignant_rate_by_age.png
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
    save(fig, "01_class_balance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s["per_patient"].plot.hist(bins=60, ax=ax, color="#12343B")
    ax.set_xlabel("images per patient")
    ax.set_title("Images per patient")
    save(fig, "02_images_per_patient.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s["site"]["mean"].sort_values().plot.barh(ax=ax, color="#C4553E")
    ax.set_xlabel("malignant rate (%)")
    ax.set_title("Malignant rate by anatomic site")
    save(fig, "03_malignant_rate_by_site.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s["age"].plot.bar(ax=ax, color="#12343B", rot=0)
    ax.set_xlabel("age")
    ax.set_ylabel("malignant rate (%)")
    ax.set_title("Malignant rate by age")
    save(fig, "04_malignant_rate_by_age.png")


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
