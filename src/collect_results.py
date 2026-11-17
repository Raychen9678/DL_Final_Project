"""
彙整全組實驗結果：把每個人從 Kaggle 下載的 exp_XX.json 放進 results/，執行

    python src/collect_results.py

產生 results/summary.md（依變因分組、和 BASE 比較的差值），也印在畫面上。
本機的表格模型結果（reports/cv_tabular.csv，E1、E2）若存在會一起列入。
"""

import json

import pandas as pd

from config import PROJECT_ROOT, REPORTS_DIR

RESULTS_DIR = PROJECT_ROOT / "results"
ORDER = ["全部", "A", "B", "C", "D", "E"]


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS_DIR.glob("exp_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        change = ", ".join(f"{k}={v}" for k, v in r.get("change", {}).items()) or "—"
        rows.append(dict(exp=r["exp"], member=r["member"], variable=r["variable"], change=change,
                         oof_pauc=r["oof_pauc"], minutes=r.get("minutes")))
    tab = REPORTS_DIR / "cv_tabular.csv"
    if tab.exists():
        t = pd.read_csv(tab)
        for exp, model in (("E1", "Logistic Regression"), ("E2", "MLP (256-128, BN, Dropout)")):
            hit = t[t.model == model]
            if len(hit) and exp not in {r["exp"] for r in rows}:
                rows.append(dict(exp=exp, member="E", variable="輸入資料（只用表格）", change=model,
                                 oof_pauc=float(hit.oof_pauc.iloc[0]), minutes=None))
    lgb_txt = REPORTS_DIR / "cv_lgbm.txt"
    if lgb_txt.exists():
        for line in lgb_txt.read_text(encoding="utf-8").splitlines():
            if line.startswith("OOF pAUC"):
                rows.append(dict(exp="LGBM", member="E", variable="對照組（只用表格）", change="LightGBM",
                                 oof_pauc=float(line.split("=")[1].split()[0]), minutes=None))
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load()
    if df.empty:
        print(f"{RESULTS_DIR} 裡還沒有 exp_*.json")
        return
    base = df.loc[df.exp == "BASE", "oof_pauc"]
    base = float(base.iloc[0]) if len(base) else None
    df["vs_base"] = df.oof_pauc - base if base is not None else float("nan")
    df["order"] = df.member.map({m: i for i, m in enumerate(ORDER)})
    df = df.sort_values(["order", "exp"]).drop(columns="order")

    lines = ["# 全組實驗結果", "",
             f"共同基準 BASE：{base:.4f}" if base is not None else "還沒有 BASE 結果", "",
             "| 編號 | 組員 | 變因 | 改了什麼 | OOF pAUC | 與 BASE 差 | 分鐘 |",
             "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        diff = "—" if r.exp == "BASE" or pd.isna(r.vs_base) else f"{r.vs_base:+.4f}"
        mins = "—" if r.minutes is None or pd.isna(r.minutes) else f"{r.minutes:.0f}"
        lines.append(f"| {r.exp} | {r.member} | {r.variable} | {r.change} | {r.oof_pauc:.4f} | {diff} | {mins} |")
    best = df[df.exp != "BASE"].sort_values("oof_pauc", ascending=False).groupby("member").head(1)
    lines += ["", "## 各變因最好的設定（W15 合併用）", ""]
    lines += [f"- {r.member}（{r.variable}）：{r.exp}，{r.oof_pauc:.4f}" for r in best.itertuples()]
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
