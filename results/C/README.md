# 組員 C：正則化與資料增強

用 `notebooks/isic2024_image_models_C_all.ipynb` 一次跑完 `BASE → C1 → C2 → C3 → C4`（Kaggle GPU T4 x2，共約 3 小時）。

| 編號 | 改了什麼 | OOF pAUC | 與 BASE 差 | 5 折中贏 BASE |
|---|---|---|---|---|
| BASE | 基本增強 + Dropout 0.3 | 0.1379 | — | — |
| C1 | 不增強 | 0.1051 | −0.0328 | 1/5 |
| C2 | 強增強 | 0.1388 | +0.0009 | 3/5 |
| C3 | 不用 Dropout | 0.1407 | +0.0028 | 3/5 |
| C4 | 強增強 + 不用 Dropout | 0.1346 | −0.0033 | 3/5 |

**結論**
- 資料增強是必要的：拿掉後明顯 overfitting（訓練 loss 降到 0.03，驗證 pAUC 第 1–2 epoch 就到頂後下滑）。
- 強增強、Dropout 的差距（±0.003）小於 fold 間波動（約 ±0.015），視為沒有差別。
- W15 合併建議：保留 `aug='basic'`；`dropout` 用 0 或 0.3 皆可（C3 分數最高、各折最穩）。

**檔案**
- `history_*.csv`：每個 fold、每個 epoch 的 train_loss / val_loss / val_pAUC
- `summary_C.csv`：五個實驗總表
- 圖：`figures/C/C_pauc_comparison.png`、`figures/C/C_overfitting_BASE_vs_C1.png`、各實驗訓練曲線 `figures/C/curves_*.png`
- 講稿：`report/W14_簡報講稿_C.md`
