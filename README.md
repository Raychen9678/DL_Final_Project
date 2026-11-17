# ISIC 2024 皮膚癌偵測：深度學習期末專題

> 競賽：<https://www.kaggle.com/competitions/isic-2024-challenge>

給電腦看一張皮膚上痣或斑點的照片，加上病人的基本資料，讓它判斷這顆是不是皮膚癌。

## 先看哪一份？

| 你想 | 看這份 |
|---|---|
| **跑我負責的實驗** | 👉 [docs/01_操作說明.md](docs/01_操作說明.md)（照步驟點，不用裝東西） |
| 搞懂這個專題在做什麼 | [docs/00_專題說明.md](docs/00_專題說明.md) |
| 知道我負責哪個變因、什麼時候要交 | [docs/03_分工與時程.md](docs/03_分工與時程.md) |
| 看用了哪些模型、結果如何 | [docs/02_模型與實驗.md](docs/02_模型與實驗.md) |
| 寫書面報告 | [report/書面報告_草稿.md](report/書面報告_草稿.md) |

---

## 三十秒摘要

| 項目 | 內容 |
|---|---|
| 任務 | 二元分類：病灶是不是惡性 |
| 資料 | 401,059 張病灶圖 + metadata，惡性只有 **393 張（0.098%）**，1,042 位病人 |
| 評分 | **pAUC（TPR ≥ 80%）**，範圍 0–0.2，亂猜約 0.02 |
| 方法 | 表格 MLP、圖片 CNN / ResNet / EfficientNet / ViT、多模態融合；LightGBM 當對照組 |
| 繳交 | Code Competition，在 Kaggle Notebook 提交；比賽已結束，可 Late Submission |

## 目前結果（5 折 OOF pAUC）

| 模型 | pAUC |
|---|---|
| Logistic Regression | 0.103 |
| MLP | 0.159 |
| LightGBM（對照組） | **0.161** |
| 圖片模型、融合 | 待 Kaggle GPU |

詳細結果與消融實驗 → [docs/02_模型與實驗.md](docs/02_模型與實驗.md)

## 快速開始（組員）

1. 下載 [`notebooks/isic2024_image_models.ipynb`](notebooks/isic2024_image_models.ipynb)
2. Kaggle 比賽頁 → **Code → New Notebook → File → Import Notebook**
3. 右側設定：**GPU**、**Internet On**
4. 把 `EXP = 'BASE'` 改成你的編號（[編號表](docs/03_分工與時程.md#實驗編號一覽)）→ **Save Version**

詳細步驟與出問題怎麼辦 → [docs/01_操作說明.md](docs/01_操作說明.md)

## 資料夾結構

```
├── README.md
├── docs/                         說明文件（先看這裡）
├── notebooks/
│   ├── isic2024_baseline.ipynb       metadata baseline + EDA 圖
│   ├── isic2024_image_models.ipynb   圖片實驗（改 EXP 跑不同設定）
│   ├── isic2024_multimodal.ipynb     多模態網路（E4）
│   └── isic2024_submission.ipynb     融合、最終提交（E3）
├── src/
│   ├── config.py           路徑與參數
│   ├── download_data.py    下載資料
│   ├── eda.py              資料分析
│   ├── features.py         特徵、共用的 5 折切分
│   ├── metric.py           官方 pAUC
│   ├── train_lgbm.py       LightGBM
│   ├── train_mlp.py        LR、MLP、消融實驗
│   └── collect_results.py  彙整全組實驗結果
├── figures/                實驗結果圖
├── results/                各人的實驗結果 json、全組比較表 summary.md
├── report/                 書面報告草稿
└── data/、models/、reports/  執行後產生（不上傳）
```

## 資料來源

- SLICE-3D 資料集，DOI [10.34970/2024-slice-3d](https://doi.org/10.34970/2024-slice-3d)，CC-BY-NC 4.0
- 資料集論文：[Scientific Data, 10.1038/s41597-024-03743-w](https://doi.org/10.1038/s41597-024-03743-w)
