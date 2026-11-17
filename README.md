# ISIC 2024 皮膚癌偵測：深度學習期末專題

> 競賽：<https://www.kaggle.com/competitions/isic-2024-challenge>

給電腦看一張皮膚上痣或斑點的照片，加上病人的基本資料，讓它判斷這顆是不是皮膚癌。

## 先看哪一份？

| 你是 | 先看這份 |
|---|---|
| **第一次接觸這個專題** | 👉 [docs/00_專題說明.md](docs/00_專題說明.md)（白話版，10 分鐘） |
| 要在 Kaggle 或自己電腦上跑程式 | [docs/01_環境與執行.md](docs/01_環境與執行.md) |
| 想知道用了哪些模型、結果如何 | [docs/02_模型與實驗.md](docs/02_模型與實驗.md) |
| 想知道自己負責什麼、什麼時候要交 | [docs/03_分工與時程.md](docs/03_分工與時程.md) |
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

## 快速開始

```bash
pip install -r requirements.txt
python src/download_data.py   # 下載 metadata
python src/train_mlp.py       # LR + MLP
```

Kaggle 上怎麼跑、怎麼提交 → [docs/01_環境與執行.md](docs/01_環境與執行.md)

## 資料夾結構

```
├── README.md
├── docs/                         說明文件（先看這裡）
├── notebooks/
│   ├── isic2024_baseline.ipynb       metadata baseline + EDA 圖
│   ├── isic2024_image_models.ipynb   圖片模型（4 種架構）
│   ├── isic2024_multimodal.ipynb     多模態網路
│   └── isic2024_submission.ipynb     融合、最終提交
├── src/
│   ├── config.py           路徑與參數
│   ├── download_data.py    下載資料
│   ├── eda.py              資料分析
│   ├── features.py         特徵、共用的 5 折切分
│   ├── metric.py           官方 pAUC
│   ├── train_lgbm.py       LightGBM
│   └── train_mlp.py        LR、MLP、消融實驗
├── figures/                實驗結果圖
├── report/                 書面報告草稿
└── data/、models/、reports/  執行後產生（不上傳）
```

## 資料來源

- SLICE-3D 資料集，DOI [10.34970/2024-slice-3d](https://doi.org/10.34970/2024-slice-3d)，CC-BY-NC 4.0
- 資料集論文：[Scientific Data, 10.1038/s41597-024-03743-w](https://doi.org/10.1038/s41597-024-03743-w)
