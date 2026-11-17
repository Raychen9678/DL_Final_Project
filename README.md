# ISIC 2024 皮膚癌偵測：深度學習期末專題

> 競賽：<https://www.kaggle.com/competitions/isic-2024-challenge>

給電腦看一張皮膚上痣或斑點的照片，加上病人的基本資料，讓它判斷這顆是不是皮膚癌。

## 先看哪一份？

第一次來，照 ① → ② → ③ 的順序看。

| 順序 | 你想 | 看這份 |
|---|---|---|
| ① | **搞懂這個專題在做什麼** | 👉 [docs/00_專題說明.md](docs/00_專題說明.md) |
| ② | **知道我負責哪個變因、什麼時候要交** | 👉 [docs/03_分工與時程.md](docs/03_分工與時程.md) |
| ③ | 看我自己的任務卡（要跑什麼、簡報哪頁、報告寫哪節） | [docs/04_任務卡.md](docs/04_任務卡.md) |
| | 跑我負責的實驗（照步驟點，不用裝東西） | [docs/01_操作說明.md](docs/01_操作說明.md) |
| | 看整體做法、用了哪些模型、結果如何 | [docs/02_模型與實驗.md](docs/02_模型與實驗.md) |
| | 寫書面報告 | [report/書面報告_草稿.md](report/書面報告_草稿.md) |
| | 看 W12 簡報 | [Canva 簡報](https://www.canva.com/design/DAHbJGLOTRo/V8vQRciYleZlanllEZBFSA/edit) |

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

**Kaggle 實際提交**（LightGBM）：private **0.149**、public 0.171，與本地 CV 0.161 接近 → 按病人切的驗證方式可靠。

詳細結果與消融實驗 → [docs/02_模型與實驗.md](docs/02_模型與實驗.md)

## 快速開始（組員）

1. 下載 [`notebooks/isic2024_image_models.ipynb`](notebooks/isic2024_image_models.ipynb)
2. Kaggle 比賽頁 → **Code → New Notebook → File → Import Notebook**
3. 右側設定：**GPU**、**Internet On**
4. 把 `EXP = 'BASE'` 改成你的編號（[編號表](docs/03_分工與時程.md#實驗編號一覽)）→ **Save Version**

詳細步驟與出問題怎麼辦 → [docs/01_操作說明.md](docs/01_操作說明.md)

每個人要做什麼 → [docs/04_任務卡.md](docs/04_任務卡.md)

## W12 繳交檢核表（11/25 前）

📎 **簡報（Canva）**：<https://www.canva.com/design/DAHbJGLOTRo/V8vQRciYleZlanllEZBFSA/edit>

老師要求：清晰易懂的簡報 **5 頁以上**（標題頁不算），一組一份。我們的簡報共 8 頁。

| 老師要求 | 簡報頁 | 內容 | 參考來源 | 負責人 |
|---|---|---|---|---|
| （標題頁） | P1 封面 | 題目、組號、組員 | — | E |
| 介紹競賽任務 | P2 任務與評分 | 二元分類、pAUC（TPR ≥ 80%）、Code Competition | 競賽頁 | B |
| 舉例說明資料的長相 | P3 資料的長相 | 3 張惡性 + 3 張良性病灶圖、metadata 範例 | ISIC 官方資料 | C |
| 資料分析：類別、變數分布 | P4 資料分析（一） | 類別長條圖、每人圖片數直方圖、各部位惡性比例 | 本組自寫 `src/eda.py` | D |
| 資料分析：目標與特徵的關係 | P5 資料分析（二） | 相關性熱力圖、中位數比較 | 本組自寫 `src/eda.py` | D |
| 預期的實作方法（**附來源**） | P6 方法 | 模型規劃、共同訓練設定 | **3 個 Kaggle 連結（已附）** | A |
| 初步實驗結果 | P7 初步結果 | LR 0.103 / MLP 0.159 / LightGBM 0.161、ROC 圖 | 本組實驗 | E |
| 分工模式 | P8 分工 | 變因分工表、Round 制時程 | — | E |

**繳交前最後確認**（E 負責）
- [ ] 封面組號、所有 `[組員 X]` 換成真名
- [x] P7 填入 Kaggle 分數（private 0.149 / public 0.171）
- [ ] 8 頁逐頁看過，沒有跑版
- [ ] 參考 Kaggle Code 的頁面都附上網址（目前只有 P6，已附）
- [ ] 簡報頁 **Share → Export → PowerPoint** 匯出後繳交

每人檢查自己負責的那頁內容，講稿在每頁的備註欄。

## 資料夾結構

```
├── README.md
├── docs/                         說明文件：00 專題說明、01 操作說明、02 模型與實驗、03 分工與時程、04 任務卡
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
