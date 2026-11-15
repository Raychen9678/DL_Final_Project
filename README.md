# ISIC 2024 皮膚癌偵測

深度學習期末專題。競賽：[ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)

## 題目

從 3D 全身攝影裁下的皮膚病灶小圖和病人資料，判斷每顆病灶是不是**惡性**，輸出機率。

| | |
|---|---|
| 任務 | 二元分類 |
| 評分 | **pAUC（TPR ≥ 80%）**：只算 ROC 曲線在高敏感度那段的面積，範圍 0–0.2，亂猜約 0.02 |
| 資料 | 401,059 張圖，惡性只有 **393 張（0.098%）**，1,042 位病人 |
| 繳交 | Code Competition，要在 Kaggle Notebook 上跑出 `submission.csv`；比賽已結束，可 Late Submission |

## 目前進度

| 版本 | 5 折 CV pAUC |
|---|---|
| LightGBM + 正樣本加權 | 0.031 |
| LightGBM 小模型 + 正則化 | 0.156 |
| **+ 病人內標準化特徵** | **0.161** |
| 圖片 CNN | 進行中 |

- 驗證集按 `patient_id` 切（StratifiedGroupKFold），同一位病人的圖不會同時出現在訓練和驗證
- 病人內標準化：每顆病灶的特徵減掉同一位病人的平均，衡量「跟這個人其他痣比有多特別」
- 正樣本太少，`scale_pos_weight` 會嚴重過擬合；模型要小、要正則化

## 怎麼跑

**在 Kaggle 上（提交用）**：比賽頁 → Code → New Notebook → File → Import Notebook，上傳 `notebooks/isic2024_baseline.ipynb`，Run All。提交前在 Settings 關掉 Internet，再 Save Version → Submit。

**在自己電腦上（做實驗）**：

```bash
pip install -r requirements.txt
python src/download_data.py    # 從 ISIC 官網下載 metadata（約 90 MB，不用 Kaggle 帳號）
python src/eda.py              # 資料分析 → reports/
python src/train_lgbm.py       # 5 折 CV，約 1 分鐘
```

要圖片的話用 `python src/download_data.py --kaggle`（約 2.7 GB，需要 Kaggle API token）。

## 檔案

```
notebooks/isic2024_baseline.ipynb   Kaggle 提交用的完整流程
src/config.py                       路徑與參數
src/download_data.py                下載資料
src/eda.py                          資料分析
src/features.py                     特徵工程
src/metric.py                       官方 pAUC 評分
src/train_lgbm.py                   LightGBM + 5 折 CV
```

## 分工

| 成員 | 負責 |
|---|---|
| [姓名] | 資料分析、簡報 |
| [姓名] | 圖片模型 |
| [姓名] | Metadata 模型、融合 |
| [姓名] | 驗證流程、Kaggle 提交 |

## 資料來源

- SLICE-3D 資料集，DOI [10.34970/2024-slice-3d](https://doi.org/10.34970/2024-slice-3d)，CC-BY-NC 4.0
- 資料集論文：[Scientific Data, 10.1038/s41597-024-03743-w](https://doi.org/10.1038/s41597-024-03743-w)
