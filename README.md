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
| EfficientNet-B0（圖片） | 程式已完成，待在 Kaggle GPU 上訓練 |
| LightGBM + CNN 融合 | 程式已完成，待 CNN 結果 |

- 驗證集按 `patient_id` 切（StratifiedGroupKFold），同一位病人的圖不會同時出現在訓練和驗證
- 病人內標準化：每顆病灶的特徵減掉同一位病人的平均，衡量「跟這個人其他痣比有多特別」
- 正樣本太少，`scale_pos_weight` 會嚴重過擬合；模型要小、要正則化

## 怎麼跑

**在 Kaggle 上**（比賽頁 → Code → New Notebook → File → Import Notebook）：

| 順序 | Notebook | 設定 | 用途 |
|---|---|---|---|
| 1 | `isic2024_baseline.ipynb` | CPU，Internet 關 | metadata baseline，可直接提交 |
| 2 | `isic2024_cnn_train.ipynb` | **GPU，Internet 開** | 訓練 CNN，輸出權重和 `oof_cnn.csv`；跑完按 Save Version |
| 3 | `isic2024_submission.ipynb` | **GPU，Internet 關** | Add Input 加入第 2 個 notebook 的輸出，融合後提交 |

第 3 個 notebook 開頭的 `CNN_DIR` 要改成第 2 個 notebook 輸出在右側 Input 顯示的路徑。

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
notebooks/isic2024_baseline.ipynb   metadata baseline
notebooks/isic2024_cnn_train.ipynb  EfficientNet-B0 訓練（產生 CNN 特徵）
notebooks/isic2024_submission.ipynb LightGBM + CNN 融合，最終提交
report/書面報告_草稿.md               W17 書面報告草稿
src/config.py                       路徑與參數
src/download_data.py                下載資料
src/eda.py                          資料分析
src/features.py                     特徵工程
src/metric.py                       官方 pAUC 評分
src/train_lgbm.py                   LightGBM + 5 折 CV（reports/oof_cnn.csv 存在時自動加入 CNN 特徵）
```

## 分工

一人負責一個流程，從程式、簡報到書面報告都包含自己那段。

| 成員 | 流程 | 相關程式 |
|---|---|---|
| [組員 A] | ① 資料分析 | `src/eda.py` |
| [組員 B] | ② 驗證與評分 | `src/metric.py` |
| [組員 C] | ③ 表格模型 | `src/features.py`、`src/train_lgbm.py`、`notebooks/isic2024_baseline.ipynb` |
| [組員 D] | ④ 圖片模型 | `notebooks/isic2024_cnn_train.ipynb` |
| [組員 E] | ⑤ 融合與提交 | `notebooks/isic2024_submission.ipynb` |

| 週次 | 目標 |
|---|---|
| W12 | metadata baseline（CV pAUC 0.161）、5 頁進度簡報 |
| W14 | 圖片 CNN baseline、10 頁進度簡報 |
| W15 | CNN＋LightGBM 融合、繳交程式 |
| W15–16 | 成果報告與口試 |
| W17 | 書面報告、最終程式碼 |

## 資料來源

- SLICE-3D 資料集，DOI [10.34970/2024-slice-3d](https://doi.org/10.34970/2024-slice-3d)，CC-BY-NC 4.0
- 資料集論文：[Scientific Data, 10.1038/s41597-024-03743-w](https://doi.org/10.1038/s41597-024-03743-w)
