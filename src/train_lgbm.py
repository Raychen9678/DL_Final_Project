"""
Metadata baseline：LightGBM + 按病人分組的 5-fold CV。

用法：
    python src/train_lgbm.py

輸出：
    reports/cv_lgbm.txt              每折與整體 OOF pAUC
    reports/oof_lgbm.csv             每筆訓練資料的 OOF 預測（之後做模型融合用）
    reports/feature_importance.csv   特徵重要度
    models/lgbm_fold{k}.txt          各折模型
    submission.csv                   有 test-metadata.csv 時才會產生
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (LGB_PARAMS, MODELS_DIR, N_FOLDS, NEG_RATIO, REPORTS_DIR, SEED,
                    SUBMISSION_CSV, ensure_dirs)
from features import build_features, load_data
from metric import pauc_above_tpr


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    train, test = load_data()
    train, test, features = build_features(train, test)
    print(f"train {train.shape}，惡性 {int(train.target.sum())} 筆，{len(features)} 個特徵")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    models, lines = [], []
    for fold, (tr, va) in enumerate(sgkf.split(train, train["target"], train["patient_id"])):
        d = train.iloc[tr]
        pos = d[d["target"] == 1]
        neg = d[d["target"] == 0].sample(len(pos) * NEG_RATIO, random_state=SEED)
        d = pd.concat([pos, neg])

        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(d[features], d["target"])
        oof[va] = m.predict_proba(train.iloc[va][features])[:, 1]
        # 不用 booster_.save_model：LightGBM 在 Windows 上寫不進含中文的路徑
        (MODELS_DIR / f"lgbm_fold{fold}.txt").write_text(m.booster_.model_to_string(), encoding="utf-8")
        models.append(m)

        line = f"fold {fold}: pAUC = {pauc_above_tpr(train.iloc[va]['target'], oof[va]):.4f}"
        print(line)
        lines.append(line)

    total = f"OOF pAUC = {pauc_above_tpr(train['target'], oof):.4f}  （上限 0.2）"
    print(total)
    lines.append(total)
    (REPORTS_DIR / "cv_lgbm.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.DataFrame({"isic_id": train["isic_id"], "target": train["target"], "oof": oof}).to_csv(
        REPORTS_DIR / "oof_lgbm.csv", index=False)
    imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=features)
    imp.sort_values(ascending=False).to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])

    if test is not None:
        pred = np.mean([m.predict_proba(test[features])[:, 1] for m in models], axis=0)
        pd.DataFrame({"isic_id": test["isic_id"], "target": pred}).to_csv(SUBMISSION_CSV, index=False)
        print(f"已寫出 {SUBMISSION_CSV}")
    else:
        print("沒有 test-metadata.csv，略過 submission（本機用 ISIC 官網資料時是正常的）")
    print(f"耗時 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
