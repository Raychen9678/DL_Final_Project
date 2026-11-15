"""
下載訓練資料到 data/raw/。兩種來源：

    python src/download_data.py              # 預設：ISIC 官網，只抓 metadata（約 90 MB，幾分鐘）
    python src/download_data.py --kaggle     # Kaggle API，完整比賽資料含圖片（約 2.7 GB）

ISIC 官網和 Kaggle 的訓練資料是同一份（SLICE-3D，401,059 張）。
跑 LightGBM baseline 只需要 metadata，用預設就好；要訓練圖片模型才需要 --kaggle，
或直接在 Kaggle Notebook 上跑（不用下載）。

ISIC 官網的 metadata 包在 1.2 GB 的圖片 zip 裡，這裡用 HTTP Range 只抓 zip 裡的
metadata.csv，不用整包下載。
"""

import argparse
import io
import struct
import subprocess
import sys
import urllib.request
import zlib

import pandas as pd

from config import COMPETITION, RAW_DIR, ensure_dirs

ISIC_BASE = "https://isic-archive.s3.amazonaws.com/challenges/2024/"
ZIP_URL = ISIC_BASE + "ISIC_2024_Training_Input.zip"
GT_URL = ISIC_BASE + "ISIC_2024_Training_GroundTruth.csv"
MEMBER = "ISIC_2024_Training_Input/metadata.csv"


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {} if start is None else {"Range": f"bytes={start}-{'' if end is None else end}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        return r.read()


def _fetch_zip_member(url: str, name: str) -> bytes:
    """從遠端 zip 只下載一個檔案：讀結尾的 central directory 找位置，再 Range 抓那一段。"""
    tail = _get_tail(url, 65536)
    key = name.encode()
    i = tail.rfind(b"PK\x01\x02")
    while i != -1:
        n_len, e_len = struct.unpack("<HH", tail[i + 28:i + 32])
        if tail[i + 46:i + 46 + n_len] == key:
            break
        i = tail.rfind(b"PK\x01\x02", 0, i)
    else:
        raise RuntimeError(f"zip 裡找不到 {name}")

    method = struct.unpack("<H", tail[i + 10:i + 12])[0]
    csize, usize = struct.unpack("<II", tail[i + 20:i + 28])
    offset = struct.unpack("<I", tail[i + 42:i + 46])[0]
    extra = tail[i + 46 + n_len:i + 46 + n_len + e_len]
    j = 0
    while j < len(extra):  # zip64 擴充欄位
        hid, hlen = struct.unpack("<HH", extra[j:j + 4])
        if hid == 1:
            vals = list(struct.unpack(f"<{hlen // 8}Q", extra[j + 4:j + 4 + hlen]))
            if usize == 0xFFFFFFFF:
                usize = vals.pop(0)
            if csize == 0xFFFFFFFF:
                csize = vals.pop(0)
            if offset == 0xFFFFFFFF:
                offset = vals.pop(0)
        j += 4 + hlen

    local = _get(url, offset, offset + 29)
    ln, le = struct.unpack("<HH", local[26:30])
    start = offset + 30 + ln + le
    print(f"下載 {name}（壓縮後 {csize / 1e6:.0f} MB）…")
    data = _get(url, start, start + csize - 1)
    return zlib.decompress(data, -15) if method == 8 else data


def _get_tail(url: str, n: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=-{n}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def from_isic() -> None:
    meta = pd.read_csv(io.BytesIO(_fetch_zip_member(ZIP_URL, MEMBER)), low_memory=False)
    print("下載答案檔 …")
    gt = pd.read_csv(io.BytesIO(_get(GT_URL)))
    train = meta.merge(gt, on="isic_id")
    train["target"] = train.pop("malignant").astype(int)
    out = RAW_DIR / "train-metadata.csv"
    train.to_csv(out, index=False)
    print(f"完成：{out}  {train.shape}，惡性 {train.target.sum()} 筆")
    print("注意：官網沒有測試集，submission 要在 Kaggle Notebook 上產生。")


def from_kaggle() -> None:
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)]
    print("執行：", " ".join(cmd))
    print("（要先在比賽頁按同意規則，並設定好 Kaggle API token）")
    subprocess.run(cmd, check=True)
    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    print(f"下載完成，請解壓縮 {zip_path} 到 {RAW_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", help="改用 Kaggle API 下載完整資料")
    args = ap.parse_args()
    ensure_dirs()
    if RAW_DIR.as_posix().startswith("/kaggle"):
        sys.exit("在 Kaggle Notebook 上不用下載，資料已在 /kaggle/input")
    from_kaggle() if args.kaggle else from_isic()
