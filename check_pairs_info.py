# src/check_pairs_info.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
path = BASE_DIR / "data" / "training_pairs_v3.tsv"

print("파일 경로:", path)

if not path.exists():
    print("❌ training_pairs_v3.tsv 파일을 찾을 수 없습니다.")
    raise SystemExit()

# TSV 읽기
df = pd.read_csv(path, sep="\t", header=None, names=["input", "output"])

print("총 pair 수:", len(df))

# 스타일 태그 개수 확인
ban_cnt = df["input"].astype(str).str.contains("<style:반말체>").sum()
yo_cnt  = df["input"].astype(str).str.contains("<style:해요체>").sum()
sho_cnt = df["input"].astype(str).str.contains("<style:합쇼체>").sum()

print("반말체 개수 :", ban_cnt)
print("해요체 개수 :", yo_cnt)
print("합쇼체 개수 :", sho_cnt)

print("\n=== 앞 5개 샘플 ===")
print(df.head(5).to_string(index=False))