# src/build_pairs_mix.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

STYLE_PATH = DATA_DIR / "training_pairs_v3.tsv"       # 기존 스타일 변환용
EMO_PATH   = DATA_DIR / "training_pairs_emotion.tsv"  # 방금 만든 감성대화 pair
OUT_MAIN   = DATA_DIR / "training_pairs_v3.tsv"        # 학습에서 그대로 사용 (덮어쓰기)
OUT_BACKUP = DATA_DIR / "training_pairs_mix.tsv"       # 참고용 백업

def load_pairs(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️ {name} 파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame(columns=["input", "output"])
    
    df = pd.read_csv(path, sep="\t", header=None, names=["input", "output"])
    df = df.dropna(subset=["input", "output"]).reset_index(drop=True)
    print(f"- {name} pair 수: {len(df)}")
    return df

def main():
    print("=== 스타일 + 감성대화 pair 믹스 ===")

    style_df = load_pairs(STYLE_PATH, "style(training_pairs_v3)")
    emo_df   = load_pairs(EMO_PATH,   "emotion(training_pairs_emotion)")

    if len(style_df) == 0 and len(emo_df) == 0:
        print("⚠️ 사용할 pair 가 없습니다. 파일 경로를 확인하세요.")
        return

    # 스타일 데이터 개수를 기준으로 감성대화 샘플링 개수 결정
    n_style = len(style_df)
    n_emo   = len(emo_df)

    if n_style == 0:
        # 스타일 데이터가 없으면 감성대화만 사용
        print("⚠️ style pair 가 없어서, emotion pair 만 사용합니다.")
        combined_df = emo_df
    else:
        # 감성대화는 스타일 개수만큼(또는 그보다 적으면 전체) 샘플링하여 비율 맞춤
        n_sample_emo = min(n_emo, n_style)
        if n_sample_emo < n_emo:
            emo_df_sampled = emo_df.sample(n=n_sample_emo, random_state=42)
            print(f"- 감성대화 pair {n_emo}개 중 {n_sample_emo}개만 샘플링하여 사용")
        else:
            emo_df_sampled = emo_df
            print(f"- 감성대화 pair 전체 {n_emo}개 사용")

        combined_df = pd.concat([style_df, emo_df_sampled], axis=0, ignore_index=True)

    # 전체 데이터 셔플
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"- 최종 믹스 pair 수: {len(combined_df)}")

    # 저장 (최종 학습용은 헤더 없이 저장)
    combined_df.to_csv(OUT_MAIN, sep="\t", header=False, index=False, encoding="utf-8")
    combined_df.to_csv(OUT_BACKUP, sep="\t", header=False, index=False, encoding="utf-8")

    print(f"✅ 최종 학습용 pair 저장 → {OUT_MAIN}")
    print(f"✅ 백업용 pair 저장      → {OUT_BACKUP}")

if __name__ == "__main__":
    main()