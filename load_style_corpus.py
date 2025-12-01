# src/load_style_corpus.py
from pathlib import Path
from typing import List, Tuple

def load_style_corpus(ban_path: str, yo_path: str) -> List[Tuple[str, str]]:
    ban_file = Path(ban_path)
    yo_file = Path(yo_path)

    if not ban_file.exists() or not yo_file.exists():
        print(f"병렬 파일 없음: {ban_file} 또는 {yo_file}")
        return []

    pairs: List[Tuple[str, str]] = []

    with ban_file.open("r", encoding="utf-8") as f_ban, yo_file.open(
        "r", encoding="utf-8"
    ) as f_yo:
        for ban_line, yo_line in zip(f_ban, f_yo):
            ban = ban_line.strip()
            yo = yo_line.strip()

            # 빈 줄 무시
            if not ban or not yo:
                continue

            pairs.append((ban, yo))

    print(f"스타일 병렬 데이터 샘플 수: {len(pairs)}")
    
    # 샘플 5개 출력
    for i, (src, trg) in enumerate(pairs[:5], start=1):
        print(f"[{i}] SRC: {src}")
        print(f"    TRG: {trg}")

    return pairs

if __name__ == "__main__":
    print("이 모듈은 run.py에서 import 해서 사용하세요.")