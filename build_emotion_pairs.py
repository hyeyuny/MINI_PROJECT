# src/build_emotion_pairs.py
from pathlib import Path
import json
import pandas as pd

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMO_DIR = DATA_DIR / "018.감성대화"
OUTPUT_PATH = DATA_DIR / "training_pairs_emotion.tsv"

def emotion_code_to_tag(emotion_type: str) -> str:
    # 감정 코드를 모델용 토큰으로 변환
    if not emotion_type:
        return "<emo:E00>"
    return f"<emo:{emotion_type}>"

def situation_to_ctx_tag(situations) -> str:
    # 상황 코드 리스트를 5개 대분류 태그로 매핑
    if not situations:
        return "<ctx:etc>"

    s_codes = {str(s) for s in situations}

    # 우선순위에 따라 태그 반환
    if "S06" in s_codes or "S07" in s_codes: return "<ctx:career>"
    if "S04" in s_codes or "S05" in s_codes: return "<ctx:emotion>"
    if "S01" in s_codes or "S02" in s_codes or "S03" in s_codes: return "<ctx:daily>"
    if "S08" in s_codes or "S09" in s_codes or "S10" in s_codes or "S11" in s_codes: return "<ctx:relationship>"
    
    return "<ctx:etc>"

def find_json_files():
    # 폴더 내 Training/Validation JSON 파일 자동 검색
    if not EMO_DIR.exists():
        print(f"⚠️ 감성대화 폴더를 찾을 수 없습니다: {EMO_DIR}")
        return None, None

    train_candidates = sorted(EMO_DIR.rglob("*Training*.json"))
    valid_candidates = sorted(EMO_DIR.rglob("*Validation*.json"))

    train_json = train_candidates[0] if train_candidates else None
    valid_json = valid_candidates[0] if valid_candidates else None

    print("🔍 검색된 Training JSON:", train_json)
    print("🔍 검색된 Validation JSON:", valid_json)

    return train_json, valid_json

def extract_pairs_from_json(json_path: Path, split: str):
    # JSON 파일에서 (input, output) 쌍 추출
    print(f"[{split}] JSON 로드: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for obj in data:
        profile = obj.get("profile", {})
        emotion_info = profile.get("emotion", {})
        
        emotion_type = emotion_info.get("type")
        situations = emotion_info.get("situation", [])

        emo_tag = emotion_code_to_tag(emotion_type)
        ctx_tag = situation_to_ctx_tag(situations)

        talk = obj.get("talk", {})
        content = talk.get("content", {})

        # HS(사람) -> SS(시스템) 대화 쌍 추출 (1~3턴)
        for i in range(1, 4):
            hs_key = f"HS0{i}"
            ss_key = f"SS0{i}"

            src = str(content.get(hs_key, "") or "").strip()
            trg = str(content.get(ss_key, "") or "").strip()

            if not src or not trg:
                continue

            # 입력 포맷: <ctx> <emo> <style> 발화문
            input_text = f"{ctx_tag} {emo_tag} {STYLE_TAG_HEYO} {src}"
            output_text = trg

            pairs.append((input_text, output_text))

    print(f"[{split}] 추출 pair 수: {len(pairs)}")
    return pairs

def main():
    train_json, valid_json = find_json_files()
    all_pairs = []

    if train_json:
        all_pairs += extract_pairs_from_json(train_json, split="train")
    else:
        print("⚠️ Training JSON을 찾지 못했습니다.")

    if valid_json:
        all_pairs += extract_pairs_from_json(valid_json, split="valid")
    else:
        print("⚠️ Validation JSON을 찾지 못했습니다.")

    print(f"총 pair 수: {len(all_pairs)}")

    if not all_pairs:
        print("⚠️ 생성된 pair가 없습니다.")
        return

    # TSV 파일로 저장
    df = pd.DataFrame(all_pairs, columns=["input", "output"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"✅ 감성대화 기반 pair 저장 완료 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()