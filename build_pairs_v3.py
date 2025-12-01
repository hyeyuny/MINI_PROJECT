# src/build_pairs_v3.py
"""
한국어 어체 변환 병렬 데이터를 기반으로
감정 + 스타일 변환 학습 페어를 생성하는 v3 스크립트 (업그레이드 버전)
+ 감성 대화 말뭉치 추가!
"""

import os
import json # JSON 읽기를 위해 추가
from pathlib import Path
import pandas as pd

# ========================================
# 기본 설정
# ========================================
BASE_DIR = Path(__file__).resolve().parent.parent  # C:/Users/User/NLP
DATA_DIR = BASE_DIR / "data" 
OUTPUT_PATH = BASE_DIR / "data" / "training_pairs_v3.tsv"

# 어체 변환 데이터 경로
STYLE_CORPUS_DIR = DATA_DIR / "한국어 어체 변환 코퍼스"

# [!!!] 감성 대화 말뭉치 경로 (학생분 폴더 구조에 맞게 수정!)
# (F:\018.감성대화\Training_221115_add\...)
# 예: r"F:\NLP (3)\NLP\data\Training_221115_add\...\Training.json"
EMOTION_CORPUS_PATH = Path(r"F:\NLP (3)\NLP\data\018.감성대화\Training_221115_add\라벨링데이터\감성대화말뭉치(최종데이터)_Training\감성대화말뭉치(최종데이터)_Training.json")

DEFAULT_EMOTION = "E18"  # 어체 변환 데이터용 기본 감정
STYLE_TAGS = {
    "ban": "반말체",
    "yo": "해요체",
    "sho": "합쇼체",
}

# ========================================
# 병렬 파일 로더 (어체 변환용)
# ========================================
def load_parallel_file(src_path: Path, trg_path: Path):
    if not src_path.exists() or not trg_path.exists():
        print(f"[경고] 파일 없음 → skip : {src_path.name}, {trg_path.name}")
        return [], []

    with open(src_path, "r", encoding="utf-8") as f1, \
         open(trg_path, "r", encoding="utf-8") as f2:

        src_lines = [x.strip() for x in f1.readlines()]
        trg_lines = [x.strip() for x in f2.readlines()]

    if len(src_lines) != len(trg_lines):
        print(f"[경고] 길이 불일치: {src_path.name}, {trg_path.name}")
        n = min(len(src_lines), len(trg_lines))
        src_lines = src_lines[:n]
        trg_lines = trg_lines[:n]

    return src_lines, trg_lines

# ========================================
# pair 생성 함수
# ========================================
def make_pairs(src_list, trg_list, style_tag, emotion_tag):
    pairs = []
    for s, t in zip(src_list, trg_list):
        if not s.strip() or not t.strip():
            continue

        # 최종 input/output 포맷
        inp = f"<emo:{emotion_tag}> <style:{style_tag}> {s}"
        out = t
        pairs.append([inp, out])

    return pairs

# ========================================
# 1. 스타일(어체) 변환 데이터 생성
# ========================================
def build_style_pairs():
    all_pairs = []
    print("--- [1] 어체 변환 데이터 로딩 중... ---")

    # 1) KETI
    keti_dir = STYLE_CORPUS_DIR / "1-8_한국어_어체_변환_코퍼스" / "KETI 일상오피스 대화 1,254 문장"
    files = {
        ("ban", "yo"): ("keti_ban_words.txt", "keti_yo_words.txt"),
        ("ban", "sho"): ("keti_ban_words.txt", "keti_sho_words.txt"),
        ("yo", "sho"): ("keti_yo_words.txt", "keti_sho_words.txt"),
    }
    for (src_key, trg_key), (src_file, trg_file) in files.items():
        src_path = keti_dir / src_file
        trg_path = keti_dir / trg_file
        src_list, trg_list = load_parallel_file(src_path, trg_path)
        all_pairs += make_pairs(src_list, trg_list, STYLE_TAGS[trg_key], DEFAULT_EMOTION)

    # 2) 셀바스 임신육아
    med_dir = STYLE_CORPUS_DIR / "1-8_한국어_어체_변환_코퍼스" / "셀바스 임신육아 대화 1,940 문장"
    med_files = {
        ("ban", "yo"): ("medical_sho_words.txt", "medical_yo_words.txt"), 
        ("yo", "sho"): ("medical_yo_words.txt", "medical_sho_words.txt"),
    }
    for (src_key, trg_key), (src_file, trg_file) in med_files.items():
        src_path = med_dir / src_file
        trg_path = med_dir / trg_file
        src_list, trg_list = load_parallel_file(src_path, trg_path)
        all_pairs += make_pairs(src_list, trg_list, STYLE_TAGS[trg_key], DEFAULT_EMOTION)

    # 3) AI-Hub 수동 태깅
    manual_dir = STYLE_CORPUS_DIR / "수동태깅_병렬데이터" / "수동태깅 병렬데이터"
    manual_sets = [
        ("train_ext.ban.txt", "train_ext.yo.txt", "해요체"),
        ("train_ext.ban.txt", "train_ext.sho.txt", "합쇼체"),
        ("train_ext.yo.txt", "train_ext.ban.txt", "반말체"),
    ]
    for src_file, trg_file, style_tag in manual_sets:
        src_path = manual_dir / src_file
        trg_path = manual_dir / trg_file
        src_list, trg_list = load_parallel_file(src_path, trg_path)
        all_pairs += make_pairs(src_list, trg_list, style_tag, DEFAULT_EMOTION)

    # 4) 합성 병렬 데이터
    synth_dir = STYLE_CORPUS_DIR / "합성_병렬데이터" / "합성 병렬데이터"
    synth_sets = [
        ("반말체-해요체 ban2yo", "opensubtitles.syn.bpe.ban.txt", "opensubtitles.syn.bpe.yo.txt", "해요체"),
        ("해요체-반말체 yo2ban", "opensubtitles.syn.bpe.yo.txt", "opensubtitles.syn.bpe.ban.txt", "반말체"),
        ("합쇼체-해요체 sho2yo", "opensubtitles.syn.bpe.sho.txt", "opensubtitles.syn.bpe.yo.txt", "해요체"),
    ]
    for folder, src_f, trg_f, style_tag in synth_sets:
        path_dir = synth_dir / folder
        src_path = path_dir / src_f
        trg_path = path_dir / trg_f
        src_list, trg_list = load_parallel_file(src_path, trg_path)
        all_pairs += make_pairs(src_list, trg_list, style_tag, DEFAULT_EMOTION)

    return all_pairs

# ========================================
# [NEW] 2. 감정 데이터 생성 (Auto-Encoder 방식)
# ========================================
def get_style_from_text(text):
    """문장 끝을 보고 간단히 스타일 추정"""
    text = text.strip()
    if text.endswith('요') or text.endswith('요.'): return "해요체"
    elif text.endswith('다') or text.endswith('까') or text.endswith('다.') or text.endswith('까?'): return "합쇼체"
    else: return "반말체"

def build_emotion_pairs():
    emo_pairs = []
    print(f"--- [2] 감성 대화 데이터 로딩 중... ({EMOTION_CORPUS_PATH}) ---")
    
    if not EMOTION_CORPUS_PATH.exists():
        print(f"[경고] 감성 대화 파일 없음: {EMOTION_CORPUS_PATH}")
        return []

    try:
        with open(EMOTION_CORPUS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[오류] JSON 읽기 실패: {e}")
        return []

    for conv in data:
        try:
            # 감정 ID (E18, E25 등)
            emotion_id = conv['profile']['emotion']['type']
            
            # 문장 (HS01 등 사람 말만 추출)
            content = conv['talk']['content']
            for turn_key, sentence in content.items():
                if turn_key.startswith('HS') and sentence.strip():
                    # 스타일 추정
                    style_tag = get_style_from_text(sentence)
                    
                    # [입력] <emo:감정> <style:말투> 문장
                    # [출력] 문장 (자기 자신 생성)
                    inp = f"<emo:{emotion_id}> <style:{style_tag}> {sentence}"
                    out = sentence
                    emo_pairs.append([inp, out])
                    
        except KeyError:
            continue
            
    return emo_pairs

# ========================================
# main()
# ========================================
def main():
    print("=== 스타일 + 감정 학습 페어 생성 v3 (Upgrade) ===")
    
    # 1. 스타일 데이터 (기존)
    style_pairs = build_style_pairs()
    print(f"-> 스타일 Pair 수: {len(style_pairs):,}")

    # 2. 감정 데이터 (신규 추가)
    emotion_pairs = build_emotion_pairs()
    print(f"-> 감정 Pair 수: {len(emotion_pairs):,}")

    # 3. 데이터 합치기
    all_pairs = style_pairs + emotion_pairs
    print(f"===> 최종 Pair 수: {len(all_pairs):,}")

    # 저장
    df = pd.DataFrame(all_pairs, columns=["input", "output"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True) # 폴더 없으면 생성
    df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"저장 완료 → {OUTPUT_PATH}")

# ========================================
if __name__ == "__main__":
    main()