# src/infer_style_seq2seq.py
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import re

# 같은 폴더에 있는 train_style_seq2seq 를 그대로 import
import train_style_seq2seq as train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 스크립트와 같은 경로 사용
MODEL_PATH: Path = train.MODEL_PATH        # models/style_seq2seq.pt
VOCAB_PATH: Path = train.VOCAB_PATH        # models/style_vocab.json


# 모델 + 단어장 로드
def load_model() -> Tuple[torch.nn.Module, Dict[str, int], Dict[int, str]]:
    print(f"✅ 모델 로드 시도: {MODEL_PATH}")
    print(f"✅ 단어장 로드 시도: {VOCAB_PATH}")

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"단어장 파일이 없습니다: {VOCAB_PATH}")

    # train.py 의 load_vocab 그대로 사용
    token2idx, idx2token = train.load_vocab(VOCAB_PATH)
    print(f"✅ 단어장 로드 완료 (vocab_size={len(token2idx)})")

    pad_idx = token2idx[train.SPECIAL_TOKENS["PAD"]]

    encoder = train.EncoderRNN(
        vocab_size=len(token2idx),
        emb_size=train.EMBED_SIZE,
        hidden_size=train.HIDDEN_SIZE,
        pad_idx=pad_idx,
        num_layers=train.NUM_LAYERS,
    )
    decoder = train.DecoderRNN(
        vocab_size=len(token2idx),
        emb_size=train.EMBED_SIZE,
        hidden_size=train.HIDDEN_SIZE,
        pad_idx=pad_idx,
        num_layers=train.NUM_LAYERS,
    )
    model = train.Seq2Seq(encoder, decoder, pad_idx).to(DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("✅ Attention Seq2Seq 모델 로드 완료")

    return model, token2idx, idx2token


# 디코딩 + 후처리
@torch.no_grad()
def greedy_decode_min_eos(
    model: torch.nn.Module,
    src_text: str,
    token2idx: Dict[str, int],
    idx2token: Dict[int, str],
    device: torch.device,
    max_len: int = None,
    min_len: int = 5,
) -> str:
    # Attention Seq2Seq greedy decoding (+ 반복 억제 버전)
    model.eval()

    if max_len is None:
        max_len = train.MAX_DECODE_LEN

    pad_idx = token2idx[train.SPECIAL_TOKENS["PAD"]]
    bos_idx = token2idx[train.SPECIAL_TOKENS["BOS"]]
    eos_idx = token2idx[train.SPECIAL_TOKENS["EOS"]]
    unk_idx = token2idx[train.SPECIAL_TOKENS["UNK"]]

    # 인코더 입력 준비
    src_ids = train.encode_sentence(src_text, token2idx, add_bos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)  # (src_len, 1)

    encoder_outputs, hidden = model.encoder(src_tensor)

    # 디코더 시작: BOS
    input_step = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
    generated: List[int] = []

    # Bigram 반복 방지용
    seen_bigrams = set()

    for step in range(max_len):
        logits, hidden, _ = model.decoder(input_step, hidden, encoder_outputs)

        # 상위 10개 후보 중에서 규칙에 맞는 것 고르기
        topv, topi = logits.topk(10, dim=1)  # (1, 10)

        chosen = None
        last_token = generated[-1] if generated else None

        for cand in topi[0]:
            ci = cand.item()

            # (1) 초반에는 EOS 피하기
            if step < min_len and ci == eos_idx:
                continue

            # (2) UNK / PAD / BOS 피하기
            if ci in (unk_idx, pad_idx, bos_idx):
                continue

            # (3) 바로 직전 토큰과 같으면 피하기 (연속 반복 방지)
            if last_token is not None and ci == last_token:
                continue

            # (4) 최근 bigram 반복 방지
            if last_token is not None:
                bg = (last_token, ci)
                if bg in seen_bigrams and step >= min_len:
                    continue

            chosen = ci
            break

        # 전부 걸러졌으면 1등 후보라도 사용
        if chosen is None:
            chosen = topi[0, 0].item()

        # EOS 처리
        if chosen == eos_idx and step >= min_len:
            break
        if chosen == eos_idx:
            # min_len 전에 EOS가 나왔으면 무시하고 계속
            continue

        # bigram 기록
        if last_token is not None:
            seen_bigrams.add((last_token, chosen))

        generated.append(chosen)
        input_step = torch.tensor([[chosen]], dtype=torch.long, device=device)

    # 인덱스를 문장으로 디코딩
    raw_text = train.decode_indices(generated, idx2token)
    return raw_text

import re

def postprocess_text(text: str) -> str:
    # 단순 후처리 개선 버전
    s = text.strip()
    if not s:
        return s

    # 0) BPE 토큰( )을 공백으로 변환 + 공백 정리
    s = s.replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s

    # 1) 같은 단어 3번 이상 연속 → 2번으로 줄이기
    words = s.split()
    compressed = []
    prev = None
    repeat_cnt = 0
    for w in words:
        if w == prev:
            repeat_cnt += 1
            if repeat_cnt >= 2:
                continue
        else:
            prev = w
            repeat_cnt = 0
        compressed.append(w)

    s = " ".join(compressed).strip()
    if not s:
        return s

    # 2) 문장 단위로 나눈 뒤, 연속으로 같은 문장은 1번만 남김
    parts = re.split(r"([\.!?])", s)
    sentences = []
    buf = ""

    for part in parts:
        if part in ".!?":
            buf += part
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf += part

    if buf.strip():
        sentences.append(buf.strip())

    cleaned_sentences = []
    for sent in sentences:
        if not cleaned_sentences or cleaned_sentences[-1] != sent:
            cleaned_sentences.append(sent)

    s = " ".join(cleaned_sentences).strip()
    if not s:
        return s

    # 3) 맨 끝에 한 글자짜리 토막(예: '게', '네', '요')만 덩그러니 있으면 제거
    tokens = s.split()
    if len(tokens) >= 2 and len(tokens[-1]) == 1:
        if tokens[-2].endswith((".", "!", "?")):
            tokens = tokens[:-1]
            s = " ".join(tokens).strip()

    # 4) 끝에 마침표/물음표/느낌표 없으면 마침표 하나 붙이기
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."

    return s


# 인터랙티브 UI
EMO_OPTIONS = {
    1: ("기쁨", "E01"),
    2: ("슬픔", "E02"),
    3: ("분노", "E18"),
    4: ("불안", "E21"),  # 기본
    5: ("상처", "E11"),
    6: ("당황", "E31"),
}

CTX_OPTIONS = {
    1: ("일상", "daily"),
    2: ("진로/학업·업무", "career"),
    3: ("관계/가족·연애", "relation"),
    4: ("감정상담", "counsel"),
}
STYLE_OPTIONS = {
    1: ("해요체", "해요체"),
    2: ("반말체", "반말체"),
    3: ("합쇼체", "합쇼체"),
}


def select_option(prompt: str, options: Dict[int, Tuple[str, str]], default: int) -> Tuple[str, str]:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            idx = default
        else:
            try:
                idx = int(raw)
            except ValueError:
                print("  → 잘못된 입력입니다. 숫자를 입력하세요.")
                continue
        if idx in options:
            return options[idx]
        print("  → 잘못된 번호입니다. 다시 선택하세요.")


def main():
    model, token2idx, idx2token = load_model()

    print()
    print("=== 감정/상황 기반 문장 스타일 변환 인터랙티브 모드 (Attention Seq2Seq + 반복 억제 + 후처리) ===")
    print("문장을 입력하면, 감정/상황/말투를 골라서 변환 결과를 볼 수 있습니다.")
    print("※ 출력 문장은 학습된 Attention Seq2Seq 모델의 결과를 기반으로,")
    print("   n-gram 반복을 줄이는 디코딩/후처리 규칙을 적용한 것입니다.")
    print("종료하려면 빈 줄에서 엔터를 누르거나 'quit', 'q' 를 입력하세요.")
    print()

    while True:
        src = input("👉 변환할 원문 문장을 입력하세요 : ").strip()
        if src == "" or src.lower() in ("quit", "q"):
            print("\n[종료] 인터랙티브 모드를 종료합니다.")
            break

        # 감정 선택
        print("\n[감정 선택]")
        for k, (name, code) in EMO_OPTIONS.items():
            print(f"  {k}. {name}")
        emo_name, emo_code = select_option("번호 선택 (엔터=4: 불안) : ", EMO_OPTIONS, default=4)

        # 상황 선택
        print("\n[상황/컨텍스트 선택]")
        for k, (name, code) in CTX_OPTIONS.items():
            print(f"  {k}. {name}")
        ctx_name, ctx_code = select_option("번호 선택 (엔터=1: 일상) : ", CTX_OPTIONS, default=1)

        # 말투 선택
        print("\n[말투(스타일) 선택]")
        for k, (name, code) in STYLE_OPTIONS.items():
            print(f"  {k}. {name}")
        style_name, style_code = select_option("번호 선택 (엔터=1: 해요체) : ", STYLE_OPTIONS, default=1)

        # 모델 입력 문장 구성
        model_input = f"<ctx:{ctx_code}> <emo:{emo_code}> <style:{style_code}> {src}"
        print("\n[모델 입력]")
        print(f"  {model_input}")
        print(f"  - 상황: {ctx_name}")
        print(f"  - 감정: {emo_name}")
        print(f"  - 말투: {style_name}\n")

        # 디코딩
        raw_out = greedy_decode_min_eos(
            model,
            model_input,
            token2idx,
            idx2token,
            DEVICE,
            max_len=40,
            min_len=5,
        )
        cleaned_out = postprocess_text(raw_out)

        print(f"🧠 모델 원출력(Seq2Seq+Attention, min_len/반복제어 적용 전) : {raw_out if raw_out else '(빈 문자열)'}")
        print(f"✨ 최종 변환 결과(후처리 적용) : {cleaned_out if cleaned_out else '(출력할 문장이 없습니다.)'}")
        print()


if __name__ == "__main__":
    main()