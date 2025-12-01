# src/infer_style_seq2seq.py
"""
감정 + 상황 + 스타일 기반 Attention Seq2Seq 추론 스크립트

- 학습된 모델: models/style_seq2seq.pt
- 단어장: models/style_vocab.json
- 구조:
    * train_style_seq2seq.py 와 같은 EncoderRNN / DecoderRNN(LuongAttention) / Seq2Seq 사용
    * n-gram 반복을 줄이는 디코딩 전략 적용
    * 디코딩 결과가 너무 짧거나 / 반복이 심하면
      -> "모델이 출력을 제대로 생성하지 못했습니다." 메시지 출력

※ 데모용 하드코딩 문장/룰 없음 (전부 모델 기반)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ==============================
# 0. 경로 / 하이퍼파라미터
# ==============================

MODEL_PATH: Path = Path("models/style_seq2seq.pt")
VOCAB_PATH: Path = Path("models/style_vocab.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = {
    "PAD": "<pad>",
    "BOS": "<bos>",
    "EOS": "<eos>",
    "UNK": "<unk>",
}

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
MAX_DECODE_LEN = 60


# ==============================
# 1. Vocab & 유틸
# ==============================

def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    token2idx: Dict[str, int] = obj["token2idx"]
    idx2token: Dict[int, str] = {idx: tok for tok, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize(text: str) -> List[str]:
    return text.strip().split()


def encode_sentence(
    text: str,
    token2idx: Dict[str, int],
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[int]:
    unk_idx = token2idx[SPECIAL_TOKENS["UNK"]]
    tokens = tokenize(text)
    ids: List[int] = []
    if add_bos:
        ids.append(token2idx[SPECIAL_TOKENS["BOS"]])
    for t in tokens:
        ids.append(token2idx.get(t, unk_idx))
    if add_eos:
        ids.append(token2idx[SPECIAL_TOKENS["EOS"]])
    return ids


def decode_indices(indices: List[int], idx2token: Dict[int, str]) -> str:
    tokens: List[str] = []
    for idx in indices:
        tok = idx2token.get(int(idx), SPECIAL_TOKENS["UNK"])
        if tok == SPECIAL_TOKENS["EOS"]:
            break
        if tok in (SPECIAL_TOKENS["BOS"], SPECIAL_TOKENS["PAD"]):
            continue
        tokens.append(tok)
    return " ".join(tokens).strip()


# 간단한 반복 감지 함수
def is_repetitive(text: str) -> bool:
    """
    토큰 중복 정도를 기준으로
    - 토큰 수가 너무 적거나(<=1)
    - 유니크 토큰 비율이 너무 낮으면(예: 0.5 미만)
    '반복적인 비정상 출력'으로 간주
    """
    tokens = [t for t in tokenize(text) if t not in [".", ",", "!", "?", "…"]]
    if len(tokens) <= 1:
        return True
    unique = set(tokens)
    ratio = len(unique) / len(tokens)
    return ratio < 0.5


# ==============================
# 2. 모델 구조 (train 과 동일)
# ==============================

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 pad_idx: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=False,
        )

    def forward(self, src: torch.Tensor):
        """
        src: (src_len, batch)
        """
        embedded = self.embedding(src)              # (src_len, B, D)
        outputs, hidden = self.gru(embedded)        # outputs: (src_len, B, H)
        return outputs, hidden                      # hidden: (num_layers, B, H)


class LuongAttention(nn.Module):
    """
    Luong dot-product attention:
    score(h_t, h_s) = h_t^T h_s
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        decoder_hidden: (1, B, H)
        encoder_outputs: (src_len, B, H)
        return:
          context: (B, H)
          attn_weights: (B, src_len)
        """
        # (1, B, H) -> (B, 1, H)
        dec = decoder_hidden.permute(1, 0, 2)      # (B, 1, H)
        # encoder_outputs: (src_len, B, H) -> (B, src_len, H)
        enc = encoder_outputs.permute(1, 0, 2)     # (B, src_len, H)

        # dot-product: (B, 1, H) x (B, H, src_len) -> (B, 1, src_len)
        scores = torch.bmm(dec, enc.transpose(1, 2))   # (B, 1, src_len)
        attn_weights = torch.softmax(scores, dim=-1)   # (B, 1, src_len)

        # context: (B, 1, src_len) x (B, src_len, H) -> (B, 1, H)
        context = torch.bmm(attn_weights, enc)         # (B, 1, H)
        context = context.squeeze(1)                   # (B, H)
        attn_weights = attn_weights.squeeze(1)         # (B, src_len)

        return context, attn_weights


class DecoderRNN(nn.Module):
    """
    Luong Attention 기반 Decoder:
      1) 임베딩 입력으로 GRU 한 스텝
      2) GRU output(h_t)와 encoder_outputs에 대해 attention 계산
      3) [h_t; context]를 linear -> vocab 분포
    """
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 pad_idx: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=num_layers)
        self.attn = LuongAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self,
                input_step: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor):
        """
        input_step: (1, B)
        hidden:     (1, B, H)
        encoder_outputs: (src_len, B, H)
        """
        embedded = self.embedding(input_step)            # (1, B, D)
        output, hidden = self.gru(embedded, hidden)      # output: (1, B, H)

        # 어텐션
        context, attn_weights = self.attn(output, encoder_outputs)  # (B, H), (B, src_len)

        # output: (1, B, H) -> (B, H)
        output = output.squeeze(0)                       # (B, H)

        # [h_t; context] -> vocab
        concat = torch.cat([output, context], dim=1)     # (B, 2H)
        logits = self.fc(concat)                         # (B, vocab_size)

        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx


# ==============================
# 3. 디코딩 전략 (n-gram 반복 줄이기)
# ==============================

@torch.no_grad()
def decode_with_strategy(
    model: Seq2Seq,
    src_text: str,
    token2idx: Dict[str, int],
    idx2token: Dict[int, str],
    device: torch.device,
    max_len: int = MAX_DECODE_LEN,
) -> str:
    """
    Attention Seq2Seq + 간단한 n-gram 반복 억제 디코딩
    (모델 출력 100% 사용, 규칙은 '선택 과정'에서만 사용)
    """
    model.eval()

    pad_idx = token2idx[SPECIAL_TOKENS["PAD"]]
    bos_idx = token2idx[SPECIAL_TOKENS["BOS"]]
    eos_idx = token2idx[SPECIAL_TOKENS["EOS"]]
    unk_idx = token2idx[SPECIAL_TOKENS["UNK"]]

    # 1) 인코더 입력 준비
    src_ids = encode_sentence(src_text, token2idx, add_bos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)  # (src_len, 1)

    encoder_outputs, hidden = model.encoder(src_tensor)

    # 2) 디코더 반복
    input_step = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
    generated: List[int] = []

    for step in range(max_len):
        logits, hidden, _ = model.decoder(input_step, hidden, encoder_outputs)
        # logits: (1, vocab_size)
        # top-k에서 후보를 골라 반복/UNK/PAD/BOS 최대한 회피
        topv, topi = logits.topk(5, dim=1)  # (1, 5)

        chosen = None
        for cand in topi[0]:
            ci = cand.item()
            # EOS를 너무 빨리 내지 않도록 최소 길이 예시 (3 토큰)
            if ci == eos_idx and len(generated) < 3:
                continue
            if ci in (pad_idx, bos_idx):
                continue

            # 간단한 n-gram(2,3-gram) 반복 방지
            if len(generated) >= 2 and ci == generated[-1]:
                # 바로 직전 토큰 반복은 일단 피함
                continue
            if len(generated) >= 3 and generated[-2:] == [generated[-3], ci]:
                # a b a b 같은 패턴도 피하기
                continue

            chosen = ci
            break

        # 전부 걸러지면 그냥 argmax로
        if chosen is None:
            chosen = topi[0, 0].item()

        if chosen == eos_idx:
            break

        generated.append(chosen)
        input_step = torch.tensor([[chosen]], dtype=torch.long, device=device)

    return decode_indices(generated, idx2token)


# ==============================
# 4. 인터랙티브 루프
# ==============================

def interactive_loop(model: Seq2Seq, token2idx: Dict[str, int], idx2token: Dict[int, str]):
    emo_map = {
        1: "E01",  # 기쁨
        2: "E02",  # 슬픔
        3: "E18",  # 분노
        4: "E04",  # 불안
        5: "E05",  # 상처
        6: "E06",  # 당황
    }

    ctx_map = {
        1: "daily",        # 일상
        2: "task",         # 진로/학업·업무
        3: "relationship", # 관계/가족·연애
        4: "counsel",      # 감정상담
    }

    style_map = {
        1: "해요체",
        2: "반말체",
        3: "합쇼체",
    }

    print("\n=== 감정/상황 기반 문장 스타일 변환 인터랙티브 모드 ===")
    print("문장을 입력하면, 감정/상황/말투를 골라서 변환 결과를 볼 수 있습니다.")
    print("종료하려면 빈 줄에서 엔터를 누르거나 'quit', 'q' 를 입력하세요.\n")

    while True:
        src = input("👉 변환할 원문 문장을 입력하세요 : ").strip()
        if src == "" or src.lower() in ("quit", "q"):
            print("\n[종료] 인터랙티브 모드를 종료합니다.")
            break

        # ----- 감정 선택 -----
        print("\n[감정 선택]")
        print("  1. 기쁨")
        print("  2. 슬픔")
        print("  3. 분노")
        print("  4. 불안")
        print("  5. 상처")
        print("  6. 당황")
        emo_in = input("번호 선택 (엔터=1: 기쁨) : ").strip()
        emo_choice = 1 if emo_in == "" else max(1, min(6, int(emo_in)))
        emo_tag = emo_map[emo_choice]

        # ----- 상황 선택 -----
        print("\n[상황/컨텍스트 선택]")
        print("  1. 일상")
        print("  2. 진로/학업·업무")
        print("  3. 관계/가족·연애")
        print("  4. 감정상담")
        ctx_in = input("번호 선택 (엔터=1: 일상) : ").strip()
        ctx_choice = 1 if ctx_in == "" else max(1, min(4, int(ctx_in)))
        ctx_tag = ctx_map[ctx_choice]

        # ----- 말투 선택 -----
        print("\n[말투(스타일) 선택]")
        print("  1. 해요체")
        print("  2. 반말체")
        print("  3. 합쇼체")
        style_in = input("번호 선택 (엔터=1: 해요체) : ").strip()
        style_choice = 1 if style_in == "" else max(1, min(3, int(style_in)))
        style_tag = style_map[style_choice]

        # ----- 모델 입력 문장 구성 -----
        control_prefix = f"<ctx:{ctx_tag}> <emo:{emo_tag}> <style:{style_tag}>"
        model_input = f"{control_prefix} {src}".strip()

        print("\n[모델 입력]")
        print(f"  {model_input}")
        print(f"  - 상황: {['일상','진로/학업·업무','관계/가족·연애','감정상담'][ctx_choice-1]}")
        print(f"  - 감정: {['기쁨','슬픔','분노','불안','상처','당황'][emo_choice-1]}")
        print(f"  - 말투: {style_tag}")

        # ----- 디코딩 -----
        raw_out = decode_with_strategy(model, model_input, token2idx, idx2token, DEVICE)

        print("\n🧠 모델 출력 :", raw_out if raw_out else "(빈 문장)")

        # 반복/비정상 여부 판단
        if (not raw_out) or is_repetitive(raw_out):
            print("⚠️ 모델이 출력을 제대로 생성하지 못했습니다. (반복 또는 비정상 출력으로 판단)\n")
        else:
            print("✨ 최종 변환 결과 :", raw_out, "\n")


# ==============================
# 5. main
# ==============================

def main():
    print(f"✅ 모델 로드 시도: {MODEL_PATH}")
    print(f"✅ 단어장 로드 시도: {VOCAB_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"단어장 파일을 찾을 수 없습니다: {VOCAB_PATH}")

    token2idx, idx2token = load_vocab(VOCAB_PATH)
    vocab_size = len(token2idx)
    print(f"✅ 단어장 로드 완료 (vocab_size={vocab_size})")

    pad_idx = token2idx[SPECIAL_TOKENS["PAD"]]

    encoder = EncoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, pad_idx, NUM_LAYERS)
    decoder = DecoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, pad_idx, NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, pad_idx).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print("✅ Attention Seq2Seq 모델 로드 완료")

    interactive_loop(model, token2idx, idx2token)


if __name__ == "__main__":
    main()
