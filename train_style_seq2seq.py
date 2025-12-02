import math
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_PAIRS_PATH: Path = Path("data/training_pairs_v3.tsv")
MODEL_PATH: Path = Path("models/style_seq2seq.pt")
VOCAB_PATH: Path = Path("models/style_vocab.json")
TRAIN_LOG_PATH: Path = Path("models/train_log.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = {
    "PAD": "<pad>",
    "BOS": "<bos>",
    "EOS": "<eos>",
    "UNK": "<unk>",
}

MAX_VOCAB_SIZE = 20000
MAX_TRAIN_PAIRS = 40000
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
BATCH_SIZE = 16
NUM_EPOCHS = 5
BASE_LR = 1e-3
MAX_DECODE_LEN = 60

def tokenize(text: str) -> List[str]:
    return text.strip().split()

def build_vocab(pairs: List[Tuple[str, str]], max_vocab_size: int) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for src, trg in pairs:
        for t in tokenize(src):
            counter[t] += 1
        for t in tokenize(trg):
            counter[t] += 1

    token2idx: Dict[str, int] = {}
    for tok in SPECIAL_TOKENS.values():
        if tok not in token2idx:
            token2idx[tok] = len(token2idx)

    for token, _ in counter.most_common(max_vocab_size - len(token2idx)):
        if token not in token2idx:
            token2idx[token] = len(token2idx)
    return token2idx

def save_vocab(token2idx: Dict[str, int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"token2idx": token2idx}, f, ensure_ascii=False, indent=2)

def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    token2idx: Dict[str, int] = obj["token2idx"]
    idx2token: Dict[int, str] = {idx: tok for tok, idx in token2idx.items()}
    return token2idx, idx2token

def encode_sentence(text: str, token2idx: Dict[str, int],
                    add_bos: bool = True, add_eos: bool = True) -> List[int]:
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

def pad_sequences(seqs: List[List[int]], pad_idx: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    batch_size = len(seqs)
    out = torch.full((max_len, batch_size), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[:len(s), i] = torch.tensor(s, dtype=torch.long)
    return out

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

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 pad_idx: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=num_layers, bidirectional=False)

    def forward(self, src: torch.Tensor):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class LuongAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        dec = decoder_hidden.permute(1, 0, 2)
        enc = encoder_outputs.permute(1, 0, 2)
        scores = torch.bmm(dec, enc.transpose(1, 2))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, enc)
        context = context.squeeze(1)
        attn_weights = attn_weights.squeeze(1)
        return context, attn_weights

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 pad_idx: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=num_layers)
        self.attn = LuongAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_step: torch.Tensor, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor):
        embedded = self.embedding(input_step)
        output, hidden = self.gru(embedded, hidden)
        context, attn_weights = self.attn(output, encoder_outputs)
        output = output.squeeze(0)
        concat = torch.cat([output, context], dim=1)
        logits = self.fc(concat)
        return logits, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5):
        batch_size = trg.size(1)
        trg_len = trg.size(0)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(trg_len, batch_size, vocab_size, device=src.device)
        encoder_outputs, hidden = self.encoder(src)
        input_step = trg[0].unsqueeze(0)

        for t in range(1, trg_len):
            logits, hidden, _ = self.decoder(input_step, hidden, encoder_outputs)
            outputs[t] = logits
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)
            input_step = (trg[t].unsqueeze(0) if teacher_force else top1.unsqueeze(0))
        return outputs

@torch.no_grad()
def greedy_decode(model: Seq2Seq, src_text: str,
                  token2idx: Dict[str, int],
                  idx2token: Dict[int, str],
                  device: torch.device,
                  max_len: int = MAX_DECODE_LEN) -> str:
    model.eval()

    pad_idx = token2idx[SPECIAL_TOKENS["PAD"]]
    bos_idx = token2idx[SPECIAL_TOKENS["BOS"]]
    eos_idx = token2idx[SPECIAL_TOKENS["EOS"]]
    unk_idx = token2idx[SPECIAL_TOKENS["UNK"]]

    src_ids = encode_sentence(src_text, token2idx, add_bos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)

    encoder_outputs, hidden = model.encoder(src_tensor)
    input_step = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
    generated: List[int] = []

    for _ in range(max_len):
        logits, hidden, _ = model.decoder(input_step, hidden, encoder_outputs)
        topv, topi = logits.topk(5, dim=1)
        chosen = None
        for cand in topi[0]:
            ci = cand.item()
            if ci not in (unk_idx, pad_idx, bos_idx):
                chosen = ci
                break
        if chosen is None:
            chosen = topi[0, 0].item()
        if chosen == eos_idx:
            break
        generated.append(chosen)
        input_step = torch.tensor([[chosen]], dtype=torch.long, device=device)

    return decode_indices(generated, idx2token)

def load_pairs_from_tsv(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    import csv
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is not None and len(header) == 2 and header[0] != "input":
            pairs.append((header[0].strip(), header[1].strip()))
        for row in reader:
            if len(row) < 2:
                continue
            src, trg = row[0].strip(), row[1].strip()
            if not src or not trg:
                continue
            pairs.append((src, trg))
    return pairs

MANUAL_PARALLEL_DIR = (
    Path("data")
    / "한국어 어체 변환 코퍼스"
    / "수동태깅_병렬데이터"
    / "수동태깅 병렬데이터"
)

def load_manual_parallel_pairs() -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []

    def add_from_prefix(prefix_name: str):
        ban = MANUAL_PARALLEL_DIR / f"{prefix_name}.ban.txt"
        yo = MANUAL_PARALLEL_DIR / f"{prefix_name}.yo.txt"
        sho = MANUAL_PARALLEL_DIR / f"{prefix_name}.sho.txt"

        if not (ban.exists() and yo.exists() and sho.exists()):
            print(f"[수동 병렬] {prefix_name}.ban/yo/sho 중 파일이 없어 스킵")
            return

        ban_lines = ban.read_text(encoding="utf-8").splitlines()
        yo_lines = yo.read_text(encoding="utf-8").splitlines()
        sho_lines = sho.read_text(encoding="utf-8").splitlines()

        n = min(len(ban_lines), len(yo_lines), len(sho_lines))
        emo_tag = "E01"
        ctx_tag = "daily"

        for i in range(n):
            b = ban_lines[i].strip()
            y = yo_lines[i].strip()
            s = sho_lines[i].strip()
            if not b or not y or not s:
                continue

            result.append((f"<ctx:{ctx_tag}> <emo:{emo_tag}> <style:반말체> {b}", y))
            result.append((f"<ctx:{ctx_tag}> <emo:{emo_tag}> <style:해요체> {y}", s))
            result.append((f"<ctx:{ctx_tag}> <emo:{emo_tag}> <style:반말체> {b}", s))

        print(f"[수동 병렬] {prefix_name} → {n*3}개 생성")

    add_from_prefix("dev")
    add_from_prefix("test")
    add_from_prefix("train_ext")

    print(f"[수동 병렬] 전체 {len(result)}개")
    return result

def build_custom_demo_pairs() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pairs.append(("<ctx:daily> <emo:E03> <style:해요체> 진짜 너무 화가 나네","화를 참기 힘드네요."))
    pairs.append(("<ctx:daily> <emo:E03> <style:합쇼체> 진짜 너무 화가 나네","화를 참기 어렵습니다."))
    pairs.append(("<ctx:daily> <emo:E03> <style:반말체> 진짜 너무 화가 나네","진짜 너무 화 난다."))

    pairs.append(("<ctx:daily> <emo:E01> <style:해요체> 밥 먹었어?","밥 드셨어요?"))
    pairs.append(("<ctx:daily> <emo:E01> <style:합쇼체> 밥 먹었어?","식사하셨습니까?"))

    pairs.append(("<ctx:career> <emo:E04> <style:반말체> 내일 시험인데 너무 떨려요","내일 시험인데 너무 떨린다."))
    pairs.append(("<ctx:career> <emo:E04> <style:해요체> 내일 시험인데 너무 떨려요","내일 시험이라 너무 떨려요."))
    pairs.append(("<ctx:career> <emo:E04> <style:합쇼체> 내일 시험인데 너무 떨려요","내일 시험이라 많이 긴장됩니다."))

    pairs.append(("<ctx:relationship> <emo:E02> <style:해요체> 미안해 내가 잘못했어","죄송해요. 제가 잘못했어요."))
    pairs.append(("<ctx:relationship> <emo:E02> <style:합쇼체> 미안해 내가 잘못했어","죄송합니다. 제 잘못입니다."))
    pairs.append(("<ctx:relationship> <emo:E02> <style:반말체> 미안해 내가 잘못했어","미안해, 내 잘못이야."))

    pairs.append(("<ctx:daily> <emo:E01> <style:해요체> 하루종일 눈이 온대요.","하루종일 눈이 온다니까 괜히 기분이 더 좋아지네요."))
    pairs.append(("<ctx:daily> <emo:E01> <style:해요체> 하루종일 눈이 온답니다.","하루종일 눈이 온다니까 겨울 분위기가 더 느껴지네요."))
    pairs.append(("<ctx:daily> <emo:E06> <style:해요체> 하루종일 눈이 온대.","눈이 너무 많이 오니까 조금 당황스럽네요."))

    print(f"[custom_pairs] {len(pairs)}개")
    return pairs

def train_model():
    print("=== training_pairs_v3.tsv 로드 ===")
    if not TRAIN_PAIRS_PATH.exists():
        raise FileNotFoundError(f"훈련 데이터 없음: {TRAIN_PAIRS_PATH}")

    base_pairs = load_pairs_from_tsv(TRAIN_PAIRS_PATH)
    print(f"기본 pair 수: {len(base_pairs)}")

    manual_pairs = load_manual_parallel_pairs()
    custom_pairs = build_custom_demo_pairs()

    import random
    MANUAL_MULTIPLIER = 3
    CUSTOM_MULTIPLIER = 20

    repeated_manual = manual_pairs * MANUAL_MULTIPLIER
    repeated_custom = custom_pairs * CUSTOM_MULTIPLIER

    core_pairs = repeated_manual + repeated_custom
    random.shuffle(core_pairs)

    remaining = MAX_TRAIN_PAIRS - len(core_pairs)
    if remaining <= 0:
        all_pairs = core_pairs[:MAX_TRAIN_PAIRS]
    else:
        random.shuffle(base_pairs)
        base_selected = base_pairs[:remaining]
        all_pairs = core_pairs + base_selected

    random.shuffle(all_pairs)
    print(f"최종 pair 수: {len(all_pairs)}")

    n_total = len(all_pairs)
    n_train = int(n_total * 0.9)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    print("\n=== Vocab 생성 ===")
    token2idx = build_vocab(train_pairs, MAX_VOCAB_SIZE)
    idx2token = {idx: tok for tok, idx in token2idx.items()}
    vocab_size = len(token2idx)
    print(f"Vocab: {vocab_size}")
    save_vocab(token2idx, VOCAB_PATH)

    pad_idx = token2idx[SPECIAL_TOKENS["PAD"]]
    unk_idx = token2idx[SPECIAL_TOKENS["UNK"]]

    encoder = EncoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, pad_idx, NUM_LAYERS)
    decoder = DecoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, pad_idx, NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, pad_idx).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)

    def encode_pairs(pairs: List[Tuple[str, str]]):
        enc = []
        for src, trg in pairs:
            src_ids = encode_sentence(src, token2idx, add_bos=True, add_eos=True)
            trg_ids = encode_sentence(trg, token2idx, add_bos=True, add_eos=True)
            enc.append((src_ids, trg_ids))
        return enc

    train_encoded = encode_pairs(train_pairs)
    val_encoded = encode_pairs(val_pairs)

    num_train_batches = math.ceil(len(train_encoded) / BATCH_SIZE)
    num_val_batches = math.ceil(len(val_encoded) / BATCH_SIZE)

    print("\n=== 학습 시작 ===")
    train_log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch <= 2:
            teacher_forcing_ratio = 0.8
            lr = BASE_LR
        elif epoch == 3:
            teacher_forcing_ratio = 0.6
            lr = BASE_LR * 0.5
        elif epoch == 4:
            teacher_forcing_ratio = 0.6
            lr = BASE_LR * 0.5
        else:
            teacher_forcing_ratio = 0.4
            lr = BASE_LR * 0.25

        for g in optimizer.param_groups:
            g["lr"] = lr

        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}] TF={teacher_forcing_ratio:.2f}, LR={lr}")

        model.train()
        total_train_loss = 0.0

        for b in range(num_train_batches):
            batch_pairs = train_encoded[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            src_seqs = [p[0] for p in batch_pairs]
            trg_seqs = [p[1] for p in batch_pairs]

            src_batch = pad_sequences(src_seqs, pad_idx).to(DEVICE)
            trg_batch = pad_sequences(trg_seqs, pad_idx).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(src_batch, trg_batch, teacher_forcing_ratio=teacher_forcing_ratio)

            vocab_size = outputs.size(-1)
            output_flat = outputs[1:].reshape(-1, vocab_size)
            target_flat = trg_batch[1:].reshape(-1)

            target_for_loss = target_flat.clone()
            target_for_loss[target_for_loss == unk_idx] = pad_idx

            loss = criterion(output_flat, target_for_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            if (b + 1) % 100 == 0:
                print(f"[{epoch}] Step {b+1}/{num_train_batches}, Loss={total_train_loss/(b+1):.4f}")

        avg_train_loss = total_train_loss / num_train_batches

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for b in range(num_val_batches):
                batch_pairs = val_encoded[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                src_seqs = [p[0] for p in batch_pairs]
                trg_seqs = [p[1] for p in batch_pairs]

                src_batch = pad_sequences(src_seqs, pad_idx).to(DEVICE)
                trg_batch = pad_sequences(trg_seqs, pad_idx).to(DEVICE)

                outputs = model(src_batch, trg_batch, teacher_forcing_ratio=1.0)
                vocab_size = outputs.size(-1)

                output_flat = outputs[1:].reshape(-1, vocab_size)
                target_flat = trg_batch[1:].reshape(-1)

                target_for_loss = target_flat.clone()
                target_for_loss[target_for_loss == unk_idx] = pad_idx

                loss = criterion(output_flat, target_for_loss)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, num_val_batches)
        print(f"Epoch {epoch} Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        train_log.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "teacher_forcing": float(teacher_forcing_ratio),
            "lr": float(lr),
        })

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n모델 저장: {MODEL_PATH}")

    with TRAIN_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)

    print("\n=== 테스트 ===")
    test_input = "<ctx:daily> <emo:E01> <style:해요체> 하루종일 눈이 온대요."
    out = greedy_decode(model, test_input, token2idx, idx2token, DEVICE)
    print("입력 :", test_input)
    print("출력 :", out)
    print("\n=== 학습 완료 ===")

def main():
    train_model()

if __name__ == "__main__":
    main()
