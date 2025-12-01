# src/load_emotion_corpus.py
import json
import pandas as pd


def _load_emotion_json(json_path: str, split: str) -> pd.DataFrame:
    rows = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 마지막 콤마 제거
            if line.endswith(","):
                line = line[:-1]

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            profile = obj.get("profile", {})
            persona_id = profile.get("persona-id")

            emotion_info = profile.get("emotion", {})
            emotion_id = emotion_info.get("emotion-id")
            emotion_type = emotion_info.get("type") 

            talk = obj.get("talk", {})
            content = talk.get("content", {})

            # HS01, SS01, HS02... 각각을 한 행으로 풀기
            for key, text in content.items():
                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue

                # role / turn 계산
                if key.startswith("H"):      # HS01, HS02 ...
                    role = "user"
                elif key.startswith("S"):    # SS01, SS02 ...
                    role = "system"
                else:
                    role = "other"

                turn = None
                if len(key) >= 3 and key[2:].isdigit():
                    turn = int(key[2:])

                rows.append(
                    {
                        "split": split,
                        "persona_id": persona_id,
                        "emotion_id": emotion_id,
                        "emotion_type": emotion_type,
                        "role": role,   # build_pairs_v2.py에서 사용
                        "slot": key,    # HS01 / SS01 등
                        "turn": turn,
                        "text": text,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "split",
            "persona_id",
            "emotion_id",
            "emotion_type",
            "role",
            "slot",
            "turn",
            "text",
        ],
    )


def load_emotion_data(train_json_path: str, valid_json_path: str) -> pd.DataFrame:
    # 감성대화 Training + Validation JSON 두 개를 읽어서 합침
    df_train = _load_emotion_json(train_json_path, split="train")
    df_valid = _load_emotion_json(valid_json_path, split="valid")
    df_all = pd.concat([df_train, df_valid], ignore_index=True)
    return df_all


if __name__ == "__main__":
    print("이 모듈은 run.py / build_pairs_v2.py에서 import 해서 사용합니다.")