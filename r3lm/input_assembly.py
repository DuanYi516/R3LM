"""Assemble Stage-II regression inputs from ShareGPT-style JSONL records."""

import pandas as pd

INPUT_MODES = {
    "oracle_cot",
    "generated_cot",
    "rcc_only",
}


def extract_reasoning_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    marker = "</think>"
    idx = text.find(marker)
    if idx == -1:
        return ""
    return text[: idx + len(marker)].lstrip()


def flatten_messages_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "messages" not in df.columns:
        return df.copy()

    rows = []
    for messages in df["messages"]:
        flattened = {"system": "", "user": "", "assistant": ""}
        if isinstance(messages, list):
            for msg in messages:
                role = msg.get("role", "")
                if role in flattened:
                    flattened[role] = msg.get("content", "")
        rows.append(flattened)

    msg_df = pd.DataFrame(rows, index=df.index)
    remaining_cols = [c for c in df.columns if c != "messages"]
    return pd.concat([msg_df, df[remaining_cols]], axis=1)


def build_model_frame(df: pd.DataFrame, input_mode: str) -> pd.DataFrame:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    frame = flatten_messages_frame(df)

    if "user" in frame.columns:
        user_texts = frame["user"].astype(str).tolist()
    elif "input" in frame.columns:
        user_texts = frame["input"].astype(str).tolist()
    else:
        raise ValueError("Expected `user` or `input` column in dataset.")

    frame = frame.copy()
    frame["user_transformed"] = user_texts

    if input_mode in {"oracle_cot", "generated_cot"}:
        assistant_texts = frame.get("assistant", pd.Series([""] * len(frame), index=frame.index))
        reasoning = assistant_texts.apply(extract_reasoning_prefix).tolist()
        frame["model_input"] = [
            f"{user_text}\n{reasoning_text}".rstrip()
            for user_text, reasoning_text in zip(user_texts, reasoning)
        ]
    else:
        frame["model_input"] = user_texts

    return frame
