"""CLI: build RCC prompts from raw enhancer sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from r3lm.jaspar_metadata import download_core_vertebrates_metadata
from r3lm.rcc import build_rcc_for_sequences


def _default_system_prompt(cell_line: str) -> str:
    return (
        f"You are a computational regulatory genomics expert specializing in "
        f"{cell_line} cell line biology. Your task is to analyze the provided "
        f"DNA sequence features and transcription factor (TF) motifs to predict "
        f"the gene expression level (0-3).\n"
        f"Provide a step-by-step reasoning chain explaining the biological mechanism."
    )


def _read_input_table(path: Path, seq_col: str, id_col: str | None) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        frame = pd.read_csv(path)

    if seq_col not in frame.columns:
        raise ValueError(f"Input file must contain `{seq_col}` column.")

    if id_col and id_col in frame.columns:
        frame = frame.set_index(id_col)
    elif frame.index.name is None:
        frame.index = frame.index.astype(str)
    return frame


def _write_jsonl(frame: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples():
            record = {
                "messages": [
                    {"role": "system", "content": _default_system_prompt(args.cell_line)},
                    {"role": "user", "content": row.rcc_prompt},
                    {"role": "assistant", "content": ""},
                ],
            }
            for field in ("expression_score", "expression_level", "label"):
                if hasattr(row, field):
                    record[field] = getattr(row, field)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_alpaca_jsonl(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples():
            record = {
                "input": row.rcc_prompt,
                "instruction": (
                    "Analyze the regulatory context and predict enhancer activity."
                ),
                "output": "",
            }
            for field in ("expression_score", "expression_level", "label"):
                if hasattr(row, field):
                    record[field] = getattr(row, field)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RCC prompts from DNA sequences.")
    parser.add_argument("--input", required=True, help="CSV/TSV/JSONL with DNA sequences.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--cell_line", default="K562", choices=["K562", "HepG2", "SKNSH"])
    parser.add_argument("--meme_file", required=True, help="JASPAR MEME file for FIMO scanning.")
    parser.add_argument("--metadata_cache", default=None, help="Cached JASPAR metadata JSON.")
    parser.add_argument("--refresh_metadata", action="store_true")
    parser.add_argument("--seq_col", default="Sequence")
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--p_threshold", type=float, default=5e-5)
    parser.add_argument(
        "--include_grammar_tags",
        action="store_true",
        help="Add grammar-tag bullets to Context (off by default to match released data).",
    )
    parser.add_argument(
        "--format",
        choices=["sharegpt", "alpaca"],
        default="sharegpt",
        help="sharegpt: Stage-I/II JSONL; alpaca: legacy seq+motif-style JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_cache) if args.metadata_cache else None
    if args.refresh_metadata:
        metadata = download_core_vertebrates_metadata(metadata_path)
    else:
        from r3lm.jaspar_metadata import load_metadata

        metadata = load_metadata(metadata_path)

    frame = _read_input_table(Path(args.input), args.seq_col, args.id_col)
    built = build_rcc_for_sequences(
        sequences=frame,
        cell_line=args.cell_line,
        meme_file=args.meme_file,
        metadata=metadata,
        p_threshold=args.p_threshold,
        include_grammar_tags=args.include_grammar_tags,
        seq_col=args.seq_col,
    )

    output_path = Path(args.output)
    if args.format == "sharegpt":
        _write_jsonl(built, output_path, args)
    else:
        _write_alpaca_jsonl(built, output_path)
    print(f"Wrote {len(built)} RCC prompts to {output_path}")


if __name__ == "__main__":
    main()
