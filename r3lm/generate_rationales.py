"""Offline rationale generation with LLaMA-Factory templates + vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from r3lm.hf_assets import resolve_stage1_model


def _load_sharegpt_records(path: Path) -> List[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_prompt_ids(template, tokenizer, messages: List[dict]) -> List[int]:
    prompt_messages = []
    for message in messages:
        if message["role"] == "assistant":
            break
        prompt_messages.append(message)
    prompt_messages.append({"role": "assistant", "content": ""})
    prompt_ids, _ = template.encode_oneturn(tokenizer, prompt_messages)
    return prompt_ids


def generate_with_vllm(
    records: Iterable[dict],
    model_path: str,
    template_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
    batch_size: int,
) -> List[str]:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.extras.misc import get_device_count
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer
    from vllm import LLM, SamplingParams

    model_args, data_args, _, _generating_args = get_infer_args(
        {
            "model_name_or_path": model_path,
            "template": template_name,
            "cutoff_len": 2048,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=model_args.infer_dtype,
        tensor_parallel_size=get_device_count() or 1,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop_token_ids=template.get_stop_token_ids(tokenizer),
    )

    records = list(records)
    predictions: List[str] = []
    for start in tqdm(range(0, len(records), batch_size), desc="vLLM generate"):
        batch = records[start : start + batch_size]
        prompts = [
            {"prompt_token_ids": _build_prompt_ids(template, tokenizer, item["messages"])}
            for item in batch
        ]
        outputs = llm.generate(prompts, sampling_params)
        predictions.extend(result.outputs[0].text for result in outputs)
    return predictions


def write_sharegpt_output(
    records: List[dict],
    predictions: List[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, predictions):
            messages = []
            for message in record["messages"]:
                if message["role"] == "assistant":
                    messages.append({"role": "assistant", "content": prediction})
                else:
                    messages.append(message)
            if not any(msg["role"] == "assistant" for msg in messages):
                messages.append({"role": "assistant", "content": prediction})

            out = dict(record)
            out["messages"] = messages
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-produced rationales for Stage II.")
    parser.add_argument("--input_jsonl", required=True, help="ShareGPT JSONL with RCC user prompts.")
    parser.add_argument("--output_jsonl", required=True, help="ShareGPT JSONL with filled assistant rationales.")
    parser.add_argument(
        "--cell_line",
        type=str,
        default="K562",
        choices=["K562", "HepG2", "SKNSH"],
        help="Cell line; used to auto-download DuanYi/R3LM_<cell_line> when --model_path is omitted.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional local path or HF repo id; defaults to DuanYi/R3LM_<cell_line>.",
    )
    parser.add_argument("--template", default="chatml_with_n")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_sharegpt_records(Path(args.input_jsonl))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    model_path = resolve_stage1_model(args.cell_line, args.model_path)
    predictions = generate_with_vllm(
        records=records,
        model_path=model_path,
        template_name=args.template,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
    )
    write_sharegpt_output(records, predictions, Path(args.output_jsonl))
    print(f"Wrote {len(predictions)} rationales to {args.output_jsonl}")


if __name__ == "__main__":
    main()
