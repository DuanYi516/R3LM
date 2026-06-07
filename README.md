<p align="center">
<h1 align="center">R3LM</h1>

<p align="center">
    <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/📄-Paper-red"></a>
    <a href="https://github.com/DuanYi516/R3LM/blob/main/LICENSE"><img src="https://img.shields.io/github/license/DuanYi516/R3LM"></a>
    <a href="https://huggingface.co/collections/DuanYi/r3lm"><img src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
</p>

Official implementation of the paper:

> **Biological Reasoning-Informed Regression for Interpretable Regulatory DNA Activity Prediction** (KDD 2026)

## Installation

```bash
git clone https://github.com/DuanYi516/R3LM.git
cd R3LM
pip install -e .
```

### Stage I environment (depends on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)):

```bash
pip install llamafactory
```

Register the custom `chatml_with_n` template before training — see [configs/TEMPLATE.md](configs/TEMPLATE.md).

### Optional: RCC motif scanning

```bash
pip install -r requirements-analysis.txt
# or: pip install -e ".[motifs]"
```

## Pipeline

### 1. RCC Construction

RCC transforms each 200 bp enhancer sequence into a structured prompt:

1. **Motif scanning** — FIMO against JASPAR 2026 PPMs (`p < 5×10⁻⁵`), deduplicated by TF family
2. **Sequence statistics** — GC content and global priors
3. **Grammar tags** — rule-based heuristics over motif arrangements
4. **Context injection** — cell type (HepG2, K562, SK-N-SH)
5. **Schema assembly** — fixed field ordering for reproducibility

End-to-end RCC construction:

```bash
pip install -r requirements-analysis.txt

bash scripts/build_rcc.sh \
  /path/to/sequences.csv \
  outputs/K562/rcc-sharegpt.jsonl \
  K562 \
  /path/to/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
```

This runs FIMO (`p < 5×10⁻⁵`), annotates motifs via JASPAR metadata, deduplicates overlapping same-family hits, and writes ShareGPT JSONL with RCC user prompts. JASPAR metadata is cached under `~/.cache/r3lm/` by default.

### 2. CRE-ReasonBench Generation

Rationales are synthesized by conditioning a frontier LLM on (RCC prompt, observed activity level) with a constrained protocol:

- 5–7 numbered causal steps referencing RCC evidence (motifs, families, grammar, GC)
- Conclusion stating discrete activity level ℓ ∈ {0, 1, 2, 3}
- Format-only filtering (schema compliance); no label-conditioned sample selection

> Rationale synthesis requires an external LLM API. We do not ship API keys. Human-reviewed corrections were applied to the 1,000-example Stage-I supervision set per cell type.

### 3. Stage I — Rationale Generation (SFT)

```bash
bash scripts/run_stage1_sft.sh K562
```

Hyperparameters (paper defaults): 100 epochs, lr `1e-5`, batch 1 × grad-accum 16, cosine schedule, DeepSpeed ZeRO-2, bf16.

### 4. Offline Rationale Generation

Generate self-produced rationales for Stage-II training:

```bash
bash scripts/generate_rationales.sh \
  K562 \
  /path/to/82k-no-reason-sharegpt.jsonl \
  ./outputs/K562/self-generated-rationales.jsonl
```

Requires `llamafactory`, `vllm`, and the registered `chatml_with_n` template. Decoding defaults: temperature 0.95, top_p 0.7, max_new_tokens 1024.

### 5. Stage II — Reason-Conditioned Regression

```bash
# Oracle-CoT (gold rationales, upper-bound regression module)
bash scripts/run_stage2_reg.sh \
  --cell_line K562 \
  --input_mode oracle_cot \
  --text_model_name_or_path Qwen/Qwen3-4B-Instruct-2507

# Generated-CoT (self-generated rationales, deployment setting)
bash scripts/run_stage2_reg.sh \
  --cell_line K562 \
  --input_mode generated_cot
```

Stage-II defaults: LoRA r=16, α=32, 10 epochs, lr `1e-5`, batch 2 × grad-accum 4, anchor = last non-padding token.

**Input modes** (`--input_mode`):

| Mode            | Description                                      |
| --------------- | ------------------------------------------------ |
| `oracle_cot`    | RCC + gold reasoning trace (conclusion stripped) |
| `generated_cot` | RCC + self-generated rationale (default)         |


### 6. Evaluation

We randomly sample **100 sequences** from the official regLM chromosome-based test split per cell type. 

## Code Structure

```
R3LM/
├── r3lm/
│   ├── train_stage2.py         # Stage-II regression trainer
│   ├── build_rcc.py            # RCC construction CLI
│   ├── generate_rationales.py  # Stage-I offline inference CLI
│   ├── rcc.py                  # RCC schema assembly
│   ├── jaspar_metadata.py    # JASPAR API metadata cache
│   ├── input_assembly.py       # Stage-II input modes
│   └── motifs.py               # FIMO scanning primitives
├── configs/
│   ├── stage1_sft.yaml
│   ├── dataset_info.json
│   ├── deepspeed_ds_z2.json
│   └── TEMPLATE.md
└── scripts/
    ├── build_rcc.sh
    ├── run_stage1_sft.sh
    ├── run_stage2_reg.sh
    └── generate_rationales.sh
```

## Citation

```bibtex
@inproceedings{Duan2026Biological,
  author    = {Yi Duan and Zhao Yang and Jiwei Zhu and Ying Ba and Chuan Cao and Bing Su},
  title     = {Biological Reasoning-Informed Regression for Interpretable Regulatory {DNA} Activity Prediction},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026)},
  year      = {2026},
  doi       = {10.1145/3770855.3818836},
}
```

## Acknowledgements

This codebase builds upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Stage-I SFT) and [regLM](https://github.com/Genentech/regLM) (data splits and preprocessing conventions).

## License

Apache License 2.0 — see [LICENSE](LICENSE).