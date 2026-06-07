"""Resolve Hugging Face datasets and Stage-I checkpoints with auto-download."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HF_DATASET_REPO = "DuanYi/Cre-ReasonBench"

STAGE1_MODEL_REPOS = {
    "HepG2": "DuanYi/R3LM_HepG2",
    "K562": "DuanYi/R3LM_K562",
    "SKNSH": "DuanYi/R3LM_SKNSH",
}

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PACKAGE_ROOT / "data"
_DATASET_INFO_SRC = _PACKAGE_ROOT / "configs" / "dataset_info.json"


def ensure_hf_endpoint() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _has_local_data(data_dir: Path) -> bool:
    return any(data_dir.glob("*/*.jsonl"))


def ensure_data_dir(data_dir: str | Path | None = None) -> Path:
    """Download CRE-ReasonBench into *data_dir* if missing."""
    ensure_hf_endpoint()
    path = Path(data_dir or os.environ.get("R3LM_DATA_DIR", _DEFAULT_DATA_DIR))
    if _has_local_data(path):
        return path.resolve()

    from huggingface_hub import snapshot_download

    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(path),
    )
    return path.resolve()


def prepare_llamafactory_data_dir(data_dir: str | Path | None = None) -> Path:
    """Ensure dataset files and dataset_info.json for LLaMA-Factory."""
    path = ensure_data_dir(data_dir)
    dest = path / "dataset_info.json"
    if not dest.exists():
        shutil.copy2(_DATASET_INFO_SRC, dest)
    return path


def stage1_model_repo(cell_line: str) -> str:
    repo = STAGE1_MODEL_REPOS.get(cell_line)
    if repo is None:
        raise ValueError(
            f"Unknown cell_line {cell_line!r}. Expected one of {sorted(STAGE1_MODEL_REPOS)}."
        )
    return repo


def resolve_stage1_model(cell_line: str, model_path: str | None = None) -> str:
    """Return a local path to the Stage-I checkpoint, downloading from HF if needed."""
    if model_path and Path(model_path).exists():
        return str(Path(model_path).resolve())

    repo_id = model_path if model_path and "/" in model_path else stage1_model_repo(cell_line)
    ensure_hf_endpoint()
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, repo_type="model")
