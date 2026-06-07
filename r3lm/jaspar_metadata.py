"""JASPAR matrix metadata helpers (name / family annotation for RCC)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

JASPAR_API = "https://jaspar.elixir.no/api/v1/matrix/"
DEFAULT_CACHE_DIR = Path(os.environ.get("R3LM_CACHE_DIR", Path.home() / ".cache" / "r3lm"))
DEFAULT_METADATA_PATH = DEFAULT_CACHE_DIR / "jaspar_core_vertebrates_metadata.json"


def _join_annotation(values: Optional[list]) -> str:
    if not values:
        return ""
    return "; ".join(str(v) for v in values if v)


def _fetch_json(url: str, retries: int = 3, sleep_s: float = 1.0) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def download_core_vertebrates_metadata(
    output_path: Path = DEFAULT_METADATA_PATH,
    page_size: int = 100,
) -> Dict[str, dict]:
    """Download CORE vertebrate matrix metadata from the JASPAR REST API."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, dict] = {}
    page = 1
    while True:
        url = (
            f"{JASPAR_API}?tax_group=vertebrates&collection=CORE"
            f"&page_size={page_size}&page={page}"
        )
        payload = _fetch_json(url)
        for item in payload.get("results", []):
            matrix_id = item["matrix_id"]
            metadata[matrix_id] = {
                "name": item.get("name", matrix_id),
                "family": _join_annotation(item.get("family")),
                "class": _join_annotation(item.get("class")),
            }
        if not payload.get("next"):
            break
        page += 1

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def load_metadata(
    metadata_path: Optional[Path] = None,
    refresh: bool = False,
) -> Dict[str, dict]:
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    if refresh or not path.exists():
        return download_core_vertebrates_metadata(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def lookup_matrix(metadata: Dict[str, dict], matrix_id: str) -> dict:
    if matrix_id in metadata:
        return metadata[matrix_id]
    base_id = matrix_id.split(".")[0]
    for key, value in metadata.items():
        if key.startswith(base_id + "."):
            return value
    return {"name": matrix_id, "family": "", "class": ""}
