"""Regulatory Context Card (RCC) construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from r3lm.jaspar_metadata import load_metadata, lookup_matrix
from r3lm.motifs import read_meme, scan

MOTIF_HEADER = "Identified Motifs:\n"
DNA_HEADER = "\nDNA Sequence:\n"
DEFAULT_P_THRESHOLD = 5e-5


@dataclass
class MotifHit:
    matrix_id: str
    name: str
    family: str
    score: float
    strand: str
    start: int
    end: int
    pval: float

    @property
    def motif_line(self) -> str:
        return (
            f"- {self.name} (Family: {self.family}, Score: {self.score:.2f}, "
            f"Strand: {self.strand}, Pos: {self.start}-{self.end})"
        )


def gc_content(sequence: str) -> float:
    sequence = sequence.upper()
    if not sequence:
        return 0.0
    return round((sequence.count("G") + sequence.count("C")) / len(sequence), 3)


def normalize_sequence(sequence: str, expected_length: Optional[int] = 200) -> str:
    sequence = sequence.strip().upper()
    if "N" in sequence:
        raise ValueError("Sequences containing N are not supported.")
    if expected_length is not None and len(sequence) != expected_length:
        raise ValueError(f"Expected sequence length {expected_length}, got {len(sequence)}.")
    return sequence


def _intervals_overlap(a: MotifHit, b: MotifHit) -> bool:
    return not (a.end < b.start or b.end < a.start)


def deduplicate_overlapping_families(hits: Sequence[MotifHit]) -> List[MotifHit]:
    """Keep the highest-scoring hit when same-family motifs overlap."""
    kept: List[MotifHit] = []
    for hit in sorted(hits, key=lambda item: (-item.score, item.start)):
        if any(
            hit.family and hit.family == other.family and _intervals_overlap(hit, other)
            for other in kept
        ):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda item: (-item.score, item.start))


def derive_grammar_tags(hits: Sequence[MotifHit], gc: float, window: int = 25) -> List[str]:
    """Lightweight grammar heuristics used in paper-facing RCC analysis."""
    tags: List[str] = []
    if gc >= 0.55:
        tags.append("high_gc_open_chromatin_prior")
    elif gc <= 0.40:
        tags.append("low_gc_closed_chromatin_prior")

    if len(hits) >= 3:
        positions = sorted((hit.start + hit.end) / 2 for hit in hits)
        for idx in range(len(positions) - 2):
            if positions[idx + 2] - positions[idx] <= window:
                tags.append(f"motif_cluster_within_{window}bp")
                break

    family_counts: Dict[str, int] = {}
    for hit in hits:
        if hit.family:
            family_counts[hit.family] = family_counts.get(hit.family, 0) + 1
    repeated = [family for family, count in family_counts.items() if count >= 3]
    if repeated:
        tags.append("repeated_family:" + repeated[0])
    return tags


def hits_from_scan(
    sites_df: pd.DataFrame,
    metadata: Dict[str, dict],
) -> List[MotifHit]:
    hits: List[MotifHit] = []
    for row in sites_df.itertuples():
        matrix_id = str(getattr(row, "Matrix_id", ""))
        meta = lookup_matrix(metadata, matrix_id)
        hits.append(
            MotifHit(
                matrix_id=matrix_id,
                name=meta["name"],
                family=meta.get("family", ""),
                score=float(row.score),
                strand=str(row.strand),
                start=int(row.start),
                end=int(row.end),
                pval=float(row.pval),
            )
        )
    return hits


def build_rcc_prompt(
    sequence: str,
    cell_line: str,
    hits: Sequence[MotifHit],
    include_grammar_tags: bool = False,
) -> str:
    sequence = normalize_sequence(sequence)
    gc = gc_content(sequence)
    lines = [
        "Context:",
        f"- Cell Type: {cell_line}",
        f"- GC Content: {gc:.3f}",
    ]
    if include_grammar_tags:
        grammar_tags = derive_grammar_tags(hits, gc)
        if grammar_tags:
            lines.append(f"- Grammar Tags: {', '.join(grammar_tags)}")
    lines.extend(["", MOTIF_HEADER.rstrip("\n")])
    if hits:
        lines.extend(hit.motif_line for hit in hits)
    lines.extend(
        [
            "",
            "DNA Sequence:",
            f"{sequence} (Total Length: {len(sequence)})",
        ]
    )
    return "\n".join(lines)


def build_rcc_for_sequences(
    sequences: pd.DataFrame,
    cell_line: str,
    meme_file: str,
    metadata: Optional[Dict[str, dict]] = None,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    include_grammar_tags: bool = False,
    seq_col: str = "Sequence",
) -> pd.DataFrame:
    if seq_col not in sequences.columns:
        raise ValueError(f"Missing `{seq_col}` column in input table.")
    if metadata is None:
        metadata = load_metadata()

    frame = sequences.copy()
    frame[seq_col] = frame[seq_col].astype(str)
    frame.index = frame.index.astype(str)
    frame["Sequence"] = frame[seq_col].map(lambda seq: normalize_sequence(seq))

    scan_input = frame[[seq_col]].rename(columns={seq_col: "Sequence"})
    scan_input.index = frame.index
    motifs, bg = read_meme(meme_file)
    sites = scan(scan_input, motifs, bg, threshold=p_threshold)

    prompts = []
    for seq_id, sequence in frame["Sequence"].items():
        seq_hits = hits_from_scan(sites.loc[sites.index == seq_id], metadata)
        seq_hits = deduplicate_overlapping_families(seq_hits)
        prompts.append(
            build_rcc_prompt(
                sequence=sequence,
                cell_line=cell_line,
                hits=seq_hits,
                include_grammar_tags=include_grammar_tags,
            )
        )

    frame["rcc_prompt"] = prompts
    return frame
