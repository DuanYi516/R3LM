"""FIMO motif scanning utilities for RCC construction."""

from collections import defaultdict

import anndata
import pandas as pd
from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO


def read_meme(meme_file):
    motifs = []
    motiffile = MotifFile(meme_file)

    while True:
        motif = motiffile.read()
        if motif is None:
            break
        motifs.append(motif)

    print(f"Read {len(motifs)} motifs")
    return motifs, motiffile.background


def scan(seq_df, motifs, bg, threshold=0.001):
    sequences = [
        Sequence(row.Sequence, name=row.Index.encode())
        for row in seq_df.itertuples()
    ]

    d = defaultdict(list)
    fimo = FIMO(both_strands=True, threshold=threshold)

    for motif in motifs:
        match = fimo.score_motif(motif, sequences, bg).matched_elements
        for m in match:
            d["Matrix_id"].append(motif.name.decode())
            d["SeqID"].append(m.source.accession.decode())
            d["strand"].append(m.strand)
            d["score"].append(m.score)
            d["pval"].append(m.pvalue)
            d["qval"].append(m.qvalue)
            if m.strand == "-":
                d["start"].append(m.stop)
                d["end"].append(m.start)
            else:
                d["start"].append(m.start)
                d["end"].append(m.stop)

    return pd.DataFrame(d).set_index("SeqID")


def calculate_motif_counts(sites_df):
    return anndata.AnnData(
        pd.pivot_table(
            sites_df, values="start", index="SeqID", columns="Matrix_id", aggfunc="count"
        ).fillna(0)
    )
