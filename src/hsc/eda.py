"""Fase 6 — exploratory data analysis figures for the paper.

Reads the frozen corpus and writes figures to reports/figures and a composition table
to reports/tables. Uses a non-interactive backend so it runs headless (Colab/CI).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hsc.config import data_config, resolve
from hsc.utils import ensure_dir, get_logger, read_parquet

log = get_logger("hsc.eda")


def _save(fig, name: str):
    figs = ensure_dir(resolve("reports/figures"))
    path = figs / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def run_eda(policy: str = "strict") -> pd.DataFrame:
    data_cfg = data_config()
    df = read_parquet(resolve(data_cfg["paths"]["processed"]) / f"corpus_{policy}.parquet")

    # composition table: rows per (source, language, label) + hate rate
    comp = (
        df.groupby(["source_dataset", "language"])
        .agg(n=("label", "size"), hate=("label", "sum"))
        .reset_index()
    )
    comp["hate_rate"] = (comp["hate"] / comp["n"]).round(3)
    tdir = ensure_dir(resolve("reports/tables"))
    comp.to_csv(tdir / f"corpus_composition_{policy}.csv", index=False)
    log.info("composition [%s]:\n%s", policy, comp.to_string(index=False))

    # Fig 1: rows and hate rate by source
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    by_src = df.groupby("source_dataset")["label"].agg(["size", "mean"])
    by_src["size"].plot(kind="bar", ax=ax[0], color="#4C72B0")
    ax[0].set_title(f"Rows by source ({policy})")
    ax[0].set_ylabel("rows")
    (by_src["mean"] * 100).plot(kind="bar", ax=ax[1], color="#C44E52")
    ax[1].set_title("Hate rate by source (%)")
    ax[1].set_ylabel("% hate")
    fig.tight_layout()
    _save(fig, f"composition_by_source_{policy}.png")

    # Fig 2: text length distribution by language (chars)
    df = df.assign(nchars=df["text_clean"].str.len().clip(upper=400))
    fig, ax = plt.subplots(figsize=(8, 4))
    for lang, g in df.groupby("language"):
        ax.hist(g["nchars"], bins=40, alpha=0.5, label=lang, density=True)
    ax.set_title(f"Cleaned text length by language ({policy})")
    ax.set_xlabel("characters (clipped at 400)")
    ax.set_ylabel("density")
    ax.legend()
    _save(fig, f"length_by_language_{policy}.png")

    # Fig 3: label balance per language
    fig, ax = plt.subplots(figsize=(6, 4))
    ct = pd.crosstab(df["language"], df["label"])
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#55A868", "#C44E52"])
    ax.set_title(f"Label balance by language ({policy})")
    ax.set_ylabel("rows")
    ax.legend(["not-hate", "hate"])
    _save(fig, f"label_balance_by_language_{policy}.png")

    return comp
