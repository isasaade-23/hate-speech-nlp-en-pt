"""Regenerate the README showcase figures from reports/tables/*.csv, styled in the
project's visual identity. Run after the pipeline:  python assets/make_figures.py

Kept separate from reports/ (which is gitignored/regenerable) because these few figures
are committed portfolio assets embedded in the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DOCS_IMG = ROOT / "docs" / "img"  # mirror for the mkdocs site


def _save(fig, name: str) -> None:
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    for d in (ASSETS, DOCS_IMG):
        fig.savefig(d / name, dpi=150, bbox_inches="tight")

# Visual identity
SLATE = "#3D5A80"    # classical
CORAL = "#EE6C4D"    # neural / accent
AMBER = "#F4A261"
INK = "#1F3050"
BONE = "#F6F2EE"


def _style():
    plt.rcParams.update({
        "font.family": ["Lato", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor": "#C9D3E0",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E7E2DC",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def leaderboard_figure():
    df = pd.read_csv(ROOT / "reports/tables/leaderboard.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for ax, pol in zip(axes, ["strict", "broad"]):
        d = df[df.policy == pol].sort_values("test_macroF1")
        colors = [CORAL if f == "neural" else SLATE for f in d.family]
        labels = [m.replace(f"_{pol}_s42", "") for m in d.model_id]
        bars = ax.barh(labels, d.test_macroF1, color=colors, height=0.68)
        for b, v in zip(bars, d.test_macroF1):
            ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", ha="left", fontsize=9, color=INK)
        ax.set_title(f"{pol} policy", color=INK, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlim(0.55, 0.80)
        ax.tick_params(colors=INK, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_xlabel("test macro-F1", color=INK)
    axes[1].set_xlabel("test macro-F1", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORAL), plt.Rectangle((0, 0), 1, 1, color=SLATE)]
    fig.legend(handles, ["transformer (neural)", "classical"], loc="upper center",
               ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "leaderboard.png")
    plt.close(fig)
    print("wrote assets/leaderboard.png")


def transfer_figure():
    df = pd.read_csv(ROOT / "reports/tables/transfer_broad.csv")
    order = ["EN_all->PT (zero-shot)", "PT->EN_tweets (zero-shot)",
             "EN_memes->EN_tweets", "EN_tweets->EN_memes"]
    df = df[df.experiment.isin(order)]
    piv = df.pivot_table(index="experiment", columns="features", values="macro_f1").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = range(len(piv))
    h = 0.36
    ax.barh([i + h / 2 for i in y], piv["tfidf"], height=h, color=SLATE, label="TF-IDF (word)")
    ax.barh([i - h / 2 for i in y], piv["sbert"], height=h, color=CORAL, label="SBERT (multilingual)")
    ax.set_yticks(list(y))
    ax.set_yticklabels([e.replace("_", " ").replace("->", "→") for e in piv.index], fontsize=9, color=INK)
    ax.axvline(0.5, color="#B8B0A8", lw=1, ls="--")
    ax.set_xlabel("test macro-F1 (train slice → test slice)", color=INK)
    ax.set_title("Cross-lingual & cross-domain transfer: TF-IDF collapses, SBERT transfers",
                 color=INK, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlim(0.35, 0.75)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    _save(fig, "transfer.png")
    plt.close(fig)
    print("wrote assets/transfer.png")


if __name__ == "__main__":
    _style()
    leaderboard_figure()
    transfer_figure()
