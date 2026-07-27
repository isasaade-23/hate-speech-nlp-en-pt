"""Fase 8 — transformer fine-tuning (runs on Colab GPU).

Reads the SAME frozen corpus parquet + a neural config, fine-tunes a HF model, and
writes metrics.json in the SAME schema as classical train.py (via hsc.evaluate), so the
classical-vs-neural comparison and all report code treat both families identically.

Local machines have no GPU/transformers; this module's heavy imports live inside the
function. A thin Colab notebook calls train_neural_from_config().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hsc.config import load_yaml, resolve
from hsc.evaluate import (
    best_threshold,
    bootstrap_macro_f1_ci,
    breakdown,
    classification_metrics,
    confusion,
)
from hsc.predictions import build_prediction_frame, save_predictions
from hsc.utils import ensure_dir, get_logger, read_json, write_json

log = get_logger("hsc.train_neural")


def _subset(df: pd.DataFrame, languages):
    if languages:
        df = df[df["language"].isin(languages)]
    return df


def _metrics_block(part: pd.DataFrame, y_pred, y_score, seed: int) -> dict:
    y = part["label"].values
    y_pred = np.asarray(y_pred)
    m = classification_metrics(y, y_pred, y_score)
    lo, hi = bootstrap_macro_f1_ci(y, y_pred, seed=seed)
    m["macro_f1_ci95"] = [round(lo, 4), round(hi, 4)]
    m["confusion"] = confusion(y, y_pred)
    pred_df = part[["language", "source_dataset", "label"]].copy()
    pred_df["pred"] = y_pred
    m["by_language"] = breakdown(pred_df, "label", "pred", "language").to_dict("records")
    m["by_source"] = breakdown(pred_df, "label", "pred", "source_dataset").to_dict("records")
    return m


def train_neural_from_config(
    config_path: str,
    corpus_path: str | None = None,
    policy: str | None = None,
    out_root: str = "models",
    metrics_root: str = "reports/metrics",
    seed: int | None = None,
) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    cfg = load_yaml(config_path)
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    # policy override lets one config train both strict and broad; corpus defaults to the
    # matching frozen parquet so the notebook cannot pair a policy with the wrong data.
    policy = policy or cfg.get("policy", "strict")
    if corpus_path is None:
        corpus_path = str(resolve(f"data/processed/corpus_{policy}.parquet"))
    text_col = cfg.get("text_column", "text_clean")
    ckpt = cfg["model"]["hf_checkpoint"]
    max_len = int(cfg["model"].get("max_length", 160))

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_parquet(corpus_path)
    if "split" not in df.columns:
        raise RuntimeError("corpus has no 'split' column; run `hsc split` first (Fase 4).")
    df = _subset(df, cfg.get("languages"))
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    log.info("neural train=%d val=%d test=%d [%s/%s]", len(tr), len(va), len(te), cfg["name"], policy)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_len)

    def to_ds(part):
        d = Dataset.from_pandas(part[[text_col, "label"]].reset_index(drop=True))
        d = d.map(tok, batched=True)
        return d.rename_column("label", "labels")

    ds_tr, ds_va, ds_te = to_ds(tr), to_ds(va), to_ds(te)

    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

    # class weights (balanced) for the imbalanced hate class
    tc = cfg["train"]
    if tc.get("class_weights") == "balanced":
        counts = tr["label"].value_counts().to_dict()
        n = len(tr)
        w = torch.tensor(
            [n / (2 * counts.get(0, 1)), n / (2 * counts.get(1, 1))], dtype=torch.float
        )
    else:
        w = None

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=w.to(outputs.logits.device) if w is not None else None
            )
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    model_id = f"{cfg['name']}_{policy}_s{seed}"
    out_dir = ensure_dir(Path(out_root) / model_id)
    args = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        num_train_epochs=tc.get("epochs", 4),
        per_device_train_batch_size=tc.get("batch_size", 16),
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=tc.get("grad_accum", 1),
        learning_rate=float(tc.get("lr", 2e-5)),
        weight_decay=float(tc.get("weight_decay", 0.01)),
        warmup_ratio=float(tc.get("warmup_ratio", 0.1)),
        fp16=bool(tc.get("fp16", True)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to=[],
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    def score(ds):
        logits = trainer.predict(ds).predictions
        return torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    # Tune the decision threshold on val exactly like the classical models (imbalance-aware),
    # so the classical-vs-neural comparison is fair — not 0.5 vs a tuned cut.
    val_score = score(ds_va)
    threshold = best_threshold(va["label"].values, val_score)

    result = {
        "model_id": model_id,
        "config": cfg["name"],
        "family": "neural",
        "policy": policy,
        "seed": seed,
        "hf_checkpoint": ckpt,
        "n_train": int(len(tr)),
        "threshold": round(float(threshold), 4),
        "splits": {},
    }
    scores = {"val": val_score, "test": score(ds_te)}
    for name, part in [("val", va), ("test", te)]:
        y_score = scores[name]
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
        result["splits"][name] = _metrics_block(part, y_pred, y_score, seed)
        # Per-example predictions -> McNemar + calibration include neural once copied back.
        save_predictions(model_id, name, build_prediction_frame(part, y_score, threshold))
        log.info(
            "%s [%s]: macro-F1=%.4f recall_hate=%.4f",
            model_id, name, result["splits"][name]["macro_f1"], result["splits"][name]["recall_hate"],
        )

    write_json(result, ensure_dir(Path(metrics_root)) / f"{model_id}.json")
    tokenizer.save_pretrained(out_dir / "hf")
    trainer.save_model(str(out_dir / "hf"))
    _register(result)
    log.info("saved neural model + metrics + predictions: %s", model_id)
    return result


def _register(result: dict) -> None:
    """Add/refresh the registry entry so report/analyze/bias/errors see neural models
    alongside classical ones (same schema as classical train.py)."""
    reg_path = resolve("models") / "registry.json"
    reg = read_json(reg_path) if reg_path.exists() else {}
    mid = result["model_id"]
    reg[mid] = {
        "path": f"models/{mid}/hf",
        "config": result["config"],
        "family": "neural",
        "policy": result["policy"],
        "seed": result["seed"],
        "hf_checkpoint": result.get("hf_checkpoint"),
        "threshold": result["threshold"],
        "val_macro_f1": result["splits"]["val"]["macro_f1"],
        "test_macro_f1": result["splits"]["test"]["macro_f1"],
    }
    write_json(reg, reg_path)
