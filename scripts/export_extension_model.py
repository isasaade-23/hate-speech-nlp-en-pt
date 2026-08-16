"""Deploy 1 — export the linear member for the browser extension.

Writes two artifacts into the extension repo:

  model/luciola_linear_v5.json
      word + char_wb vocabularies (gram -> index), per-block idf, dense coef,
      intercept, Platt calibration, threshold, the emoji -> :name: map used by
      the light cleaning profile, and provenance metadata.

  test/golden.json
      ~200 synthetic texts with the exact Python probability, for the JS
      parity test (no corpus text is shipped).

The JS runtime must replicate: clean light profile -> lowercase -> word 1-2gram
(token pattern \\b\\w\\w+\\b) and char_wb 3-5gram -> sublinear tf * idf ->
L2 PER BLOCK (FeatureUnion normalizes word and char independently) -> concat ->
dot(coef) + intercept -> sigmoid -> Platt sigmoid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "src")

from hsc.clean import clean_text  # noqa: E402
from hsc.config import data_config, resolve  # noqa: E402

OUT_ROOT = Path("C:/Users/Renato/luciola-extension")
MODEL_ID = "tfidf_logreg_strict_s42"


def emoji_map() -> dict[str, str]:
    import emoji as em

    out = {}
    for e, data in em.EMOJI_DATA.items():
        name = data.get("en")
        if name and name.startswith(":") and name.endswith(":"):
            out[e] = name.strip(":")
    return out


def golden_texts() -> list[str]:
    base = [
        "I love this community, everyone is so welcoming!",
        "i hate these faggots, they make me sick",
        "que vídeo incrível, parabéns pelo trabalho de vocês",
        "essas mulheres são todas umas vadias nojentas, deviam apanhar",
        "as mulheres da minha família são maravilhosas",
        "my gay friend is coming to dinner tonight",
        "RT @someone check this out https://example.com/x #news",
        "vc é um lixo de pessoa, some daqui",
        "great weather today ☀️ let's go outside 😄",
        "esse time joga muito 🔥🔥",
        "you are all subhuman trash and deserve nothing",
        "@fulano @beltrano olha isso www.site.com.br",
        "n1gger is a word i will not tolerate here",
        "eu odeio segunda-feira",
        "i hate mondays so much",
        "os imigrantes roubam nossos empregos, expulsem todos",
        "immigrants built this country, be kind",
        "kkkkkkkk que meme bom",
        "LOL this is hilarious",
        "morra seu desgraçado",
        "",
        "ok",
        "a b",
    ]
    variants = []
    for i, t in enumerate(base):
        variants.append(t.upper())
        variants.append(f"RT {t}")
        variants.append(f"{t} 😡")
        variants.append(f"@user{i} {t} #tag{i} http://t.co/{i}")
        variants.append(t + " " + t)
    seen, out = set(), []
    for t in base + variants:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:220]


def main() -> None:
    bundle = joblib.load(resolve("models") / MODEL_ID / "model.joblib")
    vec, est, cal = bundle["vectorizer"], bundle["estimator"], bundle["calibration"]
    coef = est.coef_.ravel().astype(float)
    blocks = []
    offset = 0
    for name, tv in vec.transformer_list:
        n = len(tv.vocabulary_)
        vocab = {g: int(i) for g, i in tv.vocabulary_.items()}
        blocks.append(
            {
                "name": name,
                "analyzer": tv.analyzer,
                "ngram_range": list(tv.ngram_range),
                "offset": offset,
                "size": n,
                "vocab": vocab,
                "idf": [round(float(x), 8) for x in tv.idf_],
            }
        )
        offset += n
    assert offset == coef.shape[0]

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    model = {
        "model": "luciola_linear_v5",
        "source_model_id": MODEL_ID,
        "git_sha": sha,
        "policy": "strict",
        "blocks": blocks,
        "coef": [round(float(x), 8) for x in coef],
        "intercept": float(est.intercept_[0]),
        "platt": {"a": float(cal["coef"]), "b": float(cal["intercept"])},
        "threshold": float(cal["threshold"]),
        "emoji": emoji_map(),
    }
    (OUT_ROOT / "model").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "test").mkdir(parents=True, exist_ok=True)
    mp = OUT_ROOT / "model" / "luciola_linear_v5.json"
    mp.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print(f"model json: {mp} ({mp.stat().st_size/1e6:.1f} MB, emoji map {len(model['emoji'])})")

    profile = data_config()["clean"]["profiles"]["light"]
    texts = golden_texts()
    cleaned = [clean_text(t, profile) for t in texts]
    raw = est.predict_proba(vec.transform(cleaned))[:, 1]
    a, b = model["platt"]["a"], model["platt"]["b"]
    prob = 1.0 / (1.0 + np.exp(-(a * raw + b)))
    golden = [
        {"text": t, "cleaned": c, "raw": round(float(r), 8), "prob": round(float(p), 8)}
        for t, c, r, p in zip(texts, cleaned, raw, prob)
    ]
    gp = OUT_ROOT / "test" / "golden.json"
    gp.write_text(json.dumps(golden, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"golden: {gp} ({len(golden)} texts)")


if __name__ == "__main__":
    main()
