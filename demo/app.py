"""Gradio web demo. Run:  python demo/app.py  (opens a local, shareable UI).

Loads the best registry model via hsc.inference and classifies EN/PT text.
"""

from __future__ import annotations

import gradio as gr

from hsc.inference import get_classifier

clf = get_classifier()

EXAMPLES = [
    "I love this community, everyone is so welcoming!",
    "Você é uma pessoa incrível, obrigado por tudo.",
    "bom dia, tudo certo por aí?",
    "Have a wonderful day everyone.",
]


def classify(text: str):
    if not text or not text.strip():
        return {}, "", ""
    p = clf.predict(text)
    # Gradio Label wants {class: confidence}
    p_hate = p["score"]
    scores = {"hate": p_hate, "not_hate": 1.0 - p_hate}
    lang = f'{p["language"]["detected"]} (conf {p["language"]["confidence"]})'
    return scores, lang, p["model_version"]


with gr.Blocks(title="Bilingual Hate-Speech Classifier") as demo:
    gr.Markdown(
        "# Bilingual Hate-Speech Classifier (EN / PT)\n"
        "Probabilistic research tool. **Not** a moderation verdict."
    )
    inp = gr.Textbox(label="Text", lines=3, placeholder="Type English or Portuguese text...")
    btn = gr.Button("Classify", variant="primary")
    out_label = gr.Label(label="Prediction", num_top_classes=2)
    out_lang = gr.Textbox(label="Detected language")
    out_model = gr.Textbox(label="Model")
    btn.click(classify, inputs=inp, outputs=[out_label, out_lang, out_model])
    gr.Examples(EXAMPLES, inputs=inp)


if __name__ == "__main__":
    demo.launch()
