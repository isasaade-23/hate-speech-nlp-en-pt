"""Bilingual (EN/PT) function-word stop lists — ablation for TF-IDF word features.

Scope: articles, prepositions, pronouns, and common conjunctions, per the requested
"remove prepositions/pronouns" experiment. **Negations are deliberately excluded**
(removing not / no / não / nunca inverts meaning and is known to hurt sentiment-style
tasks). Applied only to the WORD analyzer; char n-grams keep everything. Single-character
tokens are already dropped by the default tokenizer, so 1-char words are omitted here.
"""

from __future__ import annotations

_EN = {
    # articles / determiners
    "the", "an", "this", "that", "these", "those", "such",
    # pronouns
    "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what",
    # prepositions
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "over", "under",
    "within", "without", "upon", "among", "across", "behind", "beyond", "per", "via",
    "onto", "off", "out", "up", "down",
    # conjunctions
    "and", "or", "but", "so", "because", "as", "if", "than", "while", "whereas", "yet",
}

_PT = {
    # artigos / determinantes
    "os", "as", "um", "uma", "uns", "umas",
    # preposições e contrações
    "de", "em", "para", "pra", "por", "com", "sem", "sobre", "sob", "entre", "até", "ate",
    "após", "apos", "ante", "contra", "desde", "ao", "aos", "às", "do", "da", "dos", "das",
    # "no" omitted on purpose: PT prep (em+o) but also the EN negation "no" — removing it
    # would invert meaning in English. Keep the negation; the PT contraction loss is trivial.
    "na", "nos", "nas", "pelo", "pela", "pelos", "pelas", "num", "numa", "nuns", "numas",
    "dum", "duma", "deste", "desta", "destes", "destas", "disto", "desse", "dessa", "disso",
    "daquele", "daquela", "daquilo", "neste", "nesta", "nisto", "nesse", "nessa", "nisso",
    "naquele", "naquela", "naquilo",
    # pronomes
    "eu", "me", "mim", "comigo", "tu", "te", "ti", "contigo", "voce", "você", "voces", "vocês",
    "ele", "ela", "eles", "elas", "lhe", "lhes", "se", "si", "consigo", "vos", "vós", "nós",
    "meu", "minha", "meus", "minhas", "teu", "tua", "teus", "tuas", "seu", "sua", "seus", "suas",
    "nosso", "nossa", "nossos", "nossas", "vosso", "vossa", "vossos", "vossas",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas", "isto", "isso", "aquilo",
    "que", "quem", "qual", "quais", "cujo", "cuja", "cujos", "cujas", "quanto", "quanta",
    # conjunções
    "ou", "mas", "porque", "pois", "como", "quando", "enquanto", "embora", "porém", "porem",
    "contudo", "todavia", "logo", "portanto",
}

_MAP = {"en": _EN, "pt": _PT, "enpt": _EN | _PT}


def resolve_stopwords(spec):
    """spec: a list (used as-is), or one of 'en' / 'pt' / 'enpt' (curated lists)."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, set)):
        return sorted(spec)
    key = str(spec).lower()
    if key not in _MAP:
        raise ValueError(f"unknown stop_words spec {spec!r}; use en, pt, enpt, or a list")
    return sorted(_MAP[key])
