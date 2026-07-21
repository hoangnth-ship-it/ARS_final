"""Linguistic features from transcript (Sec 3.3).  Fillers KEPT (disfluency signal)."""
from __future__ import annotations

import re
from collections import OrderedDict

FILLERS = {"um", "uh", "erm", "ah", "eh", "hmm", "mm", "er"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their"}


def extract(text: str) -> OrderedDict:
    t = (text or "").lower()
    words = re.findall(r"[a-z']+", t)
    n = len(words)
    out = OrderedDict()
    out["n_words"] = float(n)
    out["type_token_ratio"] = float(len(set(words)) / n) if n else 0.0
    out["filler_ratio"] = float(sum(w in FILLERS for w in words) / n) if n else 0.0
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    out["avg_sentence_len"] = float(n / len(sents)) if sents else float(n)
    n_pron = sum(w in PRONOUNS for w in words)
    n_noun_proxy = sum(1 for w in words if w not in PRONOUNS and w not in FILLERS)
    out["pronoun_ratio"] = float(n_pron / n) if n else 0.0
    out["pronoun_noun_ratio"] = float(n_pron / n_noun_proxy) if n_noun_proxy else 0.0
    # immediate word repetitions (disfluency)
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b)
    out["repetition_count"] = float(reps)
    out["repetition_ratio"] = float(reps / n) if n else 0.0
    return out
