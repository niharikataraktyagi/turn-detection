"""Question (b) of the 2x2: was the utterance SYNTACTICALLY COMPLETE at this
pause?
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

# Tokens that, when they END an utterance, project more talk. These are the
# "points of maximum grammatical control" of the paper's title: the grammar is
# holding the floor open.
DANGLING_POS = {"ADP", "CCONJ", "SCONJ", "DET", "PART", "AUX"}

# Words that are complete turns on their own — an overt predicate is not
# required because it is directly recoverable from the prior turn.
ACKNOWLEDGEMENTS = {
    "okay", "ok", "yeah", "yes", "yep", "no", "nope", "right", "sure", "true",
    "exactly", "alright", "mm-hmm", "mmhmm", "mm", "hmm", "uh-huh", "aha",
    "definitely", "absolutely", "agreed", "fine", "good", "great", "cool",
    "thanks", "thank you", "sorry", "oh", "ah", "wow", "indeed", "correct",
}

# Fillers that carry no syntactic weight; stripped from the tail before judging.
FILLERS = {"um", "uh", "erm", "eh", "mm", "hmm", "er", "like", "y'know", "you know"}

_PUNCT_RE = re.compile(r"[^\w\s'-]")


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            "spaCy model missing. Run:  python -m spacy download en_core_web_sm"
        ) from exc


def normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def strip_trailing_fillers(text: str) -> str:
    toks = text.split()
    while toks and toks[-1].lower().strip(".,?!") in FILLERS:
        toks.pop()
    return " ".join(toks)


def is_syntactically_complete(
    text: str,
    ends_truncated: bool = False,
    ends_in_disfluency: bool = False,
) -> Optional[bool]:
    """True = complete clause / recoverable predicate -> com-hold or change False
    = incomplete, grammar projects more talk -> in-hold or trail-off None =.
    """
    # 1. A truncated word ("the compa-") is proof the speaker cut themselves off.
    if ends_truncated:
        return False

    cleaned = strip_trailing_fillers(text.strip())
    if not cleaned:
        return None

    # 2. Trailing filler or a disfluency marker right at the pause is an explicit
    #    floor-holding device — the speaker is signalling "I'm still going".
    if ends_in_disfluency and cleaned != text.strip():
        return False

    flat = normalise(cleaned)
    if not flat:
        return None

    # 3. Standalone acknowledgements: complete by recoverable predicate.
    if flat in ACKNOWLEDGEMENTS:
        return True
    words = flat.split()
    if len(words) <= 2 and all(w in ACKNOWLEDGEMENTS for w in words):
        return True

    doc = _nlp()(cleaned)
    if len(doc) == 0:
        return None

    # 4. Dangling function word at the end -> incomplete, regardless of what
    #    came before. "...and I went to the" has a full clause in it and is
    #    still obviously unfinished.
    last = next((t for t in reversed(doc) if not t.is_punct and not t.is_space), None)
    if last is None:
        return None
    if last.pos_ in DANGLING_POS:
        # an AUX is only dangling if nothing depends on it ("I was" vs "I was late")
        if last.pos_ != "AUX" or not list(last.children):
            return False

    # 5. Otherwise require a clause: a predicate with a subject, or an imperative.
    for sent in reversed(list(doc.sents)):
        root = sent.root
        if root.pos_ in ("VERB", "AUX"):
            has_subject = any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj")
                              for c in root.children)
            imperative = root.tag_ == "VB" and not has_subject
            if has_subject or imperative:
                return True
        # copular / nominal predicates: "that's fine", "no problem"
        if any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children):
            return True
        break  # only judge the final sentence

    return False


def label_2x2(floor_changes: bool, complete: Optional[bool]) -> Optional[str]:
    """Cross the two independent axes into the Schuppler / Kelterer taxonomy."""
    if complete is None:
        return None
    if floor_changes:
        return "change" if complete else "trail-off"
    return "com-hold" if complete else "in-hold"
