"""AMI → Inter-Pausal Units (IPUs) with turn-taking labels."""
from __future__ import annotations

import re
import glob
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- constants, chosen to match the paper -----------------------------------
PAUSE_MIN = 0.150      # IPU boundary threshold (Kelterer et al.: pauses > 150 ms)
WINDOW = 0.600         # prosodic analysis window at IPU end (they tuned 0.6 > 0.8 > 1.0)
BACKCHANNEL_MAX_DUR = 0.80   # other-speaker speech shorter than this, followed by
BACKCHANNEL_MAX_WORDS = 3    # the original speaker resuming, = backchannel, not a floor change

TOKEN_RE = re.compile(
    r'<(?P<tag>w|vocalsound|disfmarker|gap)\b(?P<attrs>[^>]*?)(?:/>|>(?P<text>.*?)</(?P=tag)>)',
    re.S,
)
ATTR_RE = re.compile(r'(\w[\w:]*)="([^"]*)"')


@dataclass
class Token:
    start: float
    end: float
    text: str            # "" for non-lexical tokens
    kind: str            # word | punc | vocal | disf | gap
    trunc: bool = False  # truncated word -> speaker cut themselves off
    vtype: str = ""      # laugh / other

    @property
    def has_duration(self) -> bool:
        return self.end > self.start


@dataclass
class IPU:
    meeting: str
    speaker: str
    start: float
    end: float
    tokens: List[Token] = field(default_factory=list)

    # filled in by label_ipus()
    pause_dur: float = 0.0
    floor_changes: bool = False
    next_speaker: Optional[str] = None
    overlap_at_end: bool = False   # someone else speaking inside the analysis window
    ends_in_laugh: bool = False
    ends_truncated: bool = False
    is_question: bool = False      # transcriber '?' — used for EXCLUSION only, see note

    @property
    def text(self) -> str:
        """Orthographic text of the IPU, punctuation attached without a space."""
        out = []
        for t in self.tokens:
            if t.kind == "punc":
                if out:
                    out[-1] = out[-1] + t.text
            elif t.kind == "word":
                out.append(t.text)
        return " ".join(out)

    @property
    def n_words(self) -> int:
        return sum(1 for t in self.tokens if t.kind == "word")


def _unescape(s: str) -> str:
    return (s.replace("&#39;", "'").replace("&amp;", "&")
             .replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")).strip()


def parse_words_file(path: str) -> List[Token]:
    """Parse one <meeting>.<speaker>.words.xml into time-ordered tokens."""
    with open(path, encoding="iso-8859-1") as fh:
        raw = fh.read()

    tokens: List[Token] = []
    for m in TOKEN_RE.finditer(raw):
        attrs = dict(ATTR_RE.findall(m.group("attrs")))
        if "starttime" not in attrs or "endtime" not in attrs:
            continue  # a handful of malformed entries
        start, end = float(attrs["starttime"]), float(attrs["endtime"])
        tag = m.group("tag")

        if tag == "w":
            kind = "punc" if attrs.get("punc") == "true" else "word"
            tok = Token(start, end, _unescape(m.group("text") or ""), kind,
                        trunc=attrs.get("trunc") == "true")
        elif tag == "vocalsound":
            tok = Token(start, end, "", "vocal", vtype=attrs.get("type", ""))
        elif tag == "disfmarker":
            tok = Token(start, end, "", "disf")
        else:  # gap
            tok = Token(start, end, "", "gap")
        tokens.append(tok)

    tokens.sort(key=lambda t: (t.start, t.end))
    return tokens


def segment_ipus(meeting: str, speaker: str, tokens: List[Token]) -> List[IPU]:
    """Split one speaker's token stream into IPUs at silences > PAUSE_MIN."""
    ipus: List[IPU] = []
    cur: Optional[IPU] = None
    last_end: Optional[float] = None

    for tok in tokens:
        if not tok.has_duration:
            if cur is not None:
                cur.tokens.append(tok)
            continue

        if cur is None or (last_end is not None and tok.start - last_end > PAUSE_MIN):
            cur = IPU(meeting, speaker, tok.start, tok.end)
            ipus.append(cur)

        cur.tokens.append(tok)
        cur.end = max(cur.end, tok.end)
        last_end = cur.end

    return ipus


def load_meeting(ann_root: str, meeting: str) -> Dict[str, List[IPU]]:
    """All speakers of one meeting -> {speaker: [IPU, ...]}."""
    out: Dict[str, List[IPU]] = {}
    for path in sorted(glob.glob(os.path.join(ann_root, "words", f"{meeting}.*.words.xml"))):
        speaker = os.path.basename(path).split(".")[1]
        toks = parse_words_file(path)
        if toks:
            out[speaker] = segment_ipus(meeting, speaker, toks)
    return out


def label_ipus(by_speaker: Dict[str, List[IPU]]) -> List[IPU]:
    """Answer question (a) — did the floor change? — for every IPU end, using the
    full multi-speaker timeline.
    """
    # flat, time-sorted list of every IPU in the meeting
    all_ipus = sorted((i for lst in by_speaker.values() for i in lst),
                      key=lambda i: i.start)

    for spk, ipus in by_speaker.items():
        for idx, ipu in enumerate(ipus):
            nxt_own = ipus[idx + 1].start if idx + 1 < len(ipus) else float("inf")

            # earliest speech by anyone else at or after this IPU's end
            others = [o for o in all_ipus
                      if o.speaker != spk and o.start >= ipu.end - 1e-6]
            nxt_other_ipu = others[0] if others else None
            nxt_other = nxt_other_ipu.start if nxt_other_ipu else float("inf")

            floor_changes = nxt_other < nxt_own
            if floor_changes and nxt_other_ipu is not None:
                # backchannel test: short, and the original speaker comes back after it
                short = ((nxt_other_ipu.end - nxt_other_ipu.start) < BACKCHANNEL_MAX_DUR
                         and nxt_other_ipu.n_words <= BACKCHANNEL_MAX_WORDS)
                if short and nxt_own < float("inf") and nxt_own < nxt_other_ipu.end + 2.0:
                    floor_changes = False

            ipu.floor_changes = floor_changes
            ipu.next_speaker = (nxt_other_ipu.speaker
                                if floor_changes and nxt_other_ipu else spk)
            ipu.pause_dur = min(nxt_own, nxt_other) - ipu.end

            # overlap inside the prosodic analysis window corrupts the acoustics,
            # and the paper excludes turn-changes with overlapping speech at IPU end
            w0 = ipu.end - WINDOW
            ipu.overlap_at_end = any(
                o.speaker != spk and o.start < ipu.end and o.end > w0
                for o in all_ipus
            )

            tail = [t for t in ipu.tokens if t.kind in ("word", "vocal")]
            ipu.ends_in_laugh = bool(tail) and tail[-1].kind == "vocal" and tail[-1].vtype == "laugh"
            ipu.ends_truncated = any(t.trunc for t in ipu.tokens[-2:])
            ipu.is_question = any(t.kind == "punc" and "?" in t.text for t in ipu.tokens[-3:])

    return all_ipus


def list_meetings(ann_root: str) -> List[str]:
    names = {os.path.basename(p).split(".")[0]
             for p in glob.glob(os.path.join(ann_root, "words", "*.words.xml"))}
    return sorted(names)
