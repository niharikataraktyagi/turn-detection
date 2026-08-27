"""AMI corpus metadata: channel -> speaker mapping, and real speaker
identity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

MEETING_RE = re.compile(r'<meeting\b[^>]*observation="(?P<obs>[^"]+)"[^>]*>(?P<body>.*?)</meeting>', re.S)
SPEAKER_RE = re.compile(
    r'<speaker\b[^>]*channel="(?P<ch>\d+)"[^>]*nxt_agent="(?P<agent>[A-Z])"'
    r'[^>]*global_name="(?P<gname>[^"]*)"[^>]*(?:role="(?P<role>[^"]*)")?'
)


@dataclass
class SpeakerInfo:
    channel: int
    agent: str        # A/B/C/D — matches the words-file naming
    global_name: str  # real person across meetings — use this for splits
    role: str = ""


def parse_meetings(path: str) -> Dict[str, List[SpeakerInfo]]:
    with open(path, encoding="iso-8859-1", errors="replace") as fh:
        raw = fh.read()

    out: Dict[str, List[SpeakerInfo]] = {}
    for m in MEETING_RE.finditer(raw):
        speakers = [
            SpeakerInfo(int(s.group("ch")), s.group("agent"),
                        s.group("gname"), s.group("role") or "")
            for s in SPEAKER_RE.finditer(m.group("body"))
        ]
        if speakers:
            out[m.group("obs")] = sorted(speakers, key=lambda s: s.channel)
    return out


def agent_to_channel(meetings: Dict[str, List[SpeakerInfo]], meeting: str) -> Dict[str, int]:
    return {s.agent: s.channel for s in meetings.get(meeting, [])}


def agent_to_global(meetings: Dict[str, List[SpeakerInfo]], meeting: str) -> Dict[str, str]:
    return {s.agent: s.global_name for s in meetings.get(meeting, [])}
