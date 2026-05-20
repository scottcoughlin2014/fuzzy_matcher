"""Text normalisation helpers for bank name fuzzy matching."""

import re
import unicodedata

# Common legal-entity suffix substitutions (long form → short form)
_SUFFIX_MAP = [
    (r"\blimited liability company\b", "llc"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\bpublic limited company\b", "plc"),
    (r"\bjoint stock company\b", "jsc"),
    (r"\bjoint stock bank\b", "jsb"),
    (r"\bsociete anonyme\b", "sa"),
    (r"\bsociedad anonima\b", "sa"),
    (r"\baktiengesellschaft\b", "ag"),
    (r"\bgesellschaft mit beschraenkter haftung\b", "gmbh"),
    (r"\bopen joint stock\b", "ojsc"),
    (r"\bclosed joint stock\b", "cjsc"),
    (r"\bprivate joint stock\b", "pjsc"),
    (r"\bpublic joint stock\b", "pjsc"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcooperative\b", "coop"),
    (r"\bco-operative\b", "coop"),
    (r"\bnational association\b", "na"),
    (r"\bfederal savings bank\b", "fsb"),
    (r"\bcommercial bank\b", "cb"),
]
_SUFFIX_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _SUFFIX_MAP]

# Characters to strip (keep alphanumeric + space)
_STRIP_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise a bank name for fuzzy comparison.

    Steps:
    1. Unicode NFC → ASCII approximation (e.g. é → e).
    2. Lowercase.
    3. Expand/collapse known legal suffixes.
    4. Strip punctuation.
    5. Collapse whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    # NFKD + drop combining chars → ASCII-ish
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    for pattern, replacement in _SUFFIX_RE:
        text = pattern.sub(replacement, text)
    text = _STRIP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def parse_lei_set(lei_value) -> set[str]:
    """Parse a (possibly pipe-separated) LEI string into a set of clean LEI strings."""
    if not lei_value or not isinstance(lei_value, str):
        return set()
    return {x.strip() for x in lei_value.split("|") if x.strip()}
