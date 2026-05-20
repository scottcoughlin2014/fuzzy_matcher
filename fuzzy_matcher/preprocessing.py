"""Text normalisation helpers for bank name fuzzy matching."""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Pre-normalisation: remove noise before any text comparison
# ---------------------------------------------------------------------------

# Stock exchange tickers appended by SNL, e.g. "(OB:NONG)", "(NASE:BKG)", "(BRVM:BOAN)"
_TICKER_RE = re.compile(r"\s*\([A-Z]{1,6}:[A-Z0-9.]+\)")

# Fancy/angle quotation marks that confuse tokenisation
_FANCY_QUOTES_RE = re.compile(r'[«»„""\u2018\u2019\u201c\u201d\u00ab\u00bb]')

# Hyphens used as word separators (e.g. "ECOBANK-Mali", "co-operative" handled separately)
# Only replace hyphens surrounded by word characters (not leading/trailing dashes)
_WORD_HYPHEN_RE = re.compile(r"(?<=\w)-(?=\w)")

# ---------------------------------------------------------------------------
# Legal-entity suffix normalisation (long form → canonical short form)
# Applied after unicode → ASCII so accented variants (e.g. Société Anonyme) are caught
# ---------------------------------------------------------------------------
_SUFFIX_MAP = [
    # English
    (r"\blimited liability company\b", "llc"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\bpublic limited company\b", "plc"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcooperative\b", "coop"),
    (r"\bnational association\b", "na"),
    (r"\bfederal savings bank\b", "fsb"),
    # French / Romance
    (r"\bsociete anonyme\b", "sa"),           # covers Société Anonyme post-ASCII
    (r"\bsociedad anonima\b", "sa"),
    (r"\bsociete en commandite\b", "sca"),
    (r"\bsociete a responsabilite limitee\b", "sarl"),
    (r"\bsociedad de responsabilidad limitada\b", "srl"),
    (r"\bsociedad limitada\b", "sl"),
    # German / Dutch
    (r"\baktiengesellschaft\b", "ag"),
    (r"\bgesellschaft mit beschraenkter haftung\b", "gmbh"),
    (r"\bnaamloze vennootschap\b", "nv"),
    (r"\bbesloten vennootschap\b", "bv"),
    # CIS / Eastern European
    (r"\bopen joint stock\b", "ojsc"),
    (r"\bclosed joint stock\b", "cjsc"),
    (r"\bprivate joint stock\b", "pjsc"),
    (r"\bpublic joint stock\b", "pjsc"),
    (r"\bjoint stock company\b", "jsc"),
    (r"\bjoint stock bank\b", "jsb"),
    (r"\bprivate joint-stock\b", "pjsc"),
    (r"\bpublic joint-stock\b", "pjsc"),
    # Misc
    (r"\bcommercial bank\b", "cb"),
]
_SUFFIX_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _SUFFIX_MAP]

# Characters to strip after all substitutions (keep alphanumeric + space)
_STRIP_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise a bank name for fuzzy comparison.

    Steps:
    1. Strip stock-exchange ticker suffixes, e.g. "(OB:NONG)".
    2. Strip fancy/angle quotation marks «» etc.
    3. Replace word-internal hyphens with spaces ("ECOBANK-Mali" → "ECOBANK Mali").
    4. Unicode NFKD → ASCII approximation (é→e, ö→o, etc.).
    5. Lowercase.
    6. Expand/collapse known legal-entity suffixes to canonical short forms.
    7. Strip remaining punctuation.
    8. Collapse whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    text = _TICKER_RE.sub("", text)
    text = _FANCY_QUOTES_RE.sub("", text)
    text = _WORD_HYPHEN_RE.sub(" ", text)
    # Unicode → ASCII
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
