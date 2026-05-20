"""Country name normalisation for consistent matching across datasets."""

import re
import unicodedata

# ---------------------------------------------------------------------------
# String-level normalisation
# ---------------------------------------------------------------------------

_AMP_RE    = re.compile(r"\s*&\s*")          # & → and
_STRIP_RE  = re.compile(r"[^a-z0-9\s]")      # strip punctuation
_WS_RE     = re.compile(r"\s+")


def _base_normalize(name: str) -> str:
    """Unicode → ASCII, lowercase, strip punctuation, collapse whitespace."""
    if not name or not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = _AMP_RE.sub(" and ", name)
    name = _STRIP_RE.sub(" ", name)
    name = _WS_RE.sub(" ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Canonical alias map
# Applied AFTER base_normalize so all keys/values are already base-normalised.
# Maps non-canonical forms → canonical form.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # CIS / Eastern Europe
    "viet nam":                             "vietnam",
    "turkiye":                              "turkey",
    # Africa
    "dem rep congo":                        "democratic republic of the congo",
    "dem rep of congo":                     "democratic republic of the congo",
    "democratic republic of congo":         "democratic republic of the congo",
    "rep of the congo":                     "republic of the congo",
    "rep of congo":                         "republic of the congo",
    "central african rep":                  "central african republic",
    "gambia":                               "the gambia",
    "eswatini":                             "swaziland",
    # Middle East / Asia
    "palestine state of":                   "palestine",
    "state of palestine":                   "palestine",
    "taiwan province of china":             "taiwan",
    "hong kong sar":                        "hong kong",
    "macau":                                "macao",
    "korea republic of":                    "south korea",
    "republic of korea":                    "south korea",
    "korea south":                          "south korea",
    # Americas
    "usa":                                  "united states",
    "us":                                   "united states",
    "united states of america":             "united states",
    # Europe
    "holy see":                             "vatican city",
    "vatican":                              "vatican city",
    "czech republic":                       "czechia",
    "uk":                                   "united kingdom",
    "great britain":                        "united kingdom",
    # Sao Tome (& vs and both normalise to "and" via _AMP_RE above)
    # Bosnia (& vs and) same
    # Channel Islands — leave as-is (Guernsey, Jersey distinct in SNL)
    # French overseas territories — leave as-is (handled by --relax-country)
    "bvi":                                  "british virgin islands",
    "timor leste":                          "east timor",
    "east timor":                           "timor leste",   # pick one canonical form below
}

# Pick single canonical form for bidirectional aliases
_CANONICAL: dict[str, str] = {}
for _k, _v in _ALIASES.items():
    _CANONICAL[_k] = _v


def normalize_country(name: str) -> str:
    """Return a canonical, normalised country key suitable for index lookups.

    Both sides of a match should be passed through this function before
    comparison so that spelling variants map to the same key.
    """
    key = _base_normalize(name)
    # Apply alias substitutions (iterating handles chained aliases, max 3 hops)
    for _ in range(3):
        mapped = _CANONICAL.get(key)
        if mapped is None or mapped == key:
            break
        key = mapped
    return key
