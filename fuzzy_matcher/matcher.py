"""Core matching engine — country-scoped fuzzy matching between branch and HQ-level data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import duckdb
import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

from .preprocessing import normalize, parse_lei_set
from .countries import normalize_country

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MatchConfig:
    """Tunable parameters for the matching pipeline."""

    # Column names in the *left* (branch-level) file
    left_name_col: str = "institution_name"
    left_country_col: str = "country_name"
    left_lei_col: str = "lei"

    # Column names in the *right* (HQ / SNL) file
    right_name_col: str = "SP_ENTITY_NAME"
    right_country_col: str = "SP_COUNTRY_NAME"
    right_lei_col: str = "SP_LEI"

    # Matching thresholds
    score_threshold: float = 50.0   # minimum fuzzy score to keep a candidate (0-100)
    top_n: int = 5                  # max matches returned per left row

    # Scorer: "WRatio", "token_sort_ratio", "token_set_ratio", "partial_ratio", or "ensemble"
    # "ensemble" takes max(WRatio, partial_ratio) — best for asymmetric-length name pairs
    scorer: str = "ensemble"

    # When True, left rows that get zero matches in the strict country-scoped pass
    # are retried against the full right dataset (cross-country fallback).
    # Results from the relaxed pass are flagged with country_match=False.
    relax_country: bool = False

    # Extra columns to carry through from each side into the output
    left_extra_cols: list[str] = field(default_factory=list)
    right_extra_cols: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorer helpers
# ---------------------------------------------------------------------------

_SCORERS: dict[str, Any] = {
    "WRatio": fuzz.WRatio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "partial_ratio": fuzz.partial_ratio,
}

_VALID_SCORERS = list(_SCORERS) + ["ensemble"]


def _get_scorer(name: str):
    if name not in _VALID_SCORERS:
        raise ValueError(f"Unknown scorer '{name}'. Choose from: {_VALID_SCORERS}")
    return _SCORERS.get(name)  # None signals ensemble mode


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

class FuzzyMatcher:
    """Load two parquet files into DuckDB, then run country-scoped fuzzy matching."""

    def __init__(self, left_path: str, right_path: str, config: MatchConfig | None = None):
        self.left_path = left_path
        self.right_path = right_path
        self.cfg = config or MatchConfig()
        self._ensemble = (self.cfg.scorer == "ensemble")
        self._scorer = _get_scorer(self.cfg.scorer)  # None if ensemble
        self._conn = duckdb.connect()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both parquet files via DuckDB and return as DataFrames."""
        logger.info("Loading left file: %s", self.left_path)
        left = self._conn.execute(f"SELECT * FROM read_parquet('{self.left_path}')").df()

        logger.info("Loading right file: %s", self.right_path)
        right = self._conn.execute(f"SELECT * FROM read_parquet('{self.right_path}')").df()

        logger.info("Left rows: %d  |  Right rows: %d", len(left), len(right))
        return left, right

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_right_index(self, right: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Build a normalised-country → sub-DataFrame index for the right file."""
        cfg = self.cfg
        index: dict[str, pd.DataFrame] = {}

        norm_keys = right[cfg.right_country_col].fillna("").apply(normalize_country)
        for country_key, grp in right.groupby(norm_keys, sort=False):
            index[str(country_key)] = grp.reset_index(drop=True)

        logger.info("Right index: %d distinct countries (after normalisation)", len(index))
        return index

    # ------------------------------------------------------------------
    # Fuzzy scoring (shared logic)
    # ------------------------------------------------------------------

    def _score(self, left_name_norm: str, right_names_norm: list[str]) -> np.ndarray:
        """Return a score array for one left name against a list of right names."""
        if self._ensemble:
            s_w = rfprocess.cdist(
                [left_name_norm], right_names_norm,
                scorer=fuzz.WRatio, processor=None,
                score_cutoff=self.cfg.score_threshold,
            )[0]
            s_p = rfprocess.cdist(
                [left_name_norm], right_names_norm,
                scorer=fuzz.partial_ratio, processor=None,
                score_cutoff=self.cfg.score_threshold,
            )[0]
            return np.maximum(s_w, s_p)
        else:
            return rfprocess.cdist(
                [left_name_norm], right_names_norm,
                scorer=self._scorer, processor=None,
                score_cutoff=self.cfg.score_threshold,
            )[0]

    # ------------------------------------------------------------------
    # Per-row matching
    # ------------------------------------------------------------------

    def _match_row(
        self,
        left_row: pd.Series,
        candidates: pd.DataFrame,
        country_matched: bool,
    ) -> list[dict]:
        """Score one left row against a candidates DataFrame; return result records."""
        cfg = self.cfg

        left_name_raw = left_row.get(cfg.left_name_col, "")
        left_name_norm = normalize(str(left_name_raw) if pd.notna(left_name_raw) else "")
        if not left_name_norm:
            return []

        right_names_norm = candidates[cfg.right_name_col].fillna("").apply(normalize).tolist()
        return self._score_and_build(
            left_row, left_name_raw, left_name_norm,
            candidates, right_names_norm, country_matched,
        )

    def _match_row_cached(
        self,
        left_row: pd.Series,
        candidates: pd.DataFrame,   # must have a '_norm_name' column already populated
        country_matched: bool,
    ) -> list[dict]:
        """Like _match_row but reuses pre-normalised '_norm_name' column to avoid
        re-normalising 2.5M strings for every relaxed-pass row."""
        left_name_raw = left_row.get(self.cfg.left_name_col, "")
        left_name_norm = normalize(str(left_name_raw) if pd.notna(left_name_raw) else "")
        if not left_name_norm:
            return []

        right_names_norm = candidates["_norm_name"].tolist()
        return self._score_and_build(
            left_row, left_name_raw, left_name_norm,
            candidates, right_names_norm, country_matched,
        )

    def _score_and_build(
        self,
        left_row: pd.Series,
        left_name_raw,
        left_name_norm: str,
        candidates: pd.DataFrame,
        right_names_norm: list[str],
        country_matched: bool,
    ) -> list[dict]:
        """Core: score left name against right_names_norm and build result records."""
        cfg = self.cfg
        scores = self._score(left_name_norm, right_names_norm)
        left_leis = parse_lei_set(left_row.get(cfg.left_lei_col))

        above = [(float(scores[i]), i) for i in range(len(scores)) if scores[i] >= cfg.score_threshold]
        above.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in above[: cfg.top_n]:
            right_row = candidates.iloc[idx]
            right_lei_raw = right_row.get(cfg.right_lei_col, "")
            rec: dict = {
                "left_" + cfg.left_name_col:    left_name_raw,
                "left_" + cfg.left_country_col: left_row.get(cfg.left_country_col),
                "left_" + cfg.left_lei_col:     left_row.get(cfg.left_lei_col),
                "right_" + cfg.right_name_col:    right_row.get(cfg.right_name_col),
                "right_" + cfg.right_country_col: right_row.get(cfg.right_country_col),
                "right_" + cfg.right_lei_col:     right_row.get(cfg.right_lei_col),
                "fuzzy_score":   round(score, 2),
                "lei_match":     bool(
                    left_leis & parse_lei_set(
                        str(right_lei_raw) if pd.notna(right_lei_raw) else ""
                    )
                ),
                "country_match": country_matched,
            }
            for col in cfg.left_extra_cols:
                rec[f"left_{col}"] = left_row.get(col)
            for col in cfg.right_extra_cols:
                rec[f"right_{col}"] = right_row.get(col)
            results.append(rec)
        return results

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, progress: bool = True) -> pd.DataFrame:
        """Execute the full matching pipeline and return a results DataFrame."""
        left, right = self._load()

        logger.info("Building country index on right file…")
        right_index = self._build_right_index(right)

        all_results: list[dict] = []
        relaxed_rows: list[pd.Series] = []   # left rows with zero strict matches

        iterator = left.iterrows()
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(left.iterrows(), total=len(left), desc="Matching rows", unit="row")
            except ImportError:
                pass

        for _, row in iterator:
            left_country_raw = row.get(self.cfg.left_country_col, "")
            country_key = normalize_country(
                str(left_country_raw) if pd.notna(left_country_raw) else ""
            )
            candidates = right_index.get(country_key)

            if candidates is None or candidates.empty:
                logger.debug("No right candidates for country '%s' (key: %s)", left_country_raw, country_key)
                if self.cfg.relax_country:
                    relaxed_rows.append(row)
                continue

            row_results = self._match_row(row, candidates, country_matched=True)
            if row_results:
                all_results.extend(row_results)
            elif self.cfg.relax_country:
                # Candidates existed but nothing scored above threshold — still retry globally
                relaxed_rows.append(row)

        # ------------------------------------------------------------------
        # Relaxed-country pass: retry unmatched rows against all right rows
        # ------------------------------------------------------------------
        if self.cfg.relax_country and relaxed_rows:
            logger.info(
                "Relaxed-country pass: retrying %d unmatched left rows against full right dataset…",
                len(relaxed_rows),
            )
            # Pre-normalise all right names once and cache on the DataFrame to
            # avoid repeating the 2.5M-row normalisation for every relaxed row.
            logger.info("Pre-normalising all right-side names (one-time cost)…")
            right["_norm_name"] = right[self.cfg.right_name_col].fillna("").apply(normalize)
            right_cached = right  # single object, no copy

            relax_iter = relaxed_rows
            if progress:
                try:
                    from tqdm import tqdm
                    relax_iter = tqdm(relaxed_rows, desc="Relaxed pass", unit="row")
                except ImportError:
                    pass

            for row in relax_iter:
                row_results = self._match_row_cached(row, right_cached, country_matched=False)
                all_results.extend(row_results)

        if not all_results:
            logger.warning("No matches found above threshold.")
            return pd.DataFrame()

        result_df = pd.DataFrame(all_results)

        # Rank per left entity
        result_df["match_rank"] = (
            result_df.groupby(
                ["left_" + self.cfg.left_name_col, "left_" + self.cfg.left_country_col]
            )["fuzzy_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        # Column ordering
        priority_cols = [
            "left_" + self.cfg.left_name_col,
            "left_" + self.cfg.left_country_col,
            "left_" + self.cfg.left_lei_col,
            "right_" + self.cfg.right_name_col,
            "right_" + self.cfg.right_country_col,
            "right_" + self.cfg.right_lei_col,
            "fuzzy_score",
            "lei_match",
            "country_match",
            "match_rank",
        ]
        extra = [c for c in result_df.columns if c not in priority_cols]
        result_df = result_df[priority_cols + extra]

        result_df = result_df.sort_values(
            ["left_" + self.cfg.left_name_col, "left_" + self.cfg.left_country_col, "match_rank"]
        ).reset_index(drop=True)

        n_strict  = result_df["country_match"].sum()
        n_relaxed = (~result_df["country_match"]).sum()
        logger.info(
            "Done. %d match pairs (%d strict country, %d relaxed) across %d left rows.",
            len(result_df), n_strict, n_relaxed,
            result_df[["left_" + self.cfg.left_name_col, "left_" + self.cfg.left_country_col]]
            .drop_duplicates().shape[0],
        )
        return result_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def validate_lei_scores(self, results: pd.DataFrame) -> pd.DataFrame:
        """Summary statistics grouped by lei_match — use to verify scoring quality."""
        if results.empty:
            return pd.DataFrame()
        if "lei_match" not in results.columns or "fuzzy_score" not in results.columns:
            return results
        return (
            results.groupby("lei_match")["fuzzy_score"]
            .agg(["count", "mean", "min", "max", "median"])
            .rename(columns={"count": "n", "mean": "avg_score", "min": "min_score",
                              "max": "max_score", "median": "median_score"})
            .round(2)
        )


