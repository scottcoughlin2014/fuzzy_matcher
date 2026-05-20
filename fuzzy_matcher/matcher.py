"""Core matching engine — country-scoped fuzzy matching with LEI-based scoring bonus."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process as rfprocess
from rapidfuzz.utils import default_process

from .preprocessing import normalize, parse_lei_set

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

    # Scorer: one of "WRatio", "token_sort_ratio", "token_set_ratio", "partial_ratio"
    scorer: str = "WRatio"

    # Extra columns to carry through from each side into the output
    left_extra_cols: list[str] = field(default_factory=list)
    right_extra_cols: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorer lookup
# ---------------------------------------------------------------------------

_SCORERS: dict[str, Any] = {
    "WRatio": fuzz.WRatio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "partial_ratio": fuzz.partial_ratio,
}


def _get_scorer(name: str):
    if name not in _SCORERS:
        raise ValueError(f"Unknown scorer '{name}'. Choose from: {list(_SCORERS)}")
    return _SCORERS[name]


# ---------------------------------------------------------------------------
# Country normalisation (handle minor spelling differences)
# ---------------------------------------------------------------------------

def _norm_country(name: str) -> str:
    """Light normalisation for country name matching."""
    if not name or not isinstance(name, str):
        return ""
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

class FuzzyMatcher:
    """Load two parquet files into DuckDB, then run country-scoped fuzzy matching."""

    def __init__(self, left_path: str, right_path: str, config: MatchConfig | None = None):
        self.left_path = left_path
        self.right_path = right_path
        self.cfg = config or MatchConfig()
        self._scorer = _get_scorer(self.cfg.scorer)
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
        """
        Build a country → sub-DataFrame index for the right file.
        Keys are lowercased country names.
        """
        cfg = self.cfg
        index: dict[str, pd.DataFrame] = {}

        for country_key, grp in right.groupby(
            right[cfg.right_country_col].str.strip().str.lower(), sort=False
        ):
            index[str(country_key)] = grp.reset_index(drop=True)

        logger.info("Right index: %d distinct countries", len(index))
        return index

    # ------------------------------------------------------------------
    # Per-row matching
    # ------------------------------------------------------------------

    def _match_row(
        self,
        left_row: pd.Series,
        right_index: dict[str, pd.DataFrame],
    ) -> list[dict]:
        """Return up to cfg.top_n match records for a single left row."""
        cfg = self.cfg

        # --- country lookup ---
        left_country_raw = left_row.get(cfg.left_country_col, "")
        country_key = _norm_country(str(left_country_raw) if pd.notna(left_country_raw) else "")
        candidates = right_index.get(country_key)

        if candidates is None or candidates.empty:
            logger.debug("No right candidates for country '%s'", country_key)
            return []

        # --- name normalisation ---
        left_name_raw = left_row.get(cfg.left_name_col, "")
        left_name_norm = normalize(str(left_name_raw) if pd.notna(left_name_raw) else "")
        if not left_name_norm:
            return []

        right_names_norm = candidates[cfg.right_name_col].fillna("").apply(normalize).tolist()

        # --- fuzzy scoring (name only) ---
        scores = rfprocess.cdist(
            [left_name_norm],
            right_names_norm,
            scorer=self._scorer,
            processor=None,   # already normalised
            score_cutoff=self.cfg.score_threshold,
        )[0]  # shape: (len(right_names_norm),)

        # --- pre-compute LEI sets for lei_match flag in output ---
        left_leis = parse_lei_set(left_row.get(cfg.left_lei_col))

        # --- filter and sort ---
        above_threshold = [
            (float(scores[i]), i)
            for i in range(len(scores))
            if scores[i] >= cfg.score_threshold
        ]
        above_threshold.sort(key=lambda x: x[0], reverse=True)
        top = above_threshold[: cfg.top_n]

        # --- build result records ---
        results = []
        for score, idx in top:
            right_row = candidates.iloc[idx]
            rec: dict = {
                # Left-side key fields
                "left_" + cfg.left_name_col: left_name_raw,
                "left_" + cfg.left_country_col: left_country_raw,
                "left_" + cfg.left_lei_col: left_row.get(cfg.left_lei_col),
                # Right-side key fields
                "right_" + cfg.right_name_col: right_row.get(cfg.right_name_col),
                "right_" + cfg.right_country_col: right_row.get(cfg.right_country_col),
                "right_" + cfg.right_lei_col: right_row.get(cfg.right_lei_col),
                # Scores
                "fuzzy_score": round(score, 2),
                "lei_match": bool(
                    parse_lei_set(left_row.get(cfg.left_lei_col))
                    & parse_lei_set(str(right_row.get(cfg.right_lei_col, "")) if pd.notna(right_row.get(cfg.right_lei_col)) else "")
                ),
            }
            # Extra pass-through columns
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

        iterator = left.iterrows()
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(left.iterrows(), total=len(left), desc="Matching rows", unit="row")
            except ImportError:
                pass

        for _, row in iterator:
            all_results.extend(self._match_row(row, right_index))

        if not all_results:
            logger.warning("No matches found above threshold.")
            return pd.DataFrame()

        result_df = pd.DataFrame(all_results)

        # Add a rank column per left entity
        result_df["match_rank"] = (
            result_df.groupby(["left_" + self.cfg.left_name_col, "left_" + self.cfg.left_country_col])
            ["fuzzy_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        # Re-order columns for readability
        priority_cols = [
            "left_" + self.cfg.left_name_col,
            "left_" + self.cfg.left_country_col,
            "left_" + self.cfg.left_lei_col,
            "right_" + self.cfg.right_name_col,
            "right_" + self.cfg.right_country_col,
            "right_" + self.cfg.right_lei_col,
            "fuzzy_score",
            "lei_match",
            "match_rank",
        ]
        extra = [c for c in result_df.columns if c not in priority_cols]
        result_df = result_df[priority_cols + extra]

        result_df = result_df.sort_values(
            ["left_" + self.cfg.left_name_col, "left_" + self.cfg.left_country_col, "match_rank"]
        ).reset_index(drop=True)

        logger.info(
            "Done. %d match pairs across %d left rows.",
            len(result_df),
            result_df[["left_" + self.cfg.left_name_col, "left_" + self.cfg.left_country_col]]
            .drop_duplicates()
            .shape[0],
        )
        return result_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def validate_lei_scores(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Return summary statistics for rows where LEI matches exist,
        so you can verify that LEI-matched pairs score highly.
        """
        if results.empty:
            return pd.DataFrame()
        lei_col = "lei_match"
        score_col = "fuzzy_score"
        if lei_col not in results.columns or score_col not in results.columns:
            return results
        summary = (
            results.groupby(lei_col)[score_col]
            .agg(["count", "mean", "min", "max", "median"])
            .rename(columns={"count": "n", "mean": "avg_score", "min": "min_score",
                              "max": "max_score", "median": "median_score"})
            .round(2)
        )
        return summary
