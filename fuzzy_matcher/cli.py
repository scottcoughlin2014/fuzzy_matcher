"""CLI entry point for fuzzy_matcher."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .matcher import FuzzyMatcher, MatchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _save(df, output: str, fmt: str) -> None:
    """Save a DataFrame to the requested format."""
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(p, index=False)
    elif fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "excel":
        df.to_excel(p, index=False)
    else:
        raise click.BadParameter(f"Unknown format '{fmt}'. Choose: parquet, csv, excel")
    click.echo(f"Results written to: {p}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """fuzzy-match — fuzzy matching between branch-level and HQ-level bank datasets."""


@cli.command("match")
@click.argument("left_file", type=click.Path(exists=True))
@click.argument("right_file", type=click.Path(exists=True))
# Output options
@click.option("-o", "--output", default="matches.parquet",
              show_default=True, help="Output file path.")
@click.option("--format", "fmt", default="parquet",
              type=click.Choice(["parquet", "csv", "excel"], case_sensitive=False),
              show_default=True, help="Output file format.")
# Column mapping — left (branch) side
@click.option("--left-name", default="institution_name", show_default=True,
              help="Name column in the left (branch-level) file.")
@click.option("--left-country", default="country_name", show_default=True,
              help="Country column in the left file.")
@click.option("--left-lei", default="lei", show_default=True,
              help="LEI column in the left file.")
@click.option("--left-extra", multiple=True, metavar="COL",
              help="Additional left-file columns to include in output. Repeatable.")
# Column mapping — right (HQ) side
@click.option("--right-name", default="SP_ENTITY_NAME", show_default=True,
              help="Name column in the right (HQ-level) file.")
@click.option("--right-country", default="SP_COUNTRY_NAME", show_default=True,
              help="Country column in the right file.")
@click.option("--right-lei", default="SP_LEI", show_default=True,
              help="LEI column in the right file.")
@click.option("--right-extra", multiple=True, metavar="COL",
              help="Additional right-file columns to include in output. Repeatable.")
# Scoring
@click.option("--scorer", default="WRatio",
              type=click.Choice(["WRatio", "token_sort_ratio", "token_set_ratio", "partial_ratio"],
                                case_sensitive=False),
              show_default=True, help="RapidFuzz scorer to use.")
@click.option("--threshold", default=50.0, show_default=True,
              help="Minimum fuzzy score (0-100) for a pair to be kept.")
@click.option("--top-n", default=5, show_default=True,
              help="Maximum number of right-side matches to return per left row.")
# Misc
@click.option("--no-progress", is_flag=True, default=False,
              help="Disable the progress bar.")
@click.option("--validate-lei", is_flag=True, default=False,
              help="Print LEI-match vs non-LEI-match score summary after matching.")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Enable debug logging.")
def match_cmd(
    left_file, right_file,
    output, fmt,
    left_name, left_country, left_lei, left_extra,
    right_name, right_country, right_lei, right_extra,
    scorer, threshold, top_n,
    no_progress, validate_lei, verbose,
):
    """Fuzzy-match LEFT_FILE (branch-level) against RIGHT_FILE (HQ-level).

    Both files may be .parquet, .csv, or any format readable by DuckDB.
    Matching is scoped by country first, then name similarity is scored
    with RapidFuzz. Pairs where LEIs agree receive a score bonus.

    \b
    Example:
        fuzzy-match match branch.parquet snl.parquet \\
            --threshold 60 --top-n 3 --output results.parquet
    """
    _setup_logging(verbose)

    cfg = MatchConfig(
        left_name_col=left_name,
        left_country_col=left_country,
        left_lei_col=left_lei,
        right_name_col=right_name,
        right_country_col=right_country,
        right_lei_col=right_lei,
        scorer=scorer,
        score_threshold=float(threshold),
        top_n=int(top_n),
        left_extra_cols=list(left_extra),
        right_extra_cols=list(right_extra),
    )

    matcher = FuzzyMatcher(left_file, right_file, cfg)

    try:
        results = matcher.run(progress=not no_progress)
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)

    if results.empty:
        click.echo("No matches found. Try lowering --threshold.", err=True)
        raise SystemExit(1)

    if validate_lei:
        click.echo("\n=== LEI match score validation ===")
        click.echo(matcher.validate_lei_scores(results).to_string())
        click.echo()

    # Print a short preview
    preview_cols = [c for c in [
        f"left_{left_name}", f"left_{left_country}", f"left_{left_lei}",
        f"right_{right_name}", f"right_{right_lei}",
        "fuzzy_score", "lei_match", "match_rank",
    ] if c in results.columns]
    click.echo(results[preview_cols].head(20).to_string(index=False))
    click.echo(f"\n… {len(results):,} total match pairs\n")

    _save(results, output, fmt)


@cli.command("schema")
@click.argument("file", type=click.Path(exists=True))
def schema_cmd(file):
    """Print column names and dtypes for FILE (parquet / csv)."""
    import duckdb
    conn = duckdb.connect()
    df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{file}')").df()
    click.echo(df.to_string(index=False))


def main():
    cli()


if __name__ == "__main__":
    main()
