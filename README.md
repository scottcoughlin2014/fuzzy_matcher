# fuzzy-matcher

A Python CLI package for fuzzy matching branch-level bank records against headquarters-level bank records. Designed specifically for the case where the left dataset contains branch observations (with country = branch country) and the right dataset contains HQ-level entities (with country = HQ country).

---

## Table of Contents

1. [Overview](#overview)
2. [Data Inputs](#data-inputs)
3. [Matching Pipeline](#matching-pipeline)
   - [Step 1 — Data Loading](#step-1--data-loading)
   - [Step 2 — Country Index](#step-2--country-index)
   - [Step 3 — Name Normalisation](#step-3--name-normalisation)
   - [Step 4 — Fuzzy Scoring](#step-4--fuzzy-scoring)
   - [Step 5 — Filtering and Ranking](#step-5--filtering-and-ranking)
   - [Step 6 — Relaxed-Country Pass (optional)](#step-6--relaxed-country-pass-optional)
4. [Country Normalisation](#country-normalisation)
5. [LEI Handling](#lei-handling)
6. [Output Schema](#output-schema)
7. [CLI Reference](#cli-reference)
8. [Configuration Reference](#configuration-reference)
9. [Known Limitations](#known-limitations)
10. [Validation Against Ground Truth](#validation-against-ground-truth)

---

## Overview

The core challenge is that both datasets contain banks at different organisational levels:

- **Left file** (`oct2025_training_sample.parquet`): branch-level records. The `country_name` column reflects the country where the *branch* is located, which may differ from the HQ country. The `institution_name` column is the name of the institution the branch belongs to (i.e. the HQ entity name, or a variant of it). One institution may have multiple branches, producing multiple left rows with the same `institution_name`.
- **Right file** (`SNL.parquet`): HQ-level entity records. The `SP_COUNTRY_NAME` column reflects the country where the *HQ* is registered. The `SP_ENTITY_NAME` is the official entity name.

Because both datasets also contain LEI codes (`lei` and `SP_LEI`), these are used exclusively as a **diagnostic signal** to evaluate matching quality — they are deliberately excluded from the scoring algorithm so that the fuzzy match quality can be assessed and tuned independently.

---

## Data Inputs

### Left file — branch-level

| Column | Description |
|---|---|
| `institution_name` | Name of the parent institution (used for matching) |
| `country_name` | Country of the branch (used for country scoping) |
| `lei` | Legal Entity Identifier; may be pipe-separated (`A \| B`) for institutions with multiple LEIs |
| `branch_name` | Name of the specific branch (not used for matching) |
| `bic` | SWIFT BIC code (not used for matching) |
| `address1`, `address2`, `city` | Branch address (not used for matching) |

### Right file — HQ-level (SNL)

| Column | Description |
|---|---|
| `SP_ENTITY_NAME` | Official HQ entity name (used for matching) |
| `SP_COUNTRY_NAME` | Country of HQ registration (used for country scoping) |
| `SP_LEI` | LEI of the HQ entity (used as diagnostic only) |
| `SP_ENTITY_ID` | SNL internal entity ID |
| `SP_SWIFT_BIC` | BIC code |
| Financial columns | `SNL_TOTAL_ASSETS_*`, `SNL_TOTAL_DEPOSITS_*`, etc. (passed through if requested) |

---

## Matching Pipeline

The pipeline runs in the following sequence:

```
Load files (DuckDB)
       │
       ▼
Build country index on right file
       │
       ▼
For each left row:
  ├─ Normalise left country name
  ├─ Look up country bucket in right index
  ├─ Normalise left institution name
  ├─ Normalise right candidate names
  ├─ Compute fuzzy scores (ensemble)
  ├─ Apply score threshold
  └─ Keep top-N matches
       │
       ▼ (if --relax-country)
For unmatched left rows:
  ├─ Score against full right dataset (pre-normalised)
  └─ Flag results as country_match=False
       │
       ▼
Assemble output DataFrame
Assign match_rank per left row
Sort and return
```

### Step 1 — Data Loading

Both files are loaded via **DuckDB** using `read_parquet()`. DuckDB handles efficient columnar reads and supports any format it can natively read (parquet, CSV, etc.). The results are materialised as pandas DataFrames for the matching loop.

### Step 2 — Country Index

The right (SNL) DataFrame is split into a dictionary keyed by **normalised country name**:

```python
right_index: dict[str, pd.DataFrame]
# e.g. right_index["france"] → all SNL rows where SP_COUNTRY_NAME normalises to "france"
```

Both left and right country names are passed through `normalize_country()` before lookup (see [Country Normalisation](#country-normalisation)). This ensures that spelling variants such as `"Bosnia and Herzegovina"` (left) and `"Bosnia & Herzegovina"` (right) resolve to the same bucket.

For each left row, only the matching country bucket is searched. This reduces the candidate pool from ~2.5 million rows to typically dozens or hundreds, making the fuzzy comparison tractable.

### Step 3 — Name Normalisation

Before any string comparison, both the left institution name and all right candidate names are passed through `normalize()` in `preprocessing.py`. The normalisation pipeline is applied in this exact order:

| Step | What it does | Example |
|---|---|---|
| **1. Strip exchange tickers** | Removes `(EXCHANGE:TICKER)` appended by SNL | `SpareBank 1 Nord-Norge (OB:NONG)` → `SpareBank 1 Nord-Norge` |
| **2. Strip fancy quotes** | Removes `«»„""''` and similar Unicode quote characters | `«International Bank»` → `International Bank` |
| **3. Word-internal hyphens → spaces** | Only hyphens between word characters are replaced | `ECOBANK-Mali` → `ECOBANK Mali` |
| **4. Unicode NFKD → ASCII** | Decomposes accented characters, drops combining marks | `Société Générale` → `Societe Generale` |
| **5. Lowercase** | | `NORDEA BANK` → `nordea bank` |
| **6. Legal suffix normalisation** | Maps long-form entity suffixes to canonical abbreviations | `Société Anonyme` → `sa`, `Aktiengesellschaft` → `ag`, `Open Joint Stock` → `ojsc` |
| **7. Strip remaining punctuation** | Keeps only alphanumerics and spaces | `B.C. MAIB` → `bc maib` |
| **8. Collapse whitespace** | | `nordea  bank` → `nordea bank` |

**Legal suffixes normalised** (exhaustive list):

| Long form | Canonical |
|---|---|
| limited liability company | llc |
| limited liability partnership | llp |
| public limited company | plc |
| incorporated | inc |
| corporation | corp |
| cooperative | coop |
| national association | na |
| federal savings bank | fsb |
| societe anonyme / sociedad anonima | sa |
| societe en commandite | sca |
| societe a responsabilite limitee | sarl |
| sociedad de responsabilidad limitada | srl |
| aktiengesellschaft | ag |
| gesellschaft mit beschraenkter haftung | gmbh |
| naamloze vennootschap | nv |
| besloten vennootschap | bv |
| open joint stock | ojsc |
| closed joint stock | cjsc |
| private joint stock / public joint stock | pjsc |
| joint stock company | jsc |
| joint stock bank | jsb |
| commercial bank | cb |

### Step 4 — Fuzzy Scoring

Fuzzy scoring uses **RapidFuzz** (`rapidfuzz.process.cdist`) operating on pre-normalised strings. The default scorer is `ensemble`.

#### Scorers available

| Scorer | Description | Best for |
|---|---|---|
| `ensemble` **(default)** | `max(WRatio, partial_ratio)` | Mixed cases — handles both equal-length and asymmetric-length names |
| `WRatio` | Weighted combination of multiple Levenshtein-based ratios, handles token reordering | Symmetric pairs where both sides have similar length |
| `partial_ratio` | Best alignment of the shorter string within the longer | Short abbreviation/branch name vs long HQ name |
| `token_sort_ratio` | Sort tokens before comparing | Word-order differences |
| `token_set_ratio` | Set intersection of tokens | Subset/superset names |

All scores are on a **0–100 scale**.

#### Why ensemble?

In this dataset, the left side (branch records) sometimes uses abbreviated or shortened versions of the HQ name, while SNL sometimes appends branch descriptors or geographic qualifiers. `WRatio` handles same-length pairs well but can under-score when one name is a strict subset of the other. `partial_ratio` handles asymmetric length well but can over-score unrelated short strings. Taking the maximum of both captures the best of each approach.

**Empirical validation** on LEI ground-truth pairs (193 pairs with known correct matches):

| Scorer | Pairs scoring ≥ 80 | Mean score |
|---|---|---|
| WRatio alone | 167 / 193 | 88.6 |
| partial_ratio alone | 180 / 193 | 92.2 |
| **ensemble (max)** | **180+ / 193** | **~95** |

#### `cdist` with `score_cutoff`

RapidFuzz's `cdist` computes a full score matrix but returns 0 for any pair below `score_cutoff`. This is significantly faster than computing exact scores for all pairs because RapidFuzz can short-circuit low-scoring comparisons internally. The `score_cutoff` passed to `cdist` is set to `score_threshold` so filtering happens inside the C extension rather than in Python.

### Step 5 — Filtering and Ranking

After scoring:
1. All candidates with `score < score_threshold` are discarded.
2. Remaining candidates are sorted descending by score.
3. The top `top_n` candidates are retained per left row.
4. A `match_rank` column is assigned (1 = best match, 2 = second-best, etc.) within each `(institution_name, country_name)` group.

### Step 6 — Relaxed-Country Pass (optional)

When `--relax-country` is passed, left rows that received **zero matches** in the strict country-scoped pass (either because no country bucket existed, or all candidates scored below threshold) are retried against the **full right dataset**.

To keep this tractable given the 2.5M-row right file:
- All right-side names are **pre-normalised once** and cached in a `_norm_name` column on the DataFrame before the relaxed pass begins (~47 seconds one-time cost).
- Each relaxed-pass left row then scores against the full 2.5M pre-normalised strings (~7 seconds per row).
- Results from the relaxed pass are flagged `country_match=False` in the output.

**Performance estimates** (benchmarked on this dataset):
- Strict country-scoped pass (174 left rows, ~2.5M right): ~45 seconds
- Relaxed pass normalisation (one-time): ~47 seconds
- Relaxed pass per unmatched row: ~7 seconds
- Relaxed pass for ~28 unmatched rows: ~3–4 minutes

---

## Country Normalisation

Country names are normalised by `normalize_country()` in `countries.py` before use as index keys. This is separate from entity name normalisation. The steps are:

1. Unicode NFKD → ASCII
2. Lowercase
3. `&` → `and`
4. Strip punctuation, collapse whitespace
5. Apply alias substitutions (see table below)

Alias substitutions handle the most common mismatches observed between the two datasets:

| Left dataset spelling | Right dataset spelling | Resolved to |
|---|---|---|
| `Bosnia and Herzegovina` | `Bosnia & Herzegovina` | `bosnia and herzegovina` |
| `Sao Tome and Principe` | `Sao Tome & Principe` | `sao tome and principe` |
| `Cote d'Ivoire` | `Côte d'Ivoire` | `cote d ivoire` |
| `Viet Nam` | `Vietnam` | `vietnam` |
| `Turkiye` | `Türkiye` | `turkey` |
| `Democratic Republic of the Congo` | `Dem. Rep. Congo` | `democratic republic of the congo` |
| `Republic of the Congo` | `Rep. Of the Congo` | `republic of the congo` |
| `Central African Republic` | `Central African Rep.` | `central african republic` |
| `The Gambia` | `Gambia` | `the gambia` |
| `Taiwan (Province of China)` | `Taiwan` | `taiwan` |
| `Palestine, State of` | `Palestine` | `palestine` |
| `United States` | `USA` | `united states` |

**Cases intentionally not aliased** (structurally different, handled by `--relax-country`):
- French overseas territories (`Martinique`, `Guadeloupe`, `Reunion`, `French Guiana`, `Mayotte`) vs `France` — the branch is in the territory but the HQ is in mainland France
- Channel Islands (`Guernsey, Channel Islands`, `Jersey, Channel Islands`) — SNL does not have these as separate country entries
- Branch countries where the parent HQ is registered elsewhere (e.g. Aland Islands branch of Nordea, whose HQ is in Finland)

---

## LEI Handling

The Legal Entity Identifier (LEI) is present in both datasets but is **not used in scoring**. It serves two roles:

### 1. Output flag (`lei_match`)

Every output row includes a boolean `lei_match` column, which is `True` when at least one LEI from the left row's (potentially pipe-separated) LEI set overlaps with the right row's LEI. This allows downstream filtering or inspection of matches where the LEI provides independent confirmation.

### 2. Validation diagnostic (`--validate-lei`)

The `--validate-lei` CLI flag prints a score summary table grouped by `lei_match` after the run:

```
             n  avg_score  min_score  max_score  median_score
lei_match
False      308      89.04       60.0      100.0          87.9
True       127      98.06       85.5      100.0         100.0
```

Since LEI-matched pairs are ground-truth correct matches, this table is the primary tool for **tuning the algorithm**: if `lei_match=True` rows are scoring low, the normalisation or scorer needs adjustment. The observed gap (avg 98.1 for LEI-confirmed vs 89.0 for others) confirms the algorithm is behaving correctly.

### Pipe-separated LEIs

Some left rows contain multiple LEIs separated by ` | ` (e.g. institutions with both a global LEI and a local subsidiary LEI). `parse_lei_set()` splits these into a Python set before comparison, so a match on any LEI in the set triggers `lei_match=True`.

### The `lei_non_matched.csv` diagnostic file

Running the `lei_non_matched.csv` generation script (see `data/lei_non_matched.csv`) produces a file of all ground-truth LEI pairs that are **not** returned by the current run. These fall into three categories:

| `miss_reason` | Count | Description |
|---|---|---|
| `country_mismatch` | 56 | LEIs match and fuzzy score is good, but branch country ≠ HQ country. Use `--relax-country` to recover these. |
| `low_fuzzy_score` | 3 | Same country, but names are too dissimilar (rebrands or pure acronyms). Cannot be recovered by tuning alone. |
| `country_mismatch+low_score` | 2 | Both issues present. |

---

## Output Schema

Each row in the output represents one (left row, right row) candidate pair. The key columns are:

| Column | Type | Description |
|---|---|---|
| `left_institution_name` | str | Institution name from the left (branch) file |
| `left_country_name` | str | Branch country from the left file |
| `left_lei` | str | LEI(s) from the left file (may be pipe-separated) |
| `right_SP_ENTITY_NAME` | str | Entity name from the right (SNL) file |
| `right_SP_COUNTRY_NAME` | str | HQ country from the right file |
| `right_SP_LEI` | str | LEI from the right file |
| `fuzzy_score` | float | Similarity score 0–100 (ensemble: max of WRatio and partial_ratio, on normalised strings) |
| `lei_match` | bool | True if any LEI in the left set matches the right LEI |
| `country_match` | bool | True if the left country and right country resolved to the same normalised key |
| `match_rank` | int | Rank of this right candidate for the given left row (1 = best) |

Additional columns from either file can be included via `--left-extra` and `--right-extra`.

---

## CLI Reference

### `fuzzy-match match`

```
fuzzy-match match LEFT_FILE RIGHT_FILE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|---|---|
| `LEFT_FILE` | Path to the branch-level parquet (or CSV) file |
| `RIGHT_FILE` | Path to the HQ-level parquet (or CSV) file |

**Output options:**

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | `matches.parquet` | Output file path |
| `--format` | `parquet` | Output format: `parquet`, `csv`, or `excel` |

**Column mapping — left file:**

| Option | Default | Description |
|---|---|---|
| `--left-name` | `institution_name` | Entity name column |
| `--left-country` | `country_name` | Country column |
| `--left-lei` | `lei` | LEI column |
| `--left-extra COL` | _(none)_ | Extra column to pass through (repeatable) |

**Column mapping — right file:**

| Option | Default | Description |
|---|---|---|
| `--right-name` | `SP_ENTITY_NAME` | Entity name column |
| `--right-country` | `SP_COUNTRY_NAME` | Country column |
| `--right-lei` | `SP_LEI` | LEI column |
| `--right-extra COL` | _(none)_ | Extra column to pass through (repeatable) |

**Scoring options:**

| Option | Default | Description |
|---|---|---|
| `--scorer` | `ensemble` | Scoring function: `ensemble`, `WRatio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio` |
| `--threshold` | `50.0` | Minimum score (0–100) to retain a candidate pair |
| `--top-n` | `5` | Maximum matches to return per left row |

**Other options:**

| Option | Default | Description |
|---|---|---|
| `--relax-country` | off | Retry unmatched left rows against the full right dataset; results flagged `country_match=False` |
| `--validate-lei` | off | Print score summary grouped by `lei_match` after the run |
| `--no-progress` | off | Suppress the tqdm progress bar |
| `-v`, `--verbose` | off | Enable DEBUG-level logging |

**Example:**

```bash
fuzzy-match match \
  data/oct2025_training_sample.parquet \
  data/SNL.parquet \
  --threshold 60 \
  --top-n 3 \
  --format csv \
  -o results/matches.csv \
  --validate-lei \
```

### `fuzzy-match schema`

```
fuzzy-match schema FILE
```

Prints all column names and data types for a parquet or CSV file. Useful for inspecting column names before setting `--left-name`, `--right-name`, etc.

---

## Configuration Reference

`MatchConfig` (in `matcher.py`) is the dataclass used to configure a `FuzzyMatcher` instance programmatically:

```python
from fuzzy_matcher.matcher import FuzzyMatcher, MatchConfig

cfg = MatchConfig(
    left_name_col="institution_name",
    left_country_col="country_name",
    left_lei_col="lei",
    right_name_col="SP_ENTITY_NAME",
    right_country_col="SP_COUNTRY_NAME",
    right_lei_col="SP_LEI",
    scorer="ensemble",          # "ensemble" | "WRatio" | "partial_ratio" | ...
    score_threshold=60.0,       # 0–100
    top_n=3,                    # max candidates per left row
    relax_country=False,        # enable cross-country fallback
    left_extra_cols=[],         # e.g. ["bic", "branch_name"]
    right_extra_cols=[],        # e.g. ["SP_ENTITY_ID", "SNL_TOTAL_ASSETS_FY2024"]
)

matcher = FuzzyMatcher("branch.parquet", "snl.parquet", cfg)
results = matcher.run(progress=True)

# LEI validation summary
print(matcher.validate_lei_scores(results))
```

---

## Known Limitations

### 1. Rebrands and acronyms
Pairs where the institution has genuinely changed its name (e.g. `LCL SA` ↔ `Crédit Lyonnais`, `BC MOLDOVA-AGROINDBANK` ↔ `B.C. MAIB`) will score poorly regardless of threshold or scorer. These require a hand-curated alias table to resolve.

### 2. Branch-country vs HQ-country structural mismatch
56 of the 193 LEI-verified ground-truth pairs have the branch in a different country from the HQ. The strict country-scoped pass will not find these. `--relax-country` recovers them at the cost of a global scan (~3–4 minutes for ~28 rows).

### 3. French overseas territories
Branches in `Martinique`, `Guadeloupe`, `Reunion`, `French Guiana`, `Saint Pierre and Miquelon`, `Mayotte`, `St Barthelemy`, and `St. Martin` will not match against French HQs in the strict pass because SNL records their HQs under `France`, not the territory. These are systematically recovered by `--relax-country`.

### 4. Duplicate left rows
The left file may contain multiple branches of the same institution (same `institution_name`, same `country_name`). Each branch row is matched independently, producing duplicate `(institution_name, country_name)` → `(SP_ENTITY_NAME)` pairs in the output. Deduplicate on `(left_institution_name, right_SP_ENTITY_NAME)` if only unique institution pairs are needed.

### 5. SNL entities are not exclusively banks
The right file contains ~2.5M entities across all industries (insurance companies, holding companies, etc.). Country scoping significantly reduces false positive volume, but some non-bank entities in the same country may score above threshold if their names happen to resemble bank names (e.g. `"ABC Asset Management"` scoring against `"AFC Commercial Bank"`). Filtering on `SP_COMPANY_TYPE` or `MI_PRIMARY_INDUSTRY` before running can help if this is a concern.
