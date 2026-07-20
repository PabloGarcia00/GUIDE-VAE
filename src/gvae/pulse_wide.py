"""
Shared loader for processed_data/<tier>/<variant>.parquet ("PULSE_WIDE"
datasets), used by both preprocess_lib.get_full_data() and
conditioning_lib.add_pulse_cluster() so both read/filter/sort the data
identically and stay row-aligned -- a mismatch between the two would
silently pair the wrong cluster label with the wrong daily profile.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

# Grants access to prepare_pulse_data.value_columns() so "what counts as a
# data column" for processed_data/<tier>/<variant>.parquet stays defined in
# exactly one place, shared with scripts/prepare_pulse_data.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from prepare_pulse_data import value_columns as pulse_value_columns  # noqa: E402


def parse_pulse_wide_path(dataset_path: str) -> tuple[str, str, str]:
    """dataset_dir + '/PULSE_WIDE/<tier>/<variant>' -> (dataset_dir, tier, variant)."""
    dataset_dir, _, rest = str(dataset_path).replace("\\", "/").rpartition("/PULSE_WIDE/")
    tier, variant = rest.split("/")
    return dataset_dir, tier, variant


def load_pulse_wide_frame(dataset_dir: str, tier: str, variant: str) -> tuple[pd.DataFrame, list[str]]:
    """Returns the cleaned, entity-complete, (entity_id, date)-sorted dataframe
    plus the list of value columns, ready to reshape into (entities, days, features)."""
    parquet_path = Path(dataset_dir) / tier / f"{variant}.parquet"
    con = duckdb.connect()
    df = con.sql(f"SELECT * FROM '{parquet_path}'").df()

    value_cols = pulse_value_columns(list(df.columns), variant)
    df = df.dropna(subset=value_cols).reset_index(drop=True)

    # `user_id` alone is not a unique household key -- the same numeric id
    # is reused across up to 5 of the 6 clusters (confirmed: 865 distinct
    # user_id vs. 1792 distinct (cluster,user_id) pairs in the `mixed`
    # tier). Build a synthetic globally-unique id, reusing the exact
    # convention already established in
    # SSMD-Internship/src/ssmd/utils.py::PulseLoader.load_cluster.
    df["entity_id"] = df["cluster"].str.lstrip("c").astype(int) * 1000 + df["user_id"]
    df = df.sort_values(["entity_id", "date"]).reset_index(drop=True)

    # Downstream reshape() needs a strictly rectangular (entities, days,
    # features) array. A handful of entities are missing a day or two (the
    # ~0.02% of rows with a legitimately missing reading, dropped above) --
    # drop those *entire* entities rather than padding/filling, since
    # fabricating a reading would be indistinguishable from a genuine
    # zero-watt reading under the zero-preserved-log scheme.
    counts = df.groupby("entity_id").size()
    num_days = int(counts.mode().iloc[0])
    complete_entities = counts.index[counts == num_days]
    if len(complete_entities) < len(counts):
        print(
            f"dropping {len(counts) - len(complete_entities)} entities "
            f"with != {num_days} days (of {len(counts)} total)"
        )
    df = df[df["entity_id"].isin(complete_entities)].reset_index(drop=True)

    return df, value_cols
