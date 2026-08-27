"""Schéma commun des profils (synthétiques et enquête)."""

from __future__ import annotations

from orientia.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    META_COLS,
    NUMERIC_FEATURES,
    TARGET_COL,
)

PROFILE_COLUMNS = FEATURE_COLS + [TARGET_COL] + META_COLS

COLUMN_DTYPES = {
    **{c: "float64" for c in NUMERIC_FEATURES},
    **{c: "object" for c in CATEGORICAL_FEATURES},
    **{c: "int64" for c in BINARY_FEATURES},
    TARGET_COL: "object",
    "population": "object",
    "source_id": "object",
    "anonymized_id": "object",
}

SCORE_0_10 = [c for c in NUMERIC_FEATURES if c != "note_moyenne"]


def empty_profile_dict() -> dict:
    row: dict = {c: 0.0 for c in NUMERIC_FEATURES}
    row.update({c: 0 for c in BINARY_FEATURES})
    row["pref_environnement"] = "bureau"
    row["pref_professionnelle"] = "developpement_logiciel"
    row[TARGET_COL] = None
    row["population"] = "unknown"
    row["source_id"] = ""
    row["anonymized_id"] = ""
    return row
