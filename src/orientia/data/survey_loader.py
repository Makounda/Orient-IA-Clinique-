"""Charge et normalise les réponses d'enquête vers le schéma ML commun."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orientia.config import (
    BINARY_FEATURES,
    ENVIRONNEMENTS,
    PREFS_PRO,
    SURVEY_PROCESSED_CSV,
    SURVEY_RAW_CSV,
    TARGET_COL,
    ensure_dirs,
    load_parcours_codes,
)
from orientia.data.schema import PROFILE_COLUMNS, SCORE_0_10


def _is_effectively_empty(path: Path) -> bool:
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) <= 1


def load_survey_raw(path: Path | None = None) -> pd.DataFrame | None:
    raw_path = path or SURVEY_RAW_CSV
    if _is_effectively_empty(raw_path):
        return None
    df = pd.read_csv(raw_path)
    if df.empty:
        return None
    return df


def normalize_survey(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {
        "n_raw": len(df),
        "dropped_no_consent": 0,
        "dropped_invalid_label": 0,
        "dropped_incomplete": 0,
        "n_kept": 0,
    }
    work = df.copy()

    if "consentement" in work.columns:
        consent_ok = work["consentement"].astype(str).str.lower().isin(
            {"1", "true", "oui", "yes", "y"}
        )
        report["dropped_no_consent"] = int((~consent_ok).sum())
        work = work.loc[consent_ok]

    codes = set(load_parcours_codes())
    if TARGET_COL not in work.columns:
        raise ValueError(f"Colonne manquante: {TARGET_COL}")

    label_ok = work[TARGET_COL].isin(codes)
    report["dropped_invalid_label"] = int((~label_ok).sum())
    work = work.loc[label_ok]

    out = pd.DataFrame(index=work.index)

    for col in SCORE_0_10:
        out[col] = pd.to_numeric(work.get(col), errors="coerce").clip(0, 10)

    out["note_moyenne"] = pd.to_numeric(work.get("note_moyenne"), errors="coerce").clip(0, 20)
    if out["note_moyenne"].isna().any():
        med = out["note_moyenne"].median()
        if pd.isna(med):
            med = 12.0
        out["note_moyenne"] = out["note_moyenne"].fillna(med)

    for col in BINARY_FEATURES:
        if col in work.columns:
            out[col] = (
                pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
        else:
            out[col] = 0

    env = work.get("pref_environnement", pd.Series(["bureau"] * len(work), index=work.index))
    out["pref_environnement"] = env.where(env.isin(ENVIRONNEMENTS), "bureau")

    pref = work.get(
        "pref_professionnelle",
        pd.Series(["developpement_logiciel"] * len(work), index=work.index),
    )
    out["pref_professionnelle"] = pref.where(pref.isin(PREFS_PRO), "developpement_logiciel")

    out[TARGET_COL] = work[TARGET_COL].astype(str)
    pop = work.get("population", pd.Series(["student"] * len(work), index=work.index))
    out["population"] = pop.where(pop.isin(["student", "professional"]), "student")
    out["source_id"] = "survey_v2"
    if "anonymized_id" in work.columns:
        out["anonymized_id"] = work["anonymized_id"].astype(str)
    else:
        out["anonymized_id"] = [f"srv_{i}" for i in range(len(out))]

    incomplete = out[SCORE_0_10].isna().any(axis=1)
    report["dropped_incomplete"] = int(incomplete.sum())
    out = out.loc[~incomplete]

    report["n_kept"] = len(out)
    return out[PROFILE_COLUMNS].reset_index(drop=True), report


def process_survey(
    raw_path: Path | None = None,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    ensure_dirs()
    raw = load_survey_raw(raw_path)
    if raw is None:
        info = {"status": "empty", "message": "Aucune réponse enquête — skip gracieux."}
        empty = pd.DataFrame(columns=PROFILE_COLUMNS)
        dest = out_path or SURVEY_PROCESSED_CSV
        empty.to_csv(dest, index=False)
        return None, info

    normalized, report = normalize_survey(raw)
    dest = out_path or SURVEY_PROCESSED_CSV
    normalized.to_csv(dest, index=False)
    report["status"] = "ok"
    report["out"] = str(dest)
    return normalized, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise l'enquête Orient'IA")
    parser.add_argument("--raw", type=str, default=str(SURVEY_RAW_CSV))
    parser.add_argument("--out", type=str, default=str(SURVEY_PROCESSED_CSV))
    args = parser.parse_args()

    df, report = process_survey(Path(args.raw), Path(args.out))
    print(report)
    if df is not None:
        print(f"Profils enquête normalisés: {len(df)}")


if __name__ == "__main__":
    main()
