"""Génération de profils synthétiques alignés sur le corpus ISPM."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from orientia.config import (
    BINARY_FEATURES,
    DEFAULT_N_SYNTHETIC,
    ENVIRONNEMENTS,
    PREFS_PRO,
    RANDOM_SEED,
    SYNTHETIC_CSV,
    TARGET_COL,
    ensure_dirs,
    load_parcours_codes,
)
from orientia.data.schema import PROFILE_COLUMNS, SCORE_0_10

# Centroïdes dérivés des matières principales du corpus (scores 0–10)
PARCOURS_CENTROIDS: dict[str, dict] = {
    "IGGLIA": {
        "score_maths": 6.5,
        "score_prog": 8.5,
        "score_stats": 5.5,
        "score_design": 4.0,
        "score_electronique": 3.5,
        "score_gestion": 7.5,
        "score_physique": 3.5,
        "score_sciences_vie": 2.5,
        "score_langues": 4.5,
        "score_droit_eco": 5.0,
        "note_moyenne": 13.5,
        "pref_environnement": ["bureau", "startup", "remote"],
        "pref_professionnelle": ["developpement_logiciel", "gestion_projet_si"],
        "binary_high": ["comp_python", "comp_web", "comp_gestion", "interet_ia"],
        "binary_low": ["comp_hardware", "comp_labo", "interet_agro", "interet_tourisme", "interet_construction"],
    },
    "ESIIA": {
        "score_maths": 7.0,
        "score_prog": 6.5,
        "score_stats": 5.0,
        "score_design": 3.0,
        "score_electronique": 8.5,
        "score_gestion": 4.0,
        "score_physique": 7.5,
        "score_sciences_vie": 2.5,
        "score_langues": 4.0,
        "score_droit_eco": 3.0,
        "note_moyenne": 13.0,
        "pref_environnement": ["labo", "atelier", "bureau"],
        "pref_professionnelle": ["electronique_embarque", "recherche_appliquee"],
        "binary_high": ["comp_hardware", "comp_python", "interet_reseaux", "interet_ia", "comp_labo"],
        "binary_low": ["comp_design", "interet_multimedia", "interet_finance", "interet_tourisme", "interet_droit"],
    },
    "IMTICIA": {
        "score_maths": 5.0,
        "score_prog": 6.5,
        "score_stats": 4.5,
        "score_design": 8.5,
        "score_electronique": 3.0,
        "score_gestion": 5.5,
        "score_physique": 3.0,
        "score_sciences_vie": 2.5,
        "score_langues": 6.0,
        "score_droit_eco": 3.5,
        "note_moyenne": 12.5,
        "pref_environnement": ["startup", "remote", "bureau"],
        "pref_professionnelle": ["multimedia_ux", "developpement_logiciel"],
        "binary_high": ["comp_design", "comp_web", "interet_multimedia", "interet_jeux"],
        "binary_low": ["comp_hardware", "comp_data", "interet_finance", "interet_industrie", "interet_agro"],
    },
    "ISAIA": {
        "score_maths": 8.5,
        "score_prog": 6.0,
        "score_stats": 8.5,
        "score_design": 3.5,
        "score_electronique": 3.0,
        "score_gestion": 5.0,
        "score_physique": 4.0,
        "score_sciences_vie": 3.0,
        "score_langues": 4.5,
        "score_droit_eco": 4.5,
        "note_moyenne": 14.0,
        "pref_environnement": ["bureau", "remote", "labo"],
        "pref_professionnelle": ["data_science", "recherche_appliquee"],
        "binary_high": ["comp_data", "comp_python", "interet_ia", "interet_finance", "interet_recherche"],
        "binary_low": ["comp_hardware", "comp_design", "interet_jeux", "interet_tourisme", "interet_construction"],
    },
    "EMII": {
        "score_maths": 7.5,
        "score_prog": 5.5,
        "score_stats": 4.5,
        "score_design": 3.0,
        "score_electronique": 7.5,
        "score_gestion": 4.5,
        "score_physique": 8.5,
        "score_sciences_vie": 3.0,
        "score_langues": 4.0,
        "score_droit_eco": 3.0,
        "note_moyenne": 13.0,
        "pref_environnement": ["atelier", "labo", "terrain"],
        "pref_professionnelle": ["industrie_automatisme", "electronique_embarque"],
        "binary_high": ["comp_hardware", "comp_labo", "interet_industrie", "interet_reseaux"],
        "binary_low": ["comp_design", "interet_multimedia", "interet_tourisme", "interet_droit", "interet_finance"],
    },
    "ICMP": {
        "score_maths": 7.0,
        "score_prog": 3.5,
        "score_stats": 5.0,
        "score_design": 2.5,
        "score_electronique": 4.0,
        "score_gestion": 4.5,
        "score_physique": 7.0,
        "score_sciences_vie": 7.5,
        "score_langues": 4.0,
        "score_droit_eco": 3.5,
        "note_moyenne": 13.0,
        "pref_environnement": ["labo", "terrain", "atelier"],
        "pref_professionnelle": ["chimie_procedes", "industrie_automatisme"],
        "binary_high": ["comp_labo", "interet_industrie", "interet_recherche"],
        "binary_low": ["comp_web", "comp_design", "interet_jeux", "interet_multimedia", "interet_tourisme"],
    },
    "GCA": {
        "score_maths": 7.5,
        "score_prog": 3.5,
        "score_stats": 4.5,
        "score_design": 6.5,
        "score_electronique": 3.0,
        "score_gestion": 5.0,
        "score_physique": 8.0,
        "score_sciences_vie": 3.0,
        "score_langues": 4.5,
        "score_droit_eco": 4.0,
        "note_moyenne": 13.0,
        "pref_environnement": ["chantier", "bureau", "terrain"],
        "pref_professionnelle": ["genie_civil_archi"],
        "binary_high": ["interet_construction", "comp_design", "interet_recherche"],
        "binary_low": ["comp_web", "comp_python", "interet_jeux", "interet_agro", "interet_tourisme"],
    },
    "CAA": {
        "score_maths": 5.5,
        "score_prog": 3.0,
        "score_stats": 5.0,
        "score_design": 4.0,
        "score_electronique": 2.0,
        "score_gestion": 8.5,
        "score_physique": 2.5,
        "score_sciences_vie": 2.5,
        "score_langues": 7.0,
        "score_droit_eco": 8.0,
        "note_moyenne": 12.5,
        "pref_environnement": ["bureau", "startup", "remote"],
        "pref_professionnelle": ["commerce_gestion"],
        "binary_high": ["comp_gestion", "interet_finance", "interet_droit"],
        "binary_low": ["comp_hardware", "comp_labo", "interet_industrie", "interet_agro", "comp_python"],
    },
    "EMP": {
        "score_maths": 6.0,
        "score_prog": 3.0,
        "score_stats": 5.5,
        "score_design": 3.5,
        "score_electronique": 2.0,
        "score_gestion": 8.5,
        "score_physique": 2.5,
        "score_sciences_vie": 2.5,
        "score_langues": 6.5,
        "score_droit_eco": 8.0,
        "note_moyenne": 13.0,
        "pref_environnement": ["bureau", "remote", "startup"],
        "pref_professionnelle": ["commerce_gestion", "gestion_projet_si"],
        "binary_high": ["comp_gestion", "interet_finance", "interet_droit"],
        "binary_low": ["comp_hardware", "comp_labo", "interet_multimedia", "interet_agro"],
    },
    "FIC": {
        "score_maths": 7.0,
        "score_prog": 3.5,
        "score_stats": 7.0,
        "score_design": 3.0,
        "score_electronique": 2.0,
        "score_gestion": 8.0,
        "score_physique": 2.5,
        "score_sciences_vie": 2.5,
        "score_langues": 6.0,
        "score_droit_eco": 8.5,
        "note_moyenne": 13.5,
        "pref_environnement": ["bureau", "remote"],
        "pref_professionnelle": ["finance_compta"],
        "binary_high": ["comp_gestion", "interet_finance", "comp_data"],
        "binary_low": ["comp_hardware", "comp_design", "interet_jeux", "interet_industrie", "interet_tourisme"],
    },
    "DTJA": {
        "score_maths": 4.5,
        "score_prog": 2.5,
        "score_stats": 4.0,
        "score_design": 3.0,
        "score_electronique": 1.5,
        "score_gestion": 7.0,
        "score_physique": 2.0,
        "score_sciences_vie": 2.0,
        "score_langues": 8.0,
        "score_droit_eco": 9.0,
        "note_moyenne": 13.0,
        "pref_environnement": ["bureau"],
        "pref_professionnelle": ["droit_affaires"],
        "binary_high": ["interet_droit", "comp_gestion", "interet_finance"],
        "binary_low": ["comp_python", "comp_hardware", "comp_labo", "interet_ia", "interet_industrie"],
    },
    "IAA": {
        "score_maths": 6.0,
        "score_prog": 3.0,
        "score_stats": 5.0,
        "score_design": 3.0,
        "score_electronique": 3.0,
        "score_gestion": 5.5,
        "score_physique": 5.0,
        "score_sciences_vie": 8.5,
        "score_langues": 4.5,
        "score_droit_eco": 3.5,
        "note_moyenne": 13.0,
        "pref_environnement": ["labo", "atelier", "terrain"],
        "pref_professionnelle": ["agroalimentaire"],
        "binary_high": ["comp_labo", "interet_agro", "interet_industrie"],
        "binary_low": ["comp_web", "interet_jeux", "interet_multimedia", "interet_droit"],
    },
    "AEE": {
        "score_maths": 5.5,
        "score_prog": 2.5,
        "score_stats": 4.5,
        "score_design": 3.0,
        "score_electronique": 2.5,
        "score_gestion": 5.0,
        "score_physique": 4.5,
        "score_sciences_vie": 8.5,
        "score_langues": 4.5,
        "score_droit_eco": 3.5,
        "note_moyenne": 12.5,
        "pref_environnement": ["terrain", "labo"],
        "pref_professionnelle": ["agriculture_elevage"],
        "binary_high": ["interet_agro", "comp_labo"],
        "binary_low": ["comp_web", "comp_python", "interet_jeux", "interet_finance", "interet_multimedia"],
    },
    "PIP": {
        "score_maths": 7.0,
        "score_prog": 3.0,
        "score_stats": 5.5,
        "score_design": 3.0,
        "score_electronique": 3.0,
        "score_gestion": 4.5,
        "score_physique": 5.5,
        "score_sciences_vie": 9.0,
        "score_langues": 5.0,
        "score_droit_eco": 3.5,
        "note_moyenne": 14.0,
        "pref_environnement": ["labo", "bureau"],
        "pref_professionnelle": ["pharmacie", "recherche_appliquee"],
        "binary_high": ["comp_labo", "interet_agro", "interet_recherche"],
        "binary_low": ["comp_web", "interet_jeux", "interet_multimedia", "interet_tourisme", "interet_construction"],
    },
    "TEE": {
        "score_maths": 4.0,
        "score_prog": 2.5,
        "score_stats": 4.0,
        "score_design": 5.0,
        "score_electronique": 2.0,
        "score_gestion": 6.0,
        "score_physique": 3.5,
        "score_sciences_vie": 6.5,
        "score_langues": 8.5,
        "score_droit_eco": 5.0,
        "note_moyenne": 12.5,
        "pref_environnement": ["terrain", "bureau"],
        "pref_professionnelle": ["tourisme_hotellerie"],
        "binary_high": ["interet_tourisme", "comp_gestion"],
        "binary_low": ["comp_python", "comp_hardware", "comp_labo", "interet_ia", "interet_industrie"],
    },
    "TEH": {
        "score_maths": 4.0,
        "score_prog": 2.5,
        "score_stats": 4.0,
        "score_design": 5.5,
        "score_electronique": 2.0,
        "score_gestion": 7.0,
        "score_physique": 3.0,
        "score_sciences_vie": 4.0,
        "score_langues": 8.5,
        "score_droit_eco": 5.5,
        "note_moyenne": 12.5,
        "pref_environnement": ["bureau", "terrain", "startup"],
        "pref_professionnelle": ["tourisme_hotellerie", "commerce_gestion"],
        "binary_high": ["interet_tourisme", "comp_gestion", "interet_multimedia"],
        "binary_low": ["comp_python", "comp_hardware", "comp_data", "interet_industrie", "interet_construction"],
    },
}

# Poids de classes : Informatique un peu plus représentée (historique ISPM), reste quasi uniforme
DEFAULT_CLASS_WEIGHTS = {
    "IGGLIA": 0.12,
    "ESIIA": 0.08,
    "IMTICIA": 0.08,
    "ISAIA": 0.08,
    "EMII": 0.06,
    "ICMP": 0.05,
    "GCA": 0.06,
    "CAA": 0.06,
    "EMP": 0.06,
    "FIC": 0.06,
    "DTJA": 0.05,
    "IAA": 0.06,
    "AEE": 0.05,
    "PIP": 0.05,
    "TEE": 0.04,
    "TEH": 0.04,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _sample_binary(rng: np.random.Generator, high: bool, noise: float = 0.15) -> int:
    p = (0.85 - noise) if high else (0.15 + noise)
    p = float(np.clip(p + rng.normal(0, 0.05), 0.05, 0.95))
    return int(rng.random() < p)


def generate_one(rng: np.random.Generator, parcours: str, noise_scale: float = 1.2) -> dict:
    c = PARCOURS_CENTROIDS[parcours]
    row: dict = {}

    for key in SCORE_0_10:
        row[key] = _clip(rng.normal(c[key], noise_scale), 0.0, 10.0)

    row["note_moyenne"] = _clip(rng.normal(c["note_moyenne"], 1.5 * noise_scale / 1.2), 8.0, 20.0)
    row["pref_environnement"] = str(rng.choice(c["pref_environnement"]))
    if rng.random() < 0.12:
        row["pref_environnement"] = str(rng.choice(ENVIRONNEMENTS))

    row["pref_professionnelle"] = str(rng.choice(c["pref_professionnelle"]))
    if rng.random() < 0.12:
        row["pref_professionnelle"] = str(rng.choice(PREFS_PRO))

    high_set = set(c["binary_high"])
    low_set = set(c["binary_low"])
    for feat in BINARY_FEATURES:
        if feat in high_set:
            row[feat] = _sample_binary(rng, True)
        elif feat in low_set:
            row[feat] = _sample_binary(rng, False)
        else:
            row[feat] = int(rng.random() < 0.35)

    row[TARGET_COL] = parcours
    row["population"] = "synthetic"
    row["source_id"] = "synthetic_corpus_v2"
    row["anonymized_id"] = f"syn_{uuid.uuid4().hex[:12]}"
    return row


def apply_coherence_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    codes = set(load_parcours_codes())
    report = {
        "n_before": len(df),
        "dropped_invalid_label": 0,
        "clipped_numeric": 0,
        "n_after": 0,
    }

    mask_label = df[TARGET_COL].isin(codes)
    report["dropped_invalid_label"] = int((~mask_label).sum())
    df = df.loc[mask_label].copy()

    for col in SCORE_0_10:
        before = df[col].copy()
        df[col] = df[col].clip(0, 10)
        report["clipped_numeric"] += int((before != df[col]).sum())

    before_note = df["note_moyenne"].copy()
    df["note_moyenne"] = df["note_moyenne"].clip(0, 20)
    report["clipped_numeric"] += int((before_note != df["note_moyenne"]).sum())

    for col in BINARY_FEATURES:
        df[col] = df[col].fillna(0).astype(int).clip(0, 1)

    df = df.dropna(subset=[TARGET_COL])
    report["n_after"] = len(df)
    return df[PROFILE_COLUMNS], report


def generate_synthetic(
    n: int = DEFAULT_N_SYNTHETIC,
    seed: int = RANDOM_SEED,
    class_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    codes = load_parcours_codes()
    missing = [c for c in codes if c not in PARCOURS_CENTROIDS]
    if missing:
        raise ValueError(f"Centroïdes manquants pour: {missing}")

    weights = class_weights or DEFAULT_CLASS_WEIGHTS
    probs = np.array([float(weights.get(c, 1.0)) for c in codes], dtype=float)
    probs = probs / probs.sum()
    labels = rng.choice(codes, size=n, p=probs)

    rows = [generate_one(rng, parcours=str(p)) for p in labels]
    df = pd.DataFrame(rows)
    df, coherence = apply_coherence_checks(df)

    n_noise = max(1, int(0.05 * len(df)))
    noise_idx = rng.choice(df.index.to_numpy(), size=n_noise, replace=False)
    for idx in noise_idx:
        other = [c for c in codes if c != df.at[idx, TARGET_COL]]
        df.at[idx, TARGET_COL] = str(rng.choice(other))

    meta = {
        "n": len(df),
        "seed": seed,
        "n_classes": len(codes),
        "class_distribution": df[TARGET_COL].value_counts().to_dict(),
        "label_noise_count": n_noise,
        "coherence": coherence,
        "source_id": "synthetic_corpus_v2",
    }
    return df, meta


def save_synthetic(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or SYNTHETIC_CSV
    df.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère des profils synthétiques Orient'IA")
    parser.add_argument("--n", type=int, default=DEFAULT_N_SYNTHETIC)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=str, default=str(SYNTHETIC_CSV))
    args = parser.parse_args()

    df, meta = generate_synthetic(n=args.n, seed=args.seed)
    out = save_synthetic(df, Path(args.out))
    print(f"Écrit {meta['n']} profils ({meta['n_classes']} parcours) → {out}")
    print("Distribution:", meta["class_distribution"])
    print("Label noise:", meta["label_noise_count"])


if __name__ == "__main__":
    main()
