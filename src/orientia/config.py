"""Configuration centrale Orient'IA (chemins, seed, labels, features)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_SURVEY_DIR = DATA_DIR / "raw" / "survey"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
CORPUS_DIR = DATA_DIR / "corpus"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

SYNTHETIC_CSV = PROCESSED_DIR / "profiles_synthetic.csv"
SURVEY_PROCESSED_CSV = PROCESSED_DIR / "profiles_survey.csv"
SURVEY_RAW_CSV = RAW_SURVEY_DIR / "responses_anonymized.csv"
PARCOURS_LABELS_PATH = METADATA_DIR / "parcours_labels.json"
CORPUS_STRUCTURED_PATH = CORPUS_DIR / "corpus_structured.json"
CORPUS_CHUNKS_PATH = CORPUS_DIR / "corpus_chunks.jsonl"

RAG_DIR = ARTIFACTS_DIR / "rag"
RAG_INDEX_PATH = RAG_DIR / "tfidf_index.joblib"
TRACES_DIR = ARTIFACTS_DIR / "traces"
TRACES_PATH = TRACES_DIR / "agent_traces.jsonl"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
COMPARISON_REPORT_PATH = METRICS_DIR / "comparison_report.json"
EDA_SUMMARY_PATH = METRICS_DIR / "eda_summary.json"
CONFUSION_MATRIX_PATH = METRICS_DIR / "confusion_matrix.png"

RANDOM_SEED = 42
DEFAULT_N_SYNTHETIC = 3000
TEST_SIZE = 0.2
TOP_K = 3

# Affinités 0–10 dérivées des matières principales du corpus ISPM
NUMERIC_FEATURES = [
    "score_maths",
    "score_prog",
    "score_stats",
    "score_design",
    "score_electronique",
    "score_gestion",
    "score_physique",
    "score_sciences_vie",
    "score_langues",
    "score_droit_eco",
    "note_moyenne",
]

CATEGORICAL_FEATURES = [
    "pref_environnement",
    "pref_professionnelle",
]

BINARY_FEATURES = [
    "comp_python",
    "comp_web",
    "comp_data",
    "comp_hardware",
    "comp_design",
    "comp_gestion",
    "comp_labo",
    "interet_ia",
    "interet_jeux",
    "interet_finance",
    "interet_multimedia",
    "interet_reseaux",
    "interet_recherche",
    "interet_industrie",
    "interet_construction",
    "interet_agro",
    "interet_tourisme",
    "interet_droit",
]

TARGET_COL = "parcours_recommande"
POPULATION_COL = "population"
META_COLS = ["source_id", "anonymized_id", POPULATION_COL]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

ENVIRONNEMENTS = ["bureau", "labo", "terrain", "remote", "startup", "atelier", "chantier"]
PREFS_PRO = [
    "developpement_logiciel",
    "data_science",
    "electronique_embarque",
    "multimedia_ux",
    "gestion_projet_si",
    "recherche_appliquee",
    "industrie_automatisme",
    "chimie_procedes",
    "genie_civil_archi",
    "commerce_gestion",
    "finance_compta",
    "droit_affaires",
    "agroalimentaire",
    "agriculture_elevage",
    "pharmacie",
    "tourisme_hotellerie",
]


def load_parcours_codes() -> list[str]:
    with PARCOURS_LABELS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    codes: list[str] = []
    for mention in data["mentions"]:
        for p in mention["parcours"]:
            codes.append(p["code"])
    return codes


def load_parcours_index() -> list[dict]:
    with PARCOURS_LABELS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for mention in data["mentions"]:
        for p in mention["parcours"]:
            rows.append({**p, "mention": mention["nom"], "mention_code": mention["code"]})
    return rows


def ensure_dirs() -> None:
    for d in (
        PROCESSED_DIR,
        METADATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        RAW_SURVEY_DIR,
        CORPUS_DIR,
        RAG_DIR,
        TRACES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
