"""Prédiction locale (outil futur analyser_profil_ml)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from orientia.config import BEST_MODEL_PATH, FEATURE_COLS, TOP_K
from orientia.data.schema import empty_profile_dict


def load_artifact(path: Path | None = None) -> dict:
    model_path = path or BEST_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Lancez d'abord: python -m orientia.ml.train"
        )
    return joblib.load(model_path)


def profile_to_frame(profile: dict) -> pd.DataFrame:
    base = empty_profile_dict()
    base.update(profile)
    row = {c: base[c] for c in FEATURE_COLS}
    return pd.DataFrame([row])


def predict_profile(profile: dict, top_k: int = TOP_K, artifact_path: Path | None = None) -> dict:
    artifact = load_artifact(artifact_path)
    pipeline = artifact["pipeline"]
    X = profile_to_frame(profile)
    classes = list(pipeline.named_steps["clf"].classes_)
    proba = pipeline.predict_proba(X)[0]
    order = list(np_argsort_desc(proba))
    ranking = [
        {"parcours": classes[i], "score": float(proba[i])}
        for i in order[:top_k]
    ]
    return {
        "model_name": artifact.get("model_name"),
        "prediction": ranking[0]["parcours"],
        "top_k": ranking,
        "source": "ml_model",
        "disclaimer": (
            "Recommandation statistique issue du modèle ML. "
            "Ne remplace ni un conseiller pédagogique ni une décision d'admission."
        ),
    }


def np_argsort_desc(arr):
    import numpy as np

    return np.argsort(arr)[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédit un parcours à partir d'un profil JSON")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help='JSON profil, ex: \'{"score_maths":9,"score_stats":8,"comp_data":1}\'',
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    args = parser.parse_args()

    profile = json.loads(args.profile)
    result = predict_profile(profile, top_k=args.top_k, artifact_path=Path(args.model))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
