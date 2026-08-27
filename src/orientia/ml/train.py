"""Entraînement et comparaison baseline + RandomForest + HistGradientBoosting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from orientia.config import (
    BEST_MODEL_PATH,
    COMPARISON_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    EDA_SUMMARY_PATH,
    RANDOM_SEED,
    TARGET_COL,
    TOP_K,
    ensure_dirs,
    load_parcours_codes,
)
from orientia.ml.evaluate import (
    class_imbalance_note,
    eda_summary,
    error_analysis,
    evaluate_predictions,
)
from orientia.ml.features import load_datasets, wrap_model


def build_candidates(seed: int = RANDOM_SEED) -> dict:
    return {
        "baseline_logreg": wrap_model(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )
        ),
        "random_forest": wrap_model(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": wrap_model(
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                max_iter=200,
                random_state=seed,
            )
        ),
    }


def eval_split(model, X, y, k: int = TOP_K) -> dict:
    y_pred = model.predict(X)
    classes = list(model.named_steps["clf"].classes_)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    metrics = evaluate_predictions(y, y_pred, y_proba, classes=classes, k=k)
    metrics["errors"] = error_analysis(y, y_pred, labels=classes)
    return metrics


def plot_confusion(cm, labels, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Vrai",
        xlabel="Prédit",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = np.max(cm) / 2.0 if np.max(cm) > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_and_compare(seed: int = RANDOM_SEED) -> dict:
    ensure_dirs()
    data = load_datasets()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    eda = eda_summary(data["synth"], TARGET_COL)
    eda["split_info"] = data["info"]
    eda["train_imbalance"] = class_imbalance_note(y_train)
    EDA_SUMMARY_PATH.write_text(json.dumps(eda, indent=2, ensure_ascii=False), encoding="utf-8")

    candidates = build_candidates(seed)
    results = {}
    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        test_metrics = eval_split(model, X_test, y_test)
        entry = {"test": test_metrics}
        if data["X_val"] is not None:
            entry["val"] = eval_split(model, data["X_val"], data["y_val"])
        results[name] = entry
        score = test_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    joblib.dump(
        {
            "pipeline": best_model,
            "model_name": best_name,
            "feature_cols": list(X_train.columns),
            "parcours": load_parcours_codes(),
            "test_source": data["info"]["test_source"],
        },
        BEST_MODEL_PATH,
    )

    best_cm = np.array(results[best_name]["test"]["confusion_matrix"])
    best_labels = results[best_name]["test"]["labels"]
    plot_confusion(
        best_cm,
        best_labels,
        CONFUSION_MATRIX_PATH,
        title=f"Matrice de confusion — {best_name}",
    )

    report = {
        "best_model": best_name,
        "best_macro_f1": best_score,
        "selection_criterion": "macro_f1 on test",
        "test_source": data["info"]["test_source"],
        "models": results,
        "artifacts": {
            "best_model": str(BEST_MODEL_PATH),
            "confusion_matrix": str(CONFUSION_MATRIX_PATH),
            "eda": str(EDA_SUMMARY_PATH),
        },
        "limits": [
            "Entraînement majoritairement sur données synthétiques dérivées du corpus : risque de sur-ajustement aux centroïdes.",
            "Sans enquête réelle suffisante, la généralisation vers profils humains n'est pas encore mesurée.",
            "Catalogue élargi à 16 parcours ISPM ; détail corpus inégal (IGGLIA riche, ICMP peu documenté).",
        ],
    }
    COMPARISON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne et compare les modèles Orient'IA")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    report = train_and_compare(seed=args.seed)
    print(f"Meilleur modèle: {report['best_model']} (macro_f1={report['best_macro_f1']:.4f})")
    print(f"Source test: {report['test_source']}")
    print(f"Rapport: {COMPARISON_REPORT_PATH}")
    print(f"Modèle: {BEST_MODEL_PATH}")
    for name, entry in report["models"].items():
        t = entry["test"]
        top_key = f"top_{TOP_K}_accuracy"
        topk = t.get(top_key)
        topk_s = f"{topk:.4f}" if isinstance(topk, float) else "n/a"
        print(
            f"  - {name}: acc={t['accuracy']:.4f} macro_f1={t['macro_f1']:.4f} {top_key}={topk_s}"
        )


if __name__ == "__main__":
    main()
