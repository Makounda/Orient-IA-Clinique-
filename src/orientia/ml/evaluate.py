"""Métriques, analyse d'erreurs et biais simples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def top_k_accuracy(y_true, y_proba, classes, k: int = 2) -> float:
    """Top-k accuracy avec alignement des classes du classifieur."""
    # y_proba colonnes = classes (ordre du modèle)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # encoder y_true en indices selon classes du modèle
    y_true_idx = np.array([class_to_idx[y] for y in y_true])
    return float(top_k_accuracy_score(y_true_idx, y_proba, k=k, labels=list(range(len(classes)))))


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba=None,
    classes=None,
    k: int = 2,
) -> dict[str, Any]:
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "labels": list(classes) if classes is not None else sorted(pd.Series(y_true).unique()),
    }
    if y_proba is not None and classes is not None and k >= 1:
        try:
            report[f"top_{k}_accuracy"] = top_k_accuracy(y_true, y_proba, classes, k=k)
        except Exception as exc:  # noqa: BLE001
            report[f"top_{k}_accuracy"] = None
            report["top_k_error"] = str(exc)
    return report


def error_analysis(y_true, y_pred, labels) -> dict[str, Any]:
    """Classes les plus confondues (hors diagonale)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append(
                    {
                        "true": true_label,
                        "predicted": pred_label,
                        "count": count,
                    }
                )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return {"top_confusions": pairs[:10], "n_errors": int((np.array(y_true) != np.array(y_pred)).sum())}


def class_imbalance_note(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts().to_dict()
    total = int(y.shape[0])
    freqs = {k: v / total for k, v in counts.items()}
    max_f = max(freqs.values()) if freqs else 0.0
    min_f = min(freqs.values()) if freqs else 0.0
    return {
        "counts": counts,
        "frequencies": freqs,
        "imbalance_ratio_max_min": float(max_f / min_f) if min_f > 0 else None,
        "note": (
            "Déséquilibre modéré à surveiller (macro-F1 plus fiable que l'accuracy)."
            if max_f / min_f > 1.3
            else "Classes relativement équilibrées."
        ),
    }


def eda_summary(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    summary: dict[str, Any] = {
        "n_rows": len(df),
        "label_distribution": df[target_col].value_counts().to_dict(),
        "numeric_describe": df[numeric_cols].describe().to_dict() if numeric_cols else {},
    }
    if "population" in df.columns:
        summary["population_distribution"] = df["population"].value_counts().to_dict()
    return summary
