"""Construction du préprocesseur et des jeux train/test."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from orientia.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    SURVEY_PROCESSED_CSV,
    SYNTHETIC_CSV,
    TARGET_COL,
    TEST_SIZE,
)
from orientia.data.survey_loader import process_survey


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def load_datasets(
    synthetic_path=None,
    use_survey_for_test: bool = True,
) -> dict[str, Any]:
    """
    Charge synthétique (+ enquête si dispo).

    Montage sujet :
    - train = synthétique (ou partie train)
    - test = enquête si non vide, sinon hold-out synthétique
    """
    synth_path = synthetic_path or SYNTHETIC_CSV
    synth = pd.read_csv(synth_path)

    survey_df, survey_report = process_survey()
    has_survey = survey_df is not None and len(survey_df) > 0

    info = {
        "n_synthetic": len(synth),
        "survey": survey_report,
        "test_source": None,
    }

    if use_survey_for_test and has_survey and len(survey_df) >= 10:
        X_train = synth[FEATURE_COLS]
        y_train = synth[TARGET_COL]
        X_test = survey_df[FEATURE_COLS]
        y_test = survey_df[TARGET_COL]
        info["test_source"] = "survey"
        # petit hold-out synthétique pour validation interne aussi
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_train,
        )
        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "info": info,
            "synth": synth,
            "survey": survey_df,
        }

    X = synth[FEATURE_COLS]
    y = synth[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    info["test_source"] = "synthetic_holdout"
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": None,
        "y_val": None,
        "X_test": X_test,
        "y_test": y_test,
        "info": info,
        "synth": synth,
        "survey": survey_df,
    }


def wrap_model(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", estimator),
        ]
    )
