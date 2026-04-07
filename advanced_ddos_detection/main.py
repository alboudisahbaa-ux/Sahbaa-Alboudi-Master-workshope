#!/usr/bin/env python3
"""
Advanced DDoS Detection Pipeline — Main Entry Point
=====================================================
Usage:
    python -m advanced_ddos_detection.main                        # default config
    python -m advanced_ddos_detection.main --config path/to.yaml  # custom config
    python -m advanced_ddos_detection.main --skip-dl              # skip deep learning
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# ── project imports ────────────────────────────────────────────────
from .utils.config import load_config
from .utils.logger import setup_logger
from .data.loader import DataLoader
from .data.eda import DataEDA
from .data.preprocessor import DataPreprocessor
from .data.feature_engineer import FeatureEngineer
from .models.factory import ModelFactory
from .evaluation.evaluator import ModelEvaluator
from .evaluation.cross_validator import CrossValidator
from .evaluation.explainer import ModelExplainer
from .evaluation.error_analysis import ErrorAnalyzer
from .utils.persistence import ModelPersistence


def _set_global_seed(seed: int) -> None:
    """Ensure reproducibility across numpy, random, and TF."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Advanced DDoS Detection Pipeline")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep learning model training")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature importance analysis")
    return parser.parse_args(argv)


def run(cfg: dict, skip_dl: bool = False, skip_features: bool = False) -> None:
    """Execute the full pipeline with the given config dict."""
    logger = logging.getLogger(__name__)

    seed = cfg["project"].get("seed", 42)
    _set_global_seed(seed)
    logger.info("Global seed set to %d", seed)

    # ── 1. Load data ──────────────────────────────────────────────
    loader = DataLoader(cfg)
    df = loader.load()

    # ── 1b. Exploratory Data Analysis (EDA) ───────────────────────
    if cfg.get("eda", {}).get("enabled", True):
        eda = DataEDA(cfg)
        eda_summary = eda.run(df)
        logger.info(
            "EDA summary → rows: %d | cols: %d | missing: %d | inf: %d",
            eda_summary["rows"],
            eda_summary["columns"],
            eda_summary["missing_values"],
            eda_summary["infinite_values"],
        )

    # ── 2. Preprocess ─────────────────────────────────────────────
    preprocessor = DataPreprocessor(cfg)
    split = preprocessor.run(df)
    logger.info("Features: %d | Train: %d | Test: %d",
                len(split.feature_names), len(split.y_train), len(split.y_test))

    # ── 2b. Persist scaler & label encoder ────────────────────────
    output_cfg = cfg.get("output", {})
    save_models = output_cfg.get("save_models", False)
    persistence = None
    if save_models:
        persistence = ModelPersistence(cfg)
        if split.scaler is not None:
            persistence.save_scaler(split.scaler)
        if split.label_encoder is not None:
            persistence.save_label_encoder(split.label_encoder)

    # ── 3. Feature engineering ────────────────────────────────────
    if not skip_features:
        fe = FeatureEngineer(cfg)
        importance = fe.analyse(split)
        reports_dir = cfg.get("output", {}).get("reports_dir")
        if reports_dir:
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
        fe.plot(importance, save_path=f"{reports_dir}/feature_importance.png" if reports_dir else None)
        logger.info("Feature importance:\n%s", importance.to_string(index=False))

    # ── 4. Train ML models ────────────────────────────────────────
    factory = ModelFactory(cfg)
    evaluator = ModelEvaluator(cfg)

    ml_models = factory.create_ml_models()
    trained_ml = []
    trained_ml_map = {}  # name → model (for SHAP/LIME)
    for model in ml_models:
        logger.info("Training %s …", model.name)
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)

        # Try to get probability predictions for ROC/PR curves
        y_proba = None
        if hasattr(model.model, "predict_proba"):
            try:
                y_proba = model.model.predict_proba(split.X_test)
            except Exception:
                pass

        evaluator.evaluate(model.name, split.y_test, y_pred,
                           y_proba=y_proba, best_params=model.get_params())
        trained_ml.append(model)
        trained_ml_map[model.name] = model

        # Save trained ML model
        if persistence and model.model is not None:
            persistence.save_sklearn_model(
                model.model, model.name,
                metrics=evaluator.results.get(model.name),
                feature_names=split.feature_names,
            )

    # ── 5. Ensemble models ────────────────────────────────────────
    if len(trained_ml) >= 2:
        ensembles = factory.create_ensemble(trained_ml)
        for ens in ensembles:
            logger.info("Training %s …", ens.name)
            ens.fit(split.X_train, split.y_train)
            y_pred_ens = ens.predict(split.X_test)
            evaluator.evaluate(ens.name, split.y_test, y_pred_ens,
                               best_params=ens.get_params())
            if persistence and ens.model is not None:
                persistence.save_sklearn_model(
                    ens.model, ens.name,
                    metrics=evaluator.results.get(ens.name),
                    feature_names=split.feature_names,
                )

    # ── 6. Train DL models (MLP, LSTM, CNN1D) ─────────────────────
    if not skip_dl:
        n_classes = len(np.unique(split.y_train))
        dl_models = factory.create_dl_models(
            n_features=split.X_train.shape[1],
            n_classes=n_classes,
        )
        for dl_model in dl_models:
            logger.info("Training %s …", dl_model.name)
            dl_model.fit(split.X_train, split.y_train)
            y_pred_dl = dl_model.predict(split.X_test)

            # DL models output softmax probabilities
            y_proba_dl = None
            if hasattr(dl_model, "model") and dl_model.model is not None:
                try:
                    X_in = split.X_test
                    if dl_model.name in ("DL_LSTM", "DL_CNN1D"):
                        X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], 1)
                    y_proba_dl = dl_model.model.predict(X_in, verbose=0)
                except Exception:
                    pass

            evaluator.evaluate(dl_model.name, split.y_test, y_pred_dl,
                               y_proba=y_proba_dl,
                               best_params=dl_model.get_params())

            # Save trained DL model
            if persistence and dl_model.model is not None:
                persistence.save_keras_model(
                    dl_model.model, dl_model.name,
                    metrics=evaluator.results.get(dl_model.name),
                    feature_names=split.feature_names,
                )

    # ── 7. K-Fold Cross-Validation ──────────────────────────────
    cv_cfg = cfg.get("cross_validation", {})
    if cv_cfg.get("enabled", False):
        cv = CrossValidator(cfg)
        factory2 = ModelFactory(cfg)
        for model in factory2.create_ml_models():
            cv.evaluate(model, split.X_train, split.y_train)
        cv_summary = cv.summary_table()
        print("\nCross-Validation Results:")
        print(cv_summary.to_string(index=False))
        cv.plot_cv_scores(save_dir=reports_dir)

    # ── 8. Results ────────────────────────────────────────────────
    summary = evaluator.summary_table()
    print("\n" + summary.to_string(index=False))
    evaluator.print_reports()

    reports_dir = cfg.get("output", {}).get("reports_dir")
    if reports_dir:
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
    evaluator.plot_confusion_matrices(save_dir=reports_dir)
    evaluator.plot_comparison_bar(save_dir=reports_dir)

    # ROC and PR curves
    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("roc_curves", True):
        evaluator.plot_roc_curves(save_dir=reports_dir)
    if eval_cfg.get("pr_curves", True):
        evaluator.plot_pr_curves(save_dir=reports_dir)

    # ── 9. SHAP / LIME Explainability ─────────────────────────────
    exp_cfg = cfg.get("explainability", {})
    if exp_cfg.get("shap_enabled", False) or exp_cfg.get("lime_enabled", False):
        explainer = ModelExplainer(cfg)
        class_names = list(split.label_encoder.classes_) if split.label_encoder else None

        shap_targets = exp_cfg.get("shap_models", [])
        lime_targets = exp_cfg.get("lime_models", [])

        for mname, mobj in trained_ml_map.items():
            if mname in shap_targets and mobj.model is not None:
                explainer.shap_explain(
                    mname, mobj.model,
                    split.X_train, split.X_test,
                    split.feature_names, save_dir=reports_dir,
                )
            if mname in lime_targets and mobj.model is not None:
                explainer.lime_explain(
                    mname, mobj.model,
                    split.X_train, split.X_test, split.y_test,
                    split.feature_names, class_names=class_names,
                    save_dir=reports_dir,
                )

    # ── 10. Error Analysis ────────────────────────────────────────
    ea_cfg = cfg.get("error_analysis", {})
    if ea_cfg.get("enabled", False):
        class_names = list(split.label_encoder.classes_) if split.label_encoder else None
        error_analyzer = ErrorAnalyzer(cfg)

        # Per-model error analysis for the best ML model
        if evaluator.results:
            best_name = summary.iloc[0]["Model"]
            best_m = evaluator.results[best_name]
            error_analyzer.analyse(
                best_name, best_m["y_true"], best_m["y_pred"],
                split.X_test, split.feature_names,
                class_names=class_names, save_dir=reports_dir,
            )

        # Cross-model error comparison
        error_analyzer.compare_errors(evaluator.results, save_dir=reports_dir)

    logger.info("Pipeline complete.")


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logger(cfg)
    run(cfg, skip_dl=args.skip_dl, skip_features=args.skip_features)


if __name__ == "__main__":
    main()
