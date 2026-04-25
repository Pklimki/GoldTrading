"""
src/model.py

Definice, trénink a uložení LightGBM / XGBoost modelů.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def build_lgbm(class_weight_ratio: Optional[float] = None, **kwargs) -> Any:
    """
    Vytvoří LGBMClassifier.

    class_weight_ratio: poměr neg/pos třídy (len(y==0)/len(y==1)).
                        Předej None pro automatické 'balanced'.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Nainstaluj lightgbm: pip install lightgbm")

    default_params: Dict[str, Any] = {
        "n_estimators": 1000,          # early stopping zastaví dříve
        "learning_rate": 0.03,         # pomalejší učení = robustnější generalizace
        "num_leaves": 31,              # méně listů = méně komplexní model
        "max_depth": -1,
        "min_child_samples": 150,      # každý list ≥150 vzorků – silná ochrana proti overfittingu
        "subsample": 0.8,
        "colsample_bytree": 0.7,       # 70 % features per strom
        "reg_alpha": 0.2,              # L1 regularizace
        "reg_lambda": 0.5,             # L2 regularizace
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if class_weight_ratio is not None:
        default_params["scale_pos_weight"] = class_weight_ratio
    else:
        default_params["class_weight"] = "balanced"

    default_params.update(kwargs)
    return lgb.LGBMClassifier(**default_params)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def build_xgb(class_weight_ratio: Optional[float] = None, **kwargs) -> Any:
    """
    Vytvoří XGBClassifier.

    class_weight_ratio: scale_pos_weight = neg/pos počet (pro nevyvážené třídy).
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("Nainstaluj xgboost: pip install xgboost")

    default_params: Dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if class_weight_ratio is not None:
        default_params["scale_pos_weight"] = class_weight_ratio

    default_params.update(kwargs)
    return xgb.XGBClassifier(**default_params)


# ---------------------------------------------------------------------------
# Uložení a načtení
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[model] Model uložen → {path}")


def load_model(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model nenalezen: {path}")
    model = joblib.load(path)
    print(f"[model] Model načten ← {path}")
    return model


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model: Any, feature_cols: list, top_n: int = 30) -> pd.DataFrame:
    """
    Vrátí DataFrame s feature importance seřazenou sestupně.
    Funguje pro LightGBM i XGBoost.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        raise AttributeError("Model nemá atribut 'feature_importances_'.")

    df_imp = pd.DataFrame({"feature": feature_cols, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp.head(top_n)
