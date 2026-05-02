"""
src/models.py – Definice a továrna ML modelů.

Používej get_model(name) v train.py pro vytvoření modelu.
Dostupné konfigurace: "default", "conservative"
"""

import lightgbm as lgb

# Registr konfigurací – klíč = jméno, hodnota = dict parametrů
_CONFIGS: dict = {
    "default": dict(
        n_estimators      = 1000,
        learning_rate     = 0.01,
        max_depth         = 6,
        num_leaves        = 31,
        min_child_samples = 50,    # jemnější vzorce v menším vzorku (bylo 20)
        random_state      = 42,
    ),
    "conservative": dict(
        n_estimators  = 1500,
        learning_rate = 0.005,
        max_depth     = 4,
        num_leaves    = 15,
        reg_alpha     = 0.1,
        reg_lambda    = 0.1,
        random_state  = 42,
    ),
}


def get_model(name: str = "default") -> lgb.LGBMClassifier:
    """
    Vrátí instanci modelu podle zadaného jména.

    Parameters
    ----------
    name : str
        Název konfigurace. Dostupné: 'default', 'conservative'.

    Raises
    ------
    ValueError
        Pokud name neexistuje v registru.
    """
    if name not in _CONFIGS:
        available = ", ".join(f"'{k}'" for k in _CONFIGS)
        raise ValueError(
            f"Neznámý model '{name}'. Dostupné: {available}"
        )
    params = _CONFIGS[name]
    return lgb.LGBMClassifier(**params, n_jobs=-1, verbose=-1)


def get_config_params(name: str = "default") -> dict:
    """Vrátí kopii parametrů dané konfigurace (pro logování)."""
    if name not in _CONFIGS:
        available = ", ".join(f"'{k}'" for k in _CONFIGS)
        raise ValueError(
            f"Neznámý model '{name}'. Dostupné: {available}"
        )
    return dict(_CONFIGS[name])


# ── TabularNN – Placeholder pro hluboké učení ─────────────────────────────────

class TabularNN:
    """
    Plně propojená neuronová síť (MLP) pro tabulární data.

    **STAV: PLACEHOLDER – NENÍ IMPLEMENTOVÁNO**

    Rozhraní je záměrně kompatibilní se sklearn:
    ``fit(X, y)`` / ``predict(X)`` / ``predict_proba(X)``

    Plánovaná architektura (PyTorch)::

        input (D features)
          → BatchNorm1d(D)
          → Linear(D → 256) → ReLU → Dropout(0.3)
          → Linear(256 → 128) → ReLU → Dropout(0.3)
          → Linear(128 → 64)  → ReLU
          → Linear(64 → 1)    → Sigmoid
          → p(Long)

    Parametry
    ---------
    input_dim : int
        Počet vstupních features. Nastavuje se automaticky ve fit().
    hidden_dims : tuple[int, ...]
        Velikosti skrytých vrstev. Výchozí: (256, 128, 64).
    dropout : float
        Dropout pravděpodobnost. Výchozí: 0.3.
    lr : float
        Learning rate pro Adam. Výchozí: 1e-3.
    epochs : int
        Počet trénovacích epoch. Výchozí: 50.
    batch_size : int
        Velikost mini-batche. Výchozí: 2048.
    """

    def __init__(
        self,
        input_dim: int = 0,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 2048,
    ) -> None:
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self._net        = None   # torch.nn.Module, inicializuje se ve fit()

    # ── Public API (sklearn-compatible) ────────────────────────────────────────

    def fit(self, X, y, eval_set=None, **kwargs):
        """Trénuje model. Vyžaduje PyTorch – zatím není implementováno."""
        raise NotImplementedError(
            "TabularNN.fit() ještě není implementováno.\n"
            "Kroky pro implementaci:\n"
            "  1. pip install torch\n"
            "  2. Doplň _build_net(), _train_epoch(), fit() v src/models.py"
        )

    def predict(self, X):
        """Predikuje třídu (0/1). Vyžaduje natrénovaný model."""
        raise NotImplementedError("TabularNN.predict() ještě není implementováno.")

    def predict_proba(self, X):
        """Vrátí pravděpodobnosti tříd [[p0, p1], ...]. sklearn-compatible."""
        raise NotImplementedError(
            "TabularNN.predict_proba() ještě není implementováno."
        )

    # ── Private helpers (skeleton pro budoucí implementaci) ────────────────────

    def _build_net(self, input_dim: int):
        """
        Sestaví torch.nn.Sequential z hidden_dims.
        Volá se na začátku fit().

        Příklad implementace::

            import torch.nn as nn
            layers = [nn.BatchNorm1d(input_dim)]
            in_d = input_dim
            for out_d in self.hidden_dims:
                layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(self.dropout)]
                in_d = out_d
            layers.append(nn.Linear(in_d, 1))
            self._net = nn.Sequential(*layers)
        """
        raise NotImplementedError("_build_net() ještě není implementováno.")

    def __repr__(self) -> str:
        return (
            f"TabularNN(hidden_dims={self.hidden_dims}, dropout={self.dropout}, "
            f"lr={self.lr}, epochs={self.epochs}, batch_size={self.batch_size})"
        )


def get_tabular_nn(**kwargs) -> "TabularNN":
    """
    Vrátí novou instanci TabularNN s danými hyperparametry.

    Příklad::

        nn_model = get_tabular_nn(hidden_dims=(512, 256, 128), epochs=100)
    """
    return TabularNN(**kwargs)
