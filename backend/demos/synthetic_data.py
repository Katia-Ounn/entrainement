"""
synthetic_data.py — Mini-dataset synthétique pour les démos pédagogiques.

Pourquoi :
  Les démos RNN/LSTM/Transformer affichent pas-à-pas les calculs sur un petit
  batch de données. Auparavant, elles consommaient le pipeline Azure
  (Machine 99). En Partie 2, on rend les démos AUTONOMES : elles tournent
  sans dépendre du pipeline Cevital ni de l'ancien pipeline Azure.

  Ce module génère un mini-dataset déterministe (seed fixe) qui imite
  l'interface des pipelines réels : X_train, y_train, feature_cols, scaler_y.

  Caractéristiques du dataset :
    - 10 séquences (batch)
    - 24 pas de temps par séquence
    - 4 features : "volt", "rotate", "pressure", "vibration"
      (mêmes noms que l'ancien dataset pour que les démos retrouvent leurs
       features cibles dans `build_demo_batch`)
    - Signaux : sinusoïdes décalées + bruit gaussien léger
    - Target y = moyenne future (1 valeur dénormalisée entre 0 et 1)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Noms de features que les démos cherchent dans `build_demo_batch`
DEMO_FEATURE_COLS: List[str] = ["volt", "rotate", "pressure", "vibration"]


class SyntheticPipeline:
    """
    Pipeline factice — interface minimale attendue par rnn_demo / lstm_demo
    / transformer_demo dans leur fonction `build_demo_batch(pipeline, ...)`.

    Expose :
        X_train      : (n_samples, seq_len, n_features)
        y_train      : (n_samples, 1)
        feature_cols : list[str]
        scaler_y     : MinMaxScaler (fitté)
    """

    def __init__(
        self,
        n_samples: int = 10,
        seq_len: int = 24,
        feature_cols: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.feature_cols: List[str] = feature_cols or DEMO_FEATURE_COLS
        self.n_samples: int = n_samples
        self.seq_len:   int = seq_len
        self.seed:      int = seed

        rng = np.random.default_rng(seed)
        n_feat = len(self.feature_cols)

        # ── Génération des séquences synthétiques ──────────────
        # Chaque feature est une sinusoïde décalée + bruit gaussien
        t = np.linspace(0.0, 4 * np.pi, seq_len)
        X = np.zeros((n_samples, seq_len, n_feat), dtype=np.float32)

        for i in range(n_samples):
            for f in range(n_feat):
                phase     = (f + 1) * 0.5 + i * 0.1
                amplitude = 0.5 + 0.1 * rng.standard_normal()
                offset    = 0.5 + 0.05 * f
                noise     = 0.05 * rng.standard_normal(seq_len)
                X[i, :, f] = offset + amplitude * np.sin(t + phase) + noise

        # Renormaliser X dans [0, 1] feature par feature (comme un MinMaxScaler)
        X_min = X.min(axis=(0, 1), keepdims=True)
        X_max = X.max(axis=(0, 1), keepdims=True)
        X = (X - X_min) / (X_max - X_min + 1e-9)

        # ── Target y : moyenne des valeurs du dernier pas, normalisée ───
        # Cible pédagogique simple = moyenne future ; reste dans [0, 1].
        y_raw = X[:, -1, :].mean(axis=1).reshape(-1, 1).astype(np.float32)

        # On fitte un scaler y identité (sur des valeurs déjà dans [0, 1])
        # → permet aux démos d'appeler scaler_y.inverse_transform() comme avec
        # un vrai pipeline, sans changer leur code.
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_raw)

        # ── Attributs publics utilisés par les démos ───────────
        self.X_train: np.ndarray = X.astype(np.float32)
        self.y_train: np.ndarray = scaler_y.transform(y_raw).astype(np.float32)
        self.scaler_y: MinMaxScaler = scaler_y

    # ─── Helpers (optionnels — pratique pour debug) ───
    def info(self) -> dict:
        return {
            "kind":          "synthetic-demo-pipeline",
            "n_samples":     int(self.n_samples),
            "seq_len":       int(self.seq_len),
            "n_features":    int(len(self.feature_cols)),
            "feature_cols":  list(self.feature_cols),
            "X_train_shape": list(self.X_train.shape),
            "y_train_shape": list(self.y_train.shape),
            "seed":          int(self.seed),
        }


# ─── Cache module-level pour ne pas régénérer à chaque appel ───
_CACHED: Optional[SyntheticPipeline] = None


def get_synthetic_pipeline(force_new: bool = False) -> SyntheticPipeline:
    """
    Retourne une instance partagée de `SyntheticPipeline`.
    Utiliser `force_new=True` pour ré-instancier (utile pour les tests).
    """
    global _CACHED
    if _CACHED is None or force_new:
        _CACHED = SyntheticPipeline()
    return _CACHED
