"""
Démos pédagogiques autonomes (RNN / LSTM / Transformer).

Toutes les démos consomment un objet "pipeline-like" exposant :
    pipeline.X_train      : ndarray (n, seq_len, n_features)
    pipeline.y_train      : ndarray (n,) ou (n, 1)
    pipeline.feature_cols : list[str]
    pipeline.scaler_y     : MinMaxScaler

En Partie 2, ces démos n'ont AUCUNE dépendance vers l'ancien pipeline
Azure ni vers le pipeline Cevital : elles utilisent un mini-dataset
synthétique généré côté serveur par `synthetic_data.get_synthetic_pipeline()`.
"""
from .synthetic_data import (
    SyntheticPipeline,
    get_synthetic_pipeline,
)

__all__ = ["SyntheticPipeline", "get_synthetic_pipeline"]
