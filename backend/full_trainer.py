"""
full_trainer.py — Entraînement de DÉPLOIEMENT sur 100 % des données.

Différence avec `CevitalTuner.train_manual` :
  - train_manual  → split train/val/test, sert à ÉVALUER honnêtement le modèle.
  - train_full    → ré-entraîne le modèle champion sur TOUTES les séquences
                    (train + val + test réunis) pour le mettre en PRODUCTION.

Méthodologie (validée avec l'utilisateur) :
  L'EarlyStopping de l'entraînement normal a déjà trouvé le nombre optimal
  d'époques (= époque du val_loss minimal). On ré-entraîne donc sur 100 % des
  données pour CE nombre d'époques exact, SANS validation ni EarlyStopping.

On hérite de `CevitalTuner` pour réutiliser tel quel `_fit_in_thread`,
`_compute_metrics`, `_send`, et la même infra WebSocket (zéro duplication).
"""
import asyncio
import time

import numpy as np

from tuner import CevitalTuner, build_model_cevital_manual, WebSocketCallback


class CevitalFullTrainer(CevitalTuner):
    """Ré-entraînement sur 100 % des données (modèle de déploiement)."""

    @staticmethod
    def _concat(*arrays):
        """Concatène les morceaux non vides sur l'axe 0 (train+val+test)."""
        parts = [a for a in arrays if a is not None and len(a) > 0]
        if not parts:
            return arrays[0]
        return np.concatenate(parts, axis=0)

    async def train_full(
        self,
        architecture: str,
        embedding_dim: int,
        num_layers: int,
        units,
        dropout_rates,
        learning_rate: float,
        epochs: int,            # = meilleure époque de l'entraînement normal
        batch_size: int,
    ) -> dict:
        await self._send({"type": "log",
                          "message": f"Réentraînement DÉPLOIEMENT {architecture} sur 100 % des données "
                                     f"— {epochs} époques (sans validation ni EarlyStopping)"})
        start = time.time()

        # ── Concaténer TOUTES les séquences : train + val + test ──────
        p = self.pipeline
        X_num = self._concat(p.X_train_num,  p.X_val_num,  p.X_test_num)
        X_cmp = self._concat(p.X_train_comp, p.X_val_comp, p.X_test_comp)
        y_all = self._concat(p.y_train,      p.y_val,      p.y_test)
        w_all = self._concat(p.w_train,      p.w_val,      p.w_test)

        n_total = int(len(y_all))
        await self._send({"type": "log",
                          "message": f"  Données : {n_total} séquences (train+val+test réunis)"})

        # ── Modèle avec hyperparamètres FIXES (ceux du champion) ──────
        model = build_model_cevital_manual(
            architecture     = architecture,
            lookback         = self.lookback,
            n_features       = self.n_features,
            num_classes_comp = self.num_classes_comp,
            embedding_dim    = embedding_dim,
            num_layers       = num_layers,
            units            = units,
            dropout_rates    = dropout_rates,
            learning_rate    = learning_rate,
        )
        await self._send({"type": "log",
                          "message": f"  Paramètres : {model.count_params():,}"})

        # ── Entraînement : exactement `epochs` époques, AUCUNE validation ──
        loop  = asyncio.get_event_loop()
        cb_ws = WebSocketCallback(loop, self._send_fn, epochs)

        history = await self._fit_in_thread(
            model,
            x_inputs      = [X_num, X_cmp],
            y             = y_all,
            sample_weight = w_all,
            val_data      = None,        # ← pas de validation : on entraîne sur tout
            epochs        = epochs,
            batch_size    = batch_size,
            callbacks     = [cb_ws],     # ← pas d'EarlyStopping / ReduceLR (pas de val_loss)
        )

        duration = time.time() - start

        # ── Métriques sur l'ensemble complet (indicatif — données vues à
        #    l'entraînement ; la vraie estimation de généralisation vient de
        #    l'expérience source avec split). ───────────────────────────
        y_pred_days = self.pipeline.predict_with_safety(model, X_num, X_cmp)
        y_true_days = self.pipeline.scaler_y.inverse_transform(
            y_all.reshape(-1, 1)
        ).flatten()
        metrics = self._compute_metrics(y_true_days, y_pred_days)

        hist = history.history
        training_history = [
            {
                "epoch":    i + 1,
                "loss":     round(float(hist["loss"][i]), 6),
                "val_loss": 0.0,     # pas de validation en déploiement
                "mae":      round(float(hist.get("mae", [0])[i] if "mae" in hist else 0), 6),
                "val_mae":  0.0,
            }
            for i in range(len(hist["loss"]))
        ]

        await self._send({
            "type":     "completed",
            "r2":       round(metrics["r2"], 4),
            "mae":      round(metrics["mae"], 3),
            "rmse":     round(metrics["rmse"], 3),
            "mape":     round(metrics["mape"], 2),
            "duration": round(duration, 1),
            "message":  f"\nDéploiement OK — modèle entraîné sur {n_total} séquences "
                        f"({epochs} époques). Prêt pour la prédiction RUL.",
        })

        return {
            "model":            model,
            "metrics":          metrics,
            "duration_sec":     duration,
            "training_history": training_history,
            "y_true":           y_true_days.tolist(),
            "y_pred":           y_pred_days.tolist(),
            "n_sequences":      n_total,
            "epochs_used":      int(epochs),
        }
