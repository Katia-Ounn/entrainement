"""
tuner.py v3 — Fix visualisation temps réel
Problème : model.fit() bloquait l'event loop → les messages WS
           étaient mis en queue et envoyés seulement à la fin.
Solution : model.fit() s'exécute dans un ThreadPoolExecutor via
           loop.run_in_executor() → l'event loop reste libre pour
           traiter les envois WebSocket à chaque époque.
"""
import os, time, math, asyncio, concurrent.futures
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dense, Dropout, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add,
    Embedding, Flatten, RepeatVector, Concatenate,
)
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import gc
import keras_tuner as kt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


# ─────────────────────────────────────────────────────────────
# Callback — s'exécute dans le thread TF
# Utilise run_coroutine_threadsafe pour envoyer au WS
# ─────────────────────────────────────────────────────────────
class WebSocketCallback(Callback):
    """
    Envoie les métriques via WebSocket à chaque époque.
    Fonctionne même quand model.fit() tourne dans un thread séparé.

    🆕 Implémente __deepcopy__ : Keras Tuner deepcopie les callbacks pour
    chaque trial, et notre callback contient des références non-copiables
    (event loop asyncio + closure async). On retourne donc une nouvelle
    instance partageant les mêmes refs (équivalent à no-op pour notre cas).
    """
    def __init__(self, loop, send_coro_fn, total_epochs, trial_id=None):
        super().__init__()
        self._loop         = loop           # event loop principal
        self._send_coro_fn = send_coro_fn   # fonction qui renvoie une coroutine
        self.total_epochs  = total_epochs
        self.trial_id      = trial_id
        self._epoch_start  = 0

    def __deepcopy__(self, memo):
        # Keras Tuner appelle copy.deepcopy(callbacks) avant chaque trial.
        # On crée juste une nouvelle instance avec les MÊMES refs partagées.
        new_cb = WebSocketCallback(
            self._loop, self._send_coro_fn, self.total_epochs, self.trial_id
        )
        return new_cb

    def _emit(self, payload):
        """Envoie depuis le thread TF vers l'event loop principal."""
        try:
            coro = self._send_coro_fn(payload)
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        elapsed = round(time.time() - self._epoch_start, 1)
        self._emit({
            "type":     "epoch",
            "epoch":    epoch + 1,
            "total":    self.total_epochs,
            "trial":    self.trial_id,
            "elapsed":  elapsed,
            "loss":     round(float(logs.get("loss",     0)), 6),
            "val_loss": round(float(logs.get("val_loss", 0)), 6),
            "mae":      round(float(logs.get("mae",      0)), 6),
            "val_mae":  round(float(logs.get("val_mae",  0)), 6),
        })


# ─────────────────────────────────────────────────────────────
# Construction des modèles
# ─────────────────────────────────────────────────────────────
def build_recurrent_model(architecture, n_timesteps, n_features,
                           num_layers, units, dropout_rates, learning_rate):
    layer_map = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}
    RNNLayer  = layer_map.get(architecture, LSTM)
    model = Sequential(name=f"{architecture}_model")
    model.add(Input(shape=(n_timesteps, n_features)))
    for i in range(num_layers):
        model.add(RNNLayer(units=units[i], return_sequences=(i < num_layers - 1)))
        model.add(Dropout(rate=dropout_rates[i]))
    model.add(Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mae"]
    )
    return model


def build_transformer_model(n_timesteps, n_features, num_layers,
                              d_model, num_heads, dropout_rate, learning_rate):
    inputs = Input(shape=(n_timesteps, n_features))
    x = Dense(d_model)(inputs)
    for _ in range(num_layers):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        attn = Dropout(dropout_rate)(attn)
        x    = LayerNormalization()(Add()([x, attn]))
        ff   = Dense(d_model * 2, activation="relu")(x)
        ff   = Dense(d_model)(ff)
        ff   = Dropout(dropout_rate)(ff)
        x    = LayerNormalization()(Add()([x, ff]))
    x      = GlobalAveragePooling1D()(x)
    output = Dense(1, activation="linear")(x)
    model  = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────
class PDMTuner:
    def __init__(self, X_train, y_train, X_test, y_test,
                 scaler_y, exports_dir, experiment_name, send_fn=None):
        self.X_train         = X_train
        self.y_train         = y_train
        self.X_test          = X_test
        self.y_test          = y_test
        self.scaler_y        = scaler_y
        self.exports_dir     = exports_dir
        self.experiment_name = experiment_name
        self.safe_name       = experiment_name.replace(" ", "_").replace("/", "-")
        # send_fn est la version ASYNC (coroutine) venant de main.py
        self._send_fn        = send_fn or (lambda x: None)
        self.n_timesteps     = X_train.shape[1]
        self.n_features      = X_train.shape[2]
        os.makedirs(exports_dir, exist_ok=True)

    async def _send(self, payload: dict):
        """Envoi async depuis le code principal."""
        try:
            await self._send_fn(payload)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # Exécute model.fit dans un thread → libère l'event loop
    # ─────────────────────────────────────────────────────────
    async def _fit_in_thread(self, model, X, y, epochs, batch_size,
                              val_data, callbacks):
        """
        Lance model.fit dans un ThreadPoolExecutor.
        L'event loop asyncio reste libre → les messages WebSocket
        partent à chaque époque sans attendre la fin.
        """
        loop = asyncio.get_event_loop()

        def _blocking_fit():
            return model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            history = await loop.run_in_executor(executor, _blocking_fit)
        return history

    # ─────────────────────────────────────────────────────────
    # MODE MANUEL
    # ─────────────────────────────────────────────────────────
    async def train_manual(self, architecture, num_layers, units,
                            dropout_rates, learning_rate, epochs,
                            batch_size, patience=10):
        await self._send({"type": "log",
                          "message": f"Entrainement {architecture} (Mode Manuel)"})
        start = time.time()

        if architecture == "Transformer":
            model = build_transformer_model(
                self.n_timesteps, self.n_features, num_layers,
                units[0], 4, dropout_rates[0], learning_rate
            )
        else:
            model = build_recurrent_model(
                architecture, self.n_timesteps, self.n_features,
                num_layers, units, dropout_rates, learning_rate
            )

        await self._send({"type": "log",
                          "message": f"   Parametres : {model.count_params():,}"})

        # Récupérer l'event loop pour le callback
        loop   = asyncio.get_event_loop()
        cb_ws  = WebSocketCallback(loop, self._send_fn, epochs)
        cb_es  = EarlyStopping(monitor="val_loss", patience=patience,
                               restore_best_weights=True)

        # model.fit dans un thread → event loop libre pour les WS
        history = await self._fit_in_thread(
            model, self.X_train, self.y_train, epochs, batch_size,
            val_data=(self.X_test, self.y_test),
            callbacks=[cb_ws, cb_es],
        )

        result = await self._evaluate_and_save(model, history, start)
        result["hyperparameters"] = {
            "architecture":  architecture,
            "num_layers":    num_layers,
            "units":         units,
            "dropout":       dropout_rates,
            "learning_rate": learning_rate,
            "epochs":        epochs,
            "batch_size":    batch_size,
        }
        return result

    # ─────────────────────────────────────────────────────────
    # MODE AUTOMATIQUE — Bayesian + TimeSeriesSplit
    # ─────────────────────────────────────────────────────────
    async def train_auto(self, architecture,
                          layers_min=1, layers_max=4,
                          units_min=32, units_max=256, units_step=32,
                          dropout_min=0.1, dropout_max=0.5,
                          lr_choices=None, max_trials=10, cv_folds=5,
                          epochs_per_trial=20, final_epochs=50, batch_size=32):

        if lr_choices is None:
            lr_choices = [1e-2, 1e-3, 1e-4]

        await self._send({"type": "log",
                          "message": f"AutoML {architecture} | {max_trials} essais | {cv_folds} plis CV"})
        start  = time.time()
        loop   = asyncio.get_event_loop()
        n_ts, n_ft = self.n_timesteps, self.n_features

        def build_model(hp):
            nl    = hp.Int("num_layers", layers_min, layers_max)
            units = [hp.Int(f"units_{i}", units_min, units_max, step=units_step) for i in range(nl)]
            drops = [hp.Float(f"dropout_{i}", dropout_min, dropout_max, step=0.1) for i in range(nl)]
            lr    = hp.Choice("lr", lr_choices)
            if architecture == "Transformer":
                return build_transformer_model(
                    n_ts, n_ft, nl, units[0],
                    hp.Choice("num_heads", [2, 4, 8]), drops[0], lr
                )
            return build_recurrent_model(architecture, n_ts, n_ft, nl, units, drops, lr)

        tuner_dir = os.path.join(self.exports_dir, "tuning", self.safe_name)
        tuner = kt.BayesianOptimization(
            build_model, objective="val_loss", max_trials=max_trials,
            directory=tuner_dir,
            project_name=f"{self.safe_name}_{architecture.lower()}",
            overwrite=True,
        )

        tscv   = TimeSeriesSplit(n_splits=cv_folds)
        X_full = np.concatenate([self.X_train, self.X_test], axis=0)
        y_full = np.concatenate([self.y_train, self.y_test], axis=0)
        trial_results = []

        for trial_num in range(max_trials):
            await self._send({
                "type":    "trial_start",
                "trial":   trial_num + 1,
                "total":   max_trials,
                "message": f"Essai {trial_num+1}/{max_trials} en cours...",
            })
            t_start   = time.time()
            cv_scores = []

            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_full)):
                await self._send({"type": "log",
                                  "message": f"   Pli {fold+1}/{cv_folds}"})
                try:
                    hp_obj = kt.HyperParameters()
                    _model = build_model(hp_obj)
                    _es    = EarlyStopping(monitor="val_loss", patience=5,
                                           restore_best_weights=True)
                    _hist  = await self._fit_in_thread(
                        _model, X_full[tr_idx], y_full[tr_idx],
                        epochs_per_trial, batch_size,
                        val_data=(X_full[val_idx], y_full[val_idx]),
                        callbacks=[_es],
                    )
                    cv_scores.append(min(_hist.history.get("val_loss", [999])))
                    del _model
                except Exception:
                    pass

            avg_cv = float(np.mean(cv_scores)) if cv_scores else 999.0
            dur    = round(time.time() - t_start, 1)
            trial_results.append({
                "trial": trial_num + 1,
                "avg_cv_loss": round(avg_cv, 6),
                "duration_sec": dur,
            })
            await self._send({
                "type":        "trial_end",
                "trial":       trial_num + 1,
                "total":       max_trials,
                "avg_cv_loss": round(avg_cv, 6),
                "duration":    dur,
                "message":     f"   Essai {trial_num+1}/{max_trials} — CV Loss: {avg_cv:.4f}",
            })

        await self._send({"type": "log",
                          "message": "Recherche Keras Tuner (entraînement final)..."})

        # Keras Tuner search dans un thread
        def _tuner_search():
            tuner.search(
                self.X_train, self.y_train,
                epochs=epochs_per_trial, validation_split=0.2,
                callbacks=[EarlyStopping("val_loss", patience=5)],
                verbose=0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, _tuner_search)

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hps   = tuner.get_best_hyperparameters(num_trials=1)[0]

        await self._send({"type": "log",
                          "message": "Entraînement final du meilleur modèle..."})

        cb_ws   = WebSocketCallback(loop, self._send_fn, final_epochs)
        cb_es   = EarlyStopping("val_loss", patience=10, restore_best_weights=True)
        history = await self._fit_in_thread(
            best_model, self.X_train, self.y_train,
            final_epochs, batch_size,
            val_data=(self.X_test, self.y_test),
            callbacks=[cb_ws, cb_es],
        )

        result = await self._evaluate_and_save(best_model, history, start)
        n_l    = best_hps.get("num_layers")
        result["hyperparameters"] = {
            "architecture":  architecture,
            "num_layers":    n_l,
            "units":         [best_hps.get(f"units_{i}") for i in range(n_l)],
            "dropout":       [best_hps.get(f"dropout_{i}") for i in range(n_l)],
            "learning_rate": best_hps.get("lr"),
            "epochs":        final_epochs,
            "batch_size":    batch_size,
        }
        result["total_trials"]  = max_trials
        result["cv_folds"]      = cv_folds
        result["trial_results"] = trial_results
        return result

    # ─────────────────────────────────────────────────────────
    # Évaluation + sauvegarde
    # ─────────────────────────────────────────────────────────
    async def _evaluate_and_save(self, model, history, start_time):
        # Prédictions (espace normalisé)
        y_pred_s  = model.predict(self.X_test, verbose=0)

        # Dénormalisation → valeurs en HEURES réelles
        y_pred_h  = self.scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).flatten()
        y_true_h  = self.scaler_y.inverse_transform(self.y_test.reshape(-1,1)).flatten()

        # ── MÉTRIQUES PRINCIPALES : calculées sur valeurs DÉNORMALISÉES (heures) ──
        # Bien plus interprétable pour le prof et l'utilisateur
        r2        = float(r2_score(y_true_h, y_pred_h))
        mae_hours = float(mean_absolute_error(y_true_h, y_pred_h))
        rmse_hours = float(np.sqrt(np.mean((y_true_h - y_pred_h) ** 2)))

        # MAE normalisée gardée pour info (valeurs entre 0 et 1)
        mae_norm  = float(mean_absolute_error(self.y_test.flatten(), y_pred_s.flatten()))

        duration  = time.time() - start_time

        # Format .keras (plus .h5 déprécié)
        model_path = os.path.join(self.exports_dir, f"model_{self.safe_name}.keras")
        model.save(model_path)

        training_history = {
            k: [round(float(v), 6) for v in vals]
            for k, vals in history.history.items()
        }

        # Données prédiction pour le frontend
        n = min(200, len(y_true_h))
        predictions_data = {
            "y_true":     [round(float(v), 2) for v in y_true_h[:n]],
            "y_pred":     [round(float(v), 2) for v in y_pred_h[:n]],
            "errors":     [round(float(abs(y_pred_h[i] - y_true_h[i])), 2) for i in range(n)],
            "mae_hours":  round(mae_hours, 2),
            "rmse_hours": round(rmse_hours, 2),
            "r2_score":   round(r2, 4),
        }

        await self._send({
            "type":       "result",
            "r2":         round(r2, 4),
            "mae":        round(mae_norm, 4),
            "rmse":       round(rmse_hours, 4),
            "mae_hours":  round(mae_hours, 2),
            "rmse_hours": round(rmse_hours, 2),
            "duration":   round(duration, 1),
            "message":    f"\nR²={r2:.4f} | MAE={mae_hours:.2f}h | RMSE={rmse_hours:.2f}h | Durée={duration:.1f}s",
        })

        return {
            "r2_score":         r2,
            "mae":              mae_norm,
            "rmse":             rmse_hours,
            "mae_hours":        mae_hours,
            "rmse_hours":       rmse_hours,
            "model_path":       model_path,
            "training_history": training_history,
            "duration_sec":     duration,
            "predictions_data": predictions_data,
        }


# ═══════════════════════════════════════════════════════════════════════
# ┌──────────────────────────────────────────────────────────────────┐
# │  CEVITAL — Modèle dual-input (X_num + X_comp) avec Embedding      │
# │  Reproduit la build_model() du notebook PFE_Cevital_CHAMPION     │
# └──────────────────────────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════════════

# Loss asymétrique — notebook PFE exact
# sous-estimation (y_true > y_pred) → erreur ×4 (plus pénalisée)
# sur-estimation  (y_true < y_pred) → erreur ×1 (poids de base)
def asymmetric_rul_loss(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(
        error < 0,
        tf.square(error) * 4.0,
        tf.square(error) * 1.0,
    ))


def build_model_cevital_manual(
    architecture: str,
    lookback: int,
    n_features: int,
    num_classes_comp: int,
    embedding_dim: int,
    num_layers: int,
    units: list,
    dropout_rates: list,
    learning_rate: float,
):
    """
    Architecture notebook PFE :
      input_cat (lookback,) → Embedding → (lookback, embed_dim)
      input_num (lookback, n_features)
      → Concat axis=-1 → (lookback, embed_dim+n_features)
      → LSTM/GRU → BatchNorm → Dropout → Dense(32,relu) → Dense(1,linear)
    """
    if architecture not in ("LSTM", "GRU"):
        raise ValueError(f"Architecture non supportée pour Cevital : {architecture}")
    LayerCls = LSTM if architecture == "LSTM" else GRU

    from tensorflow.keras.layers import BatchNormalization

    input_num  = Input(shape=(lookback, n_features), name="input_num")
    input_comp = Input(shape=(lookback,),             name="input_comp")  # (lookback,)

    x_emb = Embedding(num_classes_comp, embedding_dim, name="comp_embedding")(input_comp)
    # x_emb : (batch, lookback, embed_dim)
    x = Concatenate(axis=-1, name="concat_inputs")([x_emb, input_num])
    # x    : (batch, lookback, embed_dim + n_features)

    for i in range(num_layers):
        x = LayerCls(
            units=int(units[i]),
            return_sequences=(i < num_layers - 1),
            name=f"{architecture.lower()}_{i+1}",
        )(x)
        x = BatchNormalization(name=f"bn_{i+1}")(x)
        x = Dropout(rate=float(dropout_rates[i]), name=f"dropout_{i+1}")(x)

    x      = Dense(32, activation="relu", name="dense_hidden")(x)
    output = Dense(1,  activation="linear", name="dense_rul")(x)

    model = Model(inputs=[input_num, input_comp], outputs=output,
                  name=f"cevital_{architecture}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=asymmetric_rul_loss,
        metrics=["mae"],
    )
    return model


def build_model_cevital_hp(
    hp,
    architecture: str,
    lookback: int,
    n_features: int,
    num_classes_comp: int,
    embedding_search=(4, 8, 16, 32),
    units_search=(64, 128, 32),
    nb_layers_min: int = 1,
    nb_layers_max: int = 2,
    dropout_min:   float = 0.1,
    dropout_max:   float = 0.25,
    lr_choices=(1e-3, 2e-3),
):
    """
    Version pour keras_tuner — même architecture que le notebook (BatchNorm, Dense(32), linear).
    input_comp shape (lookback,) — X_cat notebook style.
    """
    if architecture not in ("LSTM", "GRU"):
        raise ValueError(f"Architecture non supportée : {architecture}")
    LayerCls = LSTM if architecture == "LSTM" else GRU

    from tensorflow.keras.layers import BatchNormalization

    embedding_dim = hp.Choice("embedding_dim", list(embedding_search))
    nb_layers     = hp.Int("nb_layers", int(nb_layers_min), int(nb_layers_max))

    input_num  = Input(shape=(lookback, n_features), name="input_num")
    input_comp = Input(shape=(lookback,),             name="input_comp")

    x_emb = Embedding(num_classes_comp, embedding_dim)(input_comp)
    x     = Concatenate(axis=-1)([x_emb, input_num])

    for i in range(nb_layers):
        x = LayerCls(
            units=hp.Int(f"u_{i}", units_search[0], units_search[1], step=units_search[2]),
            return_sequences=(i < nb_layers - 1),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float(f"d_{i}", float(dropout_min), float(dropout_max)))(x)

    x      = Dense(32, activation="relu")(x)
    output = Dense(1,  activation="linear")(x)

    model = Model(inputs=[input_num, input_comp], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("lr", list(lr_choices))),
        loss=asymmetric_rul_loss,
        metrics=["mae"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# CevitalTuner — Orchestration de l'entraînement Cevital (manuel + auto)
# ═══════════════════════════════════════════════════════════════════════
class CevitalTuner:
    """
    Utilise un `CevitalPipeline` déjà préparé (sequences générées) et
    entraîne un modèle LSTM/GRU avec embedding composant.

    Pré-requis du pipeline :
        X_train_num, X_train_comp, y_train, w_train
        X_test_num,  X_test_comp,  y_test
        scaler_y, num_classes_comp, lookback
    """

    def __init__(self, pipeline, exports_dir: str, experiment_name: str,
                 send_fn=None):
        self.pipeline    = pipeline
        self.exports_dir = exports_dir
        self.experiment_name = experiment_name
        self.safe_name   = experiment_name.replace(" ", "_").replace("/", "-")
        self._send_fn    = send_fn or (lambda x: None)

        if pipeline.X_train_num is None:
            raise RuntimeError(
                "Pipeline non prêt — lance `prepare_sequences()` d'abord "
                "(prétraitement obligatoire avant entraînement)."
            )

        self.lookback         = int(pipeline.lookback)
        self.n_features       = int(pipeline.X_train_num.shape[2])
        self.num_classes_comp = int(pipeline.num_classes_comp)
        os.makedirs(exports_dir, exist_ok=True)

    async def _send(self, payload: dict):
        try:
            await self._send_fn(payload)
        except Exception:
            pass

    async def _fit_in_thread(self, model, x_inputs, y, sample_weight,
                              val_data, epochs, batch_size, callbacks):
        loop = asyncio.get_event_loop()

        def _blocking_fit():
            return model.fit(
                x_inputs, y,
                sample_weight=sample_weight,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _blocking_fit)

    # ─── Mesures (régression + classification dérivée) ────────
    def _compute_metrics(self, y_true_days, y_pred_days, threshold=10.0):
        import math
        r2   = float(r2_score(y_true_days, y_pred_days))
        mae  = float(mean_absolute_error(y_true_days, y_pred_days))
        rmse = float(math.sqrt(((y_true_days - y_pred_days) ** 2).mean()))
        mask = y_true_days > 0
        if mask.any():
            mape = float((np.abs((y_true_days[mask] - y_pred_days[mask])
                                  / y_true_days[mask])).mean() * 100)
        else:
            mape = 0.0
        # Classification dérivée (alert/sain selon seuil en jours)
        y_true_alert = (y_true_days <= threshold).astype(int)
        y_pred_alert = (y_pred_days <= threshold).astype(int)
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )
        accuracy  = float(accuracy_score(y_true_alert, y_pred_alert))
        precision = float(precision_score(y_true_alert, y_pred_alert,
                                          zero_division=0))
        recall    = float(recall_score(y_true_alert, y_pred_alert,
                                       zero_division=0))
        f1        = float(f1_score(y_true_alert, y_pred_alert,
                                   zero_division=0))
        return dict(r2=r2, mae=mae, rmse=rmse, mape=mape,
                    accuracy=accuracy, precision=precision,
                    recall=recall, f1=f1, threshold=threshold)

    # ─── ENTRAÎNEMENT MANUEL ──────────────────────────────────
    async def train_manual(
        self,
        architecture: str,
        embedding_dim: int,
        num_layers: int,
        units,
        dropout_rates,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        patience: int = 10,
    ) -> dict:
        await self._send({"type": "log",
                          "message": f"Entrainement Cevital {architecture} (mode manuel)"})
        start = time.time()

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
                          "message": f"  Parametres : {model.count_params():,}"})

        loop    = asyncio.get_event_loop()
        cb_ws   = WebSocketCallback(loop, self._send_fn, epochs)
        cb_es   = EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True)
        cb_rlr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=4, min_lr=1e-6)

        history = await self._fit_in_thread(
            model,
            x_inputs       = [self.pipeline.X_train_num, self.pipeline.X_train_comp],
            y              = self.pipeline.y_train,
            sample_weight  = self.pipeline.w_train,
            val_data       = ([self.pipeline.X_val_num, self.pipeline.X_val_comp],
                              self.pipeline.y_val),
            epochs         = epochs,
            batch_size     = batch_size,
            callbacks      = [cb_ws, cb_es, cb_rlr],
        )

        duration = time.time() - start

        y_pred_days = self.pipeline.predict_with_safety(
            model, self.pipeline.X_test_num, self.pipeline.X_test_comp,
        )
        y_true_days = self.pipeline.scaler_y.inverse_transform(
            self.pipeline.y_test.reshape(-1, 1)
        ).flatten()

        metrics = self._compute_metrics(y_true_days, y_pred_days)

        hist = history.history
        training_history = [
            {
                "epoch":    i + 1,
                "loss":     round(float(hist["loss"][i]), 6),
                "val_loss": round(float(hist["val_loss"][i]), 6),
                "mae":      round(float(hist.get("mae", [0])[i] if "mae" in hist else 0), 6),
                "val_mae":  round(float(hist.get("val_mae", [0])[i] if "val_mae" in hist else 0), 6),
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
            "message":  f"\nR2={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}j | RMSE={metrics['rmse']:.2f}j | MAPE={metrics['mape']:.2f}%",
        })

        return {
            "model":            model,
            "metrics":          metrics,
            "duration_sec":     duration,
            "training_history": training_history,
            "y_true":           y_true_days.tolist(),
            "y_pred":           y_pred_days.tolist(),
        }

    # ─── ENTRAÎNEMENT AUTOML (gp_minimize + TimeSeriesSplit) ────────
    async def train_auto(
        self,
        architecture: str,
        max_trials: int = 20,
        epochs: int = 20,             # EPOCHS_CV — époques par fold CV (notebook : 20)
        batch_size: int = 32,
        patience: int = 4,            # patience EarlyStopping CV (notebook : 4)
        embedding_search=(4, 8, 16, 32),
        units_min: int = 32,
        units_max: int = 128,
        units_step: int = 32,         # ignoré — gp_minimize optimise en continu
        nb_layers_min: int = 1,
        nb_layers_max: int = 1,
        dropout_min: float = 0.1,
        dropout_max: float = 0.4,
        lr_choices=(1e-4, 1e-2),
        final_epochs: int = 35,       # EPOCHS_FIN (notebook exact)
    ) -> dict:
        await self._send({"type": "log",
                          "message": f"AutoML gp_minimize {architecture} — {max_trials} essais × CV 3 folds ({epochs} ep/fold), puis {final_epochs} ep (entraînement final)"})
        await self._send({"type": "log",
                          "message": f"   Espace : lstm_units ∈ [{units_min},{units_max}] · dropout ∈ [{dropout_min},{dropout_max}] · lr ∈ [{min(lr_choices):.1e},{max(lr_choices):.1e}] (log-uniform)"})
        start = time.time()
        loop  = asyncio.get_event_loop()

        # Paramètres fixes (non recherchés — notebook : 1 couche LSTM, embed médian)
        embedding_dim = list(embedding_search)[len(embedding_search) // 2]
        num_layers    = nb_layers_min

        # ── Espace de recherche — notebook exact ─────────────────────
        dimensions = [
            Integer(units_min, units_max, name="lstm_units"),
            Real(dropout_min,  dropout_max, name="dropout_rate"),
            Real(min(lr_choices), max(lr_choices), "log-uniform", name="learning_rate"),
        ]

        tscv        = TimeSeriesSplit(n_splits=3)
        trial_state = {"n": 0, "best": None}

        # Copies locales pour la closure (accès thread-safe aux tableaux numpy)
        X_train_num  = self.pipeline.X_train_num
        X_train_comp = self.pipeline.X_train_comp
        y_train      = self.pipeline.y_train
        w_train      = self.pipeline.w_train
        scaler_y     = self.pipeline.scaler_y
        max_rul      = self.pipeline.current_max_rul

        # ── Fonction objectif — CV walk-forward (notebook exact) ─────
        @use_named_args(dimensions)
        def objective(lstm_units, dropout_rate, learning_rate):
            trial_state["n"] += 1
            n = trial_state["n"]

            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_fn({"type": "trial_start", "trial": n, "total": max_trials}),
                    loop,
                )
            except Exception:
                pass

            t_trial   = time.time()
            fold_maes = []

            for _, (tr_idx, val_idx) in enumerate(tscv.split(X_train_num)):
                tf.keras.backend.clear_session()
                gc.collect()

                model_fold = build_model_cevital_manual(
                    architecture     = architecture,
                    lookback         = self.lookback,
                    n_features       = self.n_features,
                    num_classes_comp = self.num_classes_comp,
                    embedding_dim    = int(embedding_dim),
                    num_layers       = num_layers,
                    units            = [int(lstm_units)] * num_layers,
                    dropout_rates    = [float(dropout_rate)] * num_layers,
                    learning_rate    = float(learning_rate),
                )

                model_fold.fit(
                    [X_train_num[tr_idx], X_train_comp[tr_idx]],
                    y_train[tr_idx],
                    sample_weight   = w_train[tr_idx],
                    validation_data = (
                        [X_train_num[val_idx], X_train_comp[val_idx]],
                        y_train[val_idx],
                    ),
                    epochs     = epochs,
                    batch_size = batch_size,
                    verbose    = 0,
                    callbacks  = [
                        EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
                    ],
                )

                preds   = model_fold.predict(
                    [X_train_num[val_idx], X_train_comp[val_idx]], verbose=0
                ).flatten()
                preds_d = np.clip(
                    scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten(), 0, max_rul
                )
                true_d  = scaler_y.inverse_transform(
                    y_train[val_idx].reshape(-1, 1)
                ).flatten()
                fold_maes.append(mean_absolute_error(true_d, preds_d))

                del model_fold
                tf.keras.backend.clear_session()
                gc.collect()

            cv_mae = float(np.mean(fold_maes))

            if trial_state["best"] is None or cv_mae < trial_state["best"]:
                trial_state["best"] = cv_mae

            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_fn({
                        "type":        "trial_end",
                        "trial":       n,
                        "avg_cv_loss": round(cv_mae, 4),
                        "duration":    round(time.time() - t_trial, 1),
                        "best_so_far": round(trial_state["best"], 4),
                    }),
                    loop,
                )
            except Exception:
                pass

            return cv_mae

        # ── gp_minimize dans un thread (ne bloque pas l'event loop) ──
        def _blocking_gp():
            return gp_minimize(
                objective,
                dimensions,
                n_calls=max_trials,
                random_state=42,
                n_jobs=1,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            res_gp = await loop.run_in_executor(executor, _blocking_gp)

        best = {
            "lstm_units":    int(res_gp.x[0]),
            "dropout_rate":  float(res_gp.x[1]),
            "learning_rate": float(res_gp.x[2]),
        }
        await self._send({"type": "log",
                          "message": f"Recherche terminée. Meilleurs params : {best}"})

        # ── Entraînement final — notebook exact (EPOCHS_FIN=35, batch=64) ──
        await self._send({"type": "log",
                          "message": f"Entraînement final — {final_epochs} époques (EarlyStopping patience=6 + ReduceLROnPlateau factor=0.5)"})

        tf.keras.backend.clear_session()
        gc.collect()

        best_model = build_model_cevital_manual(
            architecture     = architecture,
            lookback         = self.lookback,
            n_features       = self.n_features,
            num_classes_comp = self.num_classes_comp,
            embedding_dim    = int(embedding_dim),
            num_layers       = num_layers,
            units            = [best["lstm_units"]] * num_layers,
            dropout_rates    = [best["dropout_rate"]] * num_layers,
            learning_rate    = best["learning_rate"],
        )

        cb_ws_final  = WebSocketCallback(loop, self._send_fn, final_epochs)
        cb_es_final  = EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True)
        cb_rlr_final = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-6)

        history_final = await self._fit_in_thread(
            best_model,
            x_inputs      = [self.pipeline.X_train_num, self.pipeline.X_train_comp],
            y             = self.pipeline.y_train,
            sample_weight = self.pipeline.w_train,
            val_data      = ([self.pipeline.X_val_num, self.pipeline.X_val_comp],
                              self.pipeline.y_val),
            epochs        = final_epochs,
            batch_size    = 64,
            callbacks     = [cb_ws_final, cb_es_final, cb_rlr_final],
        )

        hist = history_final.history
        training_history = [
            {
                "epoch":    i + 1,
                "loss":     round(float(hist["loss"][i]), 6),
                "val_loss": round(float(hist["val_loss"][i]), 6),
                "mae":      round(float(hist.get("mae", [0])[i] if "mae" in hist else 0), 6),
                "val_mae":  round(float(hist.get("val_mae", [0])[i] if "val_mae" in hist else 0), 6),
            }
            for i in range(len(hist["loss"]))
        ]
        duration = time.time() - start

        y_pred_days = self.pipeline.predict_with_safety(
            best_model, self.pipeline.X_test_num, self.pipeline.X_test_comp,
        )
        y_true_days = self.pipeline.scaler_y.inverse_transform(
            self.pipeline.y_test.reshape(-1, 1)
        ).flatten()

        metrics = self._compute_metrics(y_true_days, y_pred_days)

        await self._send({
            "type":     "completed",
            "r2":       round(metrics["r2"], 4),
            "mae":      round(metrics["mae"], 3),
            "rmse":     round(metrics["rmse"], 3),
            "mape":     round(metrics["mape"], 2),
            "duration": round(duration, 1),
            "message":  f"\nAutoML terminé | R2={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}j",
        })

        return {
            "model":            best_model,
            "best_hps":         best,
            "metrics":          metrics,
            "duration_sec":     duration,
            "total_trials":     max_trials,
            "training_history": training_history,
            "y_true":           y_true_days.tolist(),
            "y_pred":           y_pred_days.tolist(),
        }
