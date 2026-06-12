"""
lstm_demo.py — Démo LSTM MULTI-COUCHES — pédagogique (mini-dataset synthétique).

Implémente les 4 portes classiques :
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate cell state
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # New cell state (mémoire long terme)
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
  h_t = o_t ⊙ tanh(C_t)                  # Hidden state (mémoire court terme)

Pour chaque porte, on stocke les matrices intermédiaires et les contributions
pour permettre une visualisation pédagogique pas-à-pas.
"""
import os
import numpy as np
from typing import Dict, List, Optional


def _round_tensor(arr, decimals=3):
    return np.round(arr, decimals).tolist()


def _sigmoid(x):
    """Sigmoïde numériquement stable."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ─────────────────────────────────────────────────────────────
# Construction du batch (identique à rnn_demo)
# ─────────────────────────────────────────────────────────────
def build_demo_batch(pipeline, batch_size=4, seq_length=3, n_features_show=4):
    if pipeline is None or pipeline.X_train is None:
        raise ValueError("Mini-dataset démo indisponible (pipeline synthétique non initialisé).")

    target_features = ["volt", "rotate", "pressure", "vibration"]
    feat_cols = pipeline.feature_cols
    feat_indices, feat_names = [], []
    for f in target_features:
        if f in feat_cols:
            feat_indices.append(feat_cols.index(f))
            feat_names.append(f)
    while len(feat_indices) < n_features_show:
        for i, name in enumerate(feat_cols):
            if i not in feat_indices:
                feat_indices.append(i)
                feat_names.append(name)
                break

    feat_indices = feat_indices[:n_features_show]
    feat_names   = feat_names[:n_features_show]

    X_full = pipeline.X_train[:batch_size]
    X_batch = X_full[:, :seq_length, :][:, :, feat_indices]
    y_batch_norm = pipeline.y_train[:batch_size].flatten()
    y_batch_hours = pipeline.scaler_y.inverse_transform(
        y_batch_norm.reshape(-1, 1)
    ).flatten()

    return {
        "X_batch":       X_batch.astype(np.float32),
        "y_batch_norm":  y_batch_norm.astype(np.float32),
        "y_batch_hours": y_batch_hours.astype(np.float32),
        "feat_names":    feat_names,
        "batch_size":    batch_size,
        "seq_length":    seq_length,
        "n_features":    n_features_show,
    }


# ─────────────────────────────────────────────────────────────
# Initialisation des poids LSTM multi-couches
# ─────────────────────────────────────────────────────────────
def init_lstm_weights_multilayer(n_features, layers, seed=7):
    """
    Pour chaque couche L :
      W_f, W_i, W_C, W_o de forme (input_dim_L + units_L, units_L)
      b_f, b_i, b_C, b_o de forme (units_L,)
    Pour faciliter la visu, on les sépare en (W_xf, W_hf), etc.
    """
    rng = np.random.RandomState(seed)
    layer_weights = []
    prev_dim = n_features
    for units in layers:
        scale_x = np.sqrt(1.0 / prev_dim)
        scale_h = np.sqrt(1.0 / units)
        layer_weights.append({
            "input_dim": prev_dim,
            "units":     units,
            # Forget gate
            "W_xf": (rng.randn(prev_dim, units) * scale_x).astype(np.float32),
            "W_hf": (rng.randn(units,   units) * scale_h).astype(np.float32),
            "b_f":  np.ones(units, dtype=np.float32),  # biais forget initialisé à 1 (best practice)
            # Input gate
            "W_xi": (rng.randn(prev_dim, units) * scale_x).astype(np.float32),
            "W_hi": (rng.randn(units,   units) * scale_h).astype(np.float32),
            "b_i":  np.zeros(units, dtype=np.float32),
            # Candidate cell state
            "W_xC": (rng.randn(prev_dim, units) * scale_x).astype(np.float32),
            "W_hC": (rng.randn(units,   units) * scale_h).astype(np.float32),
            "b_C":  np.zeros(units, dtype=np.float32),
            # Output gate
            "W_xo": (rng.randn(prev_dim, units) * scale_x).astype(np.float32),
            "W_ho": (rng.randn(units,   units) * scale_h).astype(np.float32),
            "b_o":  np.zeros(units, dtype=np.float32),
        })
        prev_dim = units

    return {
        "layers": layer_weights,
        "W_out": (rng.randn(layers[-1], 1) * 0.3).astype(np.float32),
        "b_out": np.zeros(1, dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────
# Forward pass MULTI-COUCHES
# ─────────────────────────────────────────────────────────────
def forward_lstm_multilayer(X_batch, weights):
    """
    Pour chaque couche L et chaque temps t, calcule les 4 portes + cell state + hidden state.
    """
    B, T, F = X_batch.shape
    n_layers = len(weights["layers"])

    # États stockés par couche : layer → list[T] de matrices (B, units_L)
    activations_h = [[None] * T for _ in range(n_layers)]
    cell_states_C = [[None] * T for _ in range(n_layers)]
    gates = [[None] * T for _ in range(n_layers)]
    pre_gates = [[None] * T for _ in range(n_layers)]
    contributions = [[None] * T for _ in range(n_layers)]

    h_prev = [np.zeros((B, layer["units"]), dtype=np.float32)
              for layer in weights["layers"]]
    C_prev = [np.zeros((B, layer["units"]), dtype=np.float32)
              for layer in weights["layers"]]

    for t in range(T):
        input_t = X_batch[:, t, :]
        for L, layer in enumerate(weights["layers"]):
            # Pré-activations des 4 portes
            z_f = input_t @ layer["W_xf"] + h_prev[L] @ layer["W_hf"] + layer["b_f"]
            z_i = input_t @ layer["W_xi"] + h_prev[L] @ layer["W_hi"] + layer["b_i"]
            z_C = input_t @ layer["W_xC"] + h_prev[L] @ layer["W_hC"] + layer["b_C"]
            z_o = input_t @ layer["W_xo"] + h_prev[L] @ layer["W_ho"] + layer["b_o"]

            # Activations des portes
            f_t = _sigmoid(z_f)              # forget gate
            i_t = _sigmoid(z_i)              # input gate
            C_tilde = np.tanh(z_C)           # candidate cell state
            o_t = _sigmoid(z_o)              # output gate

            # Mise à jour cell state
            C_t = f_t * C_prev[L] + i_t * C_tilde

            # Hidden state
            h_t = o_t * np.tanh(C_t)

            # Stockage
            pre_gates[L][t] = {
                "z_f": z_f, "z_i": z_i, "z_C": z_C, "z_o": z_o
            }
            gates[L][t] = {
                "f_t": f_t, "i_t": i_t, "C_tilde": C_tilde, "o_t": o_t,
                "C_t": C_t,
            }
            activations_h[L][t]  = h_t
            cell_states_C[L][t]  = C_t
            contributions[L][t] = {
                "input_t": input_t,
                "h_prev":  h_prev[L],
                "C_prev":  C_prev[L],
            }

            h_prev[L] = h_t
            C_prev[L] = C_t
            input_t = h_t  # entrée pour la couche suivante

    H_last = activations_h[-1][T - 1]
    Y_pred = H_last @ weights["W_out"] + weights["b_out"]

    return {
        "activations_h": activations_h,
        "cell_states_C": cell_states_C,
        "gates":         gates,
        "pre_gates":     pre_gates,
        "contributions": contributions,
        "Y_pred":        Y_pred,
        "H_last":        H_last,
        "n_layers":      n_layers,
    }


# ─────────────────────────────────────────────────────────────
# Loss (MSE)
# ─────────────────────────────────────────────────────────────
def compute_loss(forward, y_true, batch_size):
    y_true = y_true.reshape(-1, 1)
    y_pred = forward["Y_pred"]
    per_seq_error = (y_pred - y_true)
    per_seq_loss  = per_seq_error ** 2
    loss = float(per_seq_loss.mean())
    return {
        "loss": loss,
        "per_seq_error": per_seq_error,
        "per_seq_loss": per_seq_loss,
    }


# ─────────────────────────────────────────────────────────────
# Backward pass (BPTT) MULTI-COUCHES
# ─────────────────────────────────────────────────────────────
def backward_lstm_multilayer(X_batch, weights, forward, y_true):
    B, T, F = X_batch.shape
    n_layers = forward["n_layers"]
    y_true = y_true.reshape(-1, 1)
    y_pred = forward["Y_pred"]

    # Loss
    per_seq_error = (y_pred - y_true)
    loss_val = float((per_seq_error ** 2).mean())

    # Gradient à la sortie
    dY = (2.0 / B) * per_seq_error
    dW_out = forward["H_last"].T @ dY
    db_out = dY.sum(axis=0)
    dH_top = dY @ weights["W_out"].T

    # Gradients par couche
    layer_grads = [{
        "dW_xf": np.zeros_like(L["W_xf"]), "dW_hf": np.zeros_like(L["W_hf"]), "db_f": np.zeros_like(L["b_f"]),
        "dW_xi": np.zeros_like(L["W_xi"]), "dW_hi": np.zeros_like(L["W_hi"]), "db_i": np.zeros_like(L["b_i"]),
        "dW_xC": np.zeros_like(L["W_xC"]), "dW_hC": np.zeros_like(L["W_hC"]), "db_C": np.zeros_like(L["b_C"]),
        "dW_xo": np.zeros_like(L["W_xo"]), "dW_ho": np.zeros_like(L["W_ho"]), "db_o": np.zeros_like(L["b_o"]),
    } for L in weights["layers"]]

    # dH entrant à chaque temps pour chaque couche
    dH_per_layer = [
        [np.zeros((B, L["units"]), dtype=np.float32) for _ in range(T)]
        for L in weights["layers"]
    ]
    dH_per_layer[-1][T - 1] = dH_top.copy()

    # On parcourt les couches du HAUT vers le BAS
    for L_idx in reversed(range(n_layers)):
        layer = weights["layers"][L_idx]
        units = layer["units"]
        dh_future = np.zeros((B, units), dtype=np.float32)
        dC_future = np.zeros((B, units), dtype=np.float32)
        dInput_per_t = [None] * T

        for t in reversed(range(T)):
            dh_t = dH_per_layer[L_idx][t] + dh_future
            gates_t = forward["gates"][L_idx][t]
            f_t = gates_t["f_t"]; i_t = gates_t["i_t"]; C_tilde = gates_t["C_tilde"]
            o_t = gates_t["o_t"]; C_t = gates_t["C_t"]

            X_t   = forward["contributions"][L_idx][t]["input_t"]
            H_prv = forward["contributions"][L_idx][t]["h_prev"]
            C_prv = forward["contributions"][L_idx][t]["C_prev"]

            tanh_C = np.tanh(C_t)
            # dh/dC = o_t * (1 - tanh(C)^2)
            dC_t = dh_t * o_t * (1 - tanh_C ** 2) + dC_future

            # Gradients par porte (avant sigmoïde/tanh)
            do_t  = dh_t * tanh_C * o_t * (1 - o_t)
            df_t  = dC_t * C_prv * f_t * (1 - f_t)
            di_t  = dC_t * C_tilde * i_t * (1 - i_t)
            dC_tilde = dC_t * i_t * (1 - C_tilde ** 2)

            # Gradients sur les poids
            layer_grads[L_idx]["dW_xf"] += X_t.T   @ df_t
            layer_grads[L_idx]["dW_hf"] += H_prv.T @ df_t
            layer_grads[L_idx]["db_f"]  += df_t.sum(axis=0)

            layer_grads[L_idx]["dW_xi"] += X_t.T   @ di_t
            layer_grads[L_idx]["dW_hi"] += H_prv.T @ di_t
            layer_grads[L_idx]["db_i"]  += di_t.sum(axis=0)

            layer_grads[L_idx]["dW_xC"] += X_t.T   @ dC_tilde
            layer_grads[L_idx]["dW_hC"] += H_prv.T @ dC_tilde
            layer_grads[L_idx]["db_C"]  += dC_tilde.sum(axis=0)

            layer_grads[L_idx]["dW_xo"] += X_t.T   @ do_t
            layer_grads[L_idx]["dW_ho"] += H_prv.T @ do_t
            layer_grads[L_idx]["db_o"]  += do_t.sum(axis=0)

            # Gradient vers l'entrée de cette couche
            dInput_per_t[t] = (df_t @ layer["W_xf"].T
                             + di_t @ layer["W_xi"].T
                             + dC_tilde @ layer["W_xC"].T
                             + do_t @ layer["W_xo"].T)

            # Gradients vers le passé temporel
            dh_future = (df_t @ layer["W_hf"].T
                       + di_t @ layer["W_hi"].T
                       + dC_tilde @ layer["W_hC"].T
                       + do_t @ layer["W_ho"].T)
            dC_future = dC_t * f_t

        if L_idx > 0:
            for t in range(T):
                dH_per_layer[L_idx - 1][t] = dInput_per_t[t]

    return {
        "loss":          loss_val,
        "per_seq_error": per_seq_error,
        "per_seq_loss":  per_seq_error ** 2,
        "dY":            dY,
        "dW_out":        dW_out,
        "db_out":        db_out,
        "layer_grads":   layer_grads,
    }


# ─────────────────────────────────────────────────────────────
# API principale
# ─────────────────────────────────────────────────────────────
def run_demo(pipeline, layers=[4], batch_size=4, seq_length=3,
             learning_rate=0.1, seed=7):
    batch = build_demo_batch(pipeline, batch_size, seq_length, n_features_show=4)
    X = batch["X_batch"]
    y = batch["y_batch_norm"]
    B, T, F = X.shape

    weights = init_lstm_weights_multilayer(F, layers=layers, seed=seed)
    forward = forward_lstm_multilayer(X, weights)
    bwd = backward_lstm_multilayer(X, weights, forward, y)

    # SGD update
    updated_layers = []
    for L_idx, L in enumerate(weights["layers"]):
        g = bwd["layer_grads"][L_idx]
        updated_layers.append({
            "W_xf_new": L["W_xf"] - learning_rate * g["dW_xf"],
            "W_hf_new": L["W_hf"] - learning_rate * g["dW_hf"],
            "b_f_new":  L["b_f"]  - learning_rate * g["db_f"],
            "W_xi_new": L["W_xi"] - learning_rate * g["dW_xi"],
            "W_hi_new": L["W_hi"] - learning_rate * g["dW_hi"],
            "b_i_new":  L["b_i"]  - learning_rate * g["db_i"],
            "W_xC_new": L["W_xC"] - learning_rate * g["dW_xC"],
            "W_hC_new": L["W_hC"] - learning_rate * g["dW_hC"],
            "b_C_new":  L["b_C"]  - learning_rate * g["db_C"],
            "W_xo_new": L["W_xo"] - learning_rate * g["dW_xo"],
            "W_ho_new": L["W_ho"] - learning_rate * g["dW_ho"],
            "b_o_new":  L["b_o"]  - learning_rate * g["db_o"],
        })
    W_out_new = weights["W_out"] - learning_rate * bwd["dW_out"]

    y_pred_norm = forward["Y_pred"].flatten()
    y_pred_hours = pipeline.scaler_y.inverse_transform(
        y_pred_norm.reshape(-1, 1)
    ).flatten()

    def tl(arr, decimals=3):
        return _round_tensor(np.asarray(arr), decimals)

    return {
        "config": {
            "batch_size":    B,
            "seq_length":    T,
            "n_features":    F,
            "layers":        layers,
            "n_layers":      len(layers),
            "learning_rate": learning_rate,
            "machine_id":    getattr(pipeline, "machine_id", 99),
        },
        "input_tensor_3d": tl(X),
        "feat_names":      batch["feat_names"],
        "y_true_norm":     tl(y),
        "y_true_hours":    tl(batch["y_batch_hours"]),
        "slices_2d":       [tl(X[:, t, :]) for t in range(T)],

        # Poids initiaux par couche (12 matrices par couche !)
        "weights": {
            "layers": [{
                "L":         L_idx + 1,
                "input_dim": layer["input_dim"],
                "units":     layer["units"],
                "W_xf": tl(layer["W_xf"]), "W_hf": tl(layer["W_hf"]), "b_f": tl(layer["b_f"]),
                "W_xi": tl(layer["W_xi"]), "W_hi": tl(layer["W_hi"]), "b_i": tl(layer["b_i"]),
                "W_xC": tl(layer["W_xC"]), "W_hC": tl(layer["W_hC"]), "b_C": tl(layer["b_C"]),
                "W_xo": tl(layer["W_xo"]), "W_ho": tl(layer["W_ho"]), "b_o": tl(layer["b_o"]),
            } for L_idx, layer in enumerate(weights["layers"])],
            "W_out": tl(weights["W_out"]),
            "b_out": tl(weights["b_out"]),
        },

        # Forward steps : pour chaque (couche, temps)
        "forward_steps": [
            [
                {
                    "L":       L_idx + 1,
                    "t":       t + 1,
                    "input_t": tl(forward["contributions"][L_idx][t]["input_t"]),
                    "h_prev":  tl(forward["contributions"][L_idx][t]["h_prev"]),
                    "C_prev":  tl(forward["contributions"][L_idx][t]["C_prev"]),
                    # Pré-activations
                    "z_f": tl(forward["pre_gates"][L_idx][t]["z_f"]),
                    "z_i": tl(forward["pre_gates"][L_idx][t]["z_i"]),
                    "z_C": tl(forward["pre_gates"][L_idx][t]["z_C"]),
                    "z_o": tl(forward["pre_gates"][L_idx][t]["z_o"]),
                    # Activations des portes
                    "f_t":     tl(forward["gates"][L_idx][t]["f_t"]),
                    "i_t":     tl(forward["gates"][L_idx][t]["i_t"]),
                    "C_tilde": tl(forward["gates"][L_idx][t]["C_tilde"]),
                    "o_t":     tl(forward["gates"][L_idx][t]["o_t"]),
                    # États
                    "C_t":     tl(forward["gates"][L_idx][t]["C_t"]),
                    "H_t":     tl(forward["activations_h"][L_idx][t]),
                }
                for t in range(T)
            ]
            for L_idx in range(len(layers))
        ],

        "prediction": {
            "H_last":       tl(forward["H_last"]),
            "Y_pred_norm":  tl(forward["Y_pred"].flatten()),
            "Y_pred_hours": tl(y_pred_hours),
        },
        "loss_info": {
            "per_seq_error_norm": tl(bwd["per_seq_error"].flatten()),
            "per_seq_loss":       tl(bwd["per_seq_loss"].flatten()),
            "loss":               round(bwd["loss"], 6),
        },
        "backward": {
            "dY":          tl(bwd["dY"]),
            "dW_out":      tl(bwd["dW_out"]),
            "db_out":      tl(bwd["db_out"]),
            "layer_grads": [{
                "L":      L_idx + 1,
                "dW_xf":  tl(g["dW_xf"]), "dW_hf": tl(g["dW_hf"]), "db_f": tl(g["db_f"]),
                "dW_xi":  tl(g["dW_xi"]), "dW_hi": tl(g["dW_hi"]), "db_i": tl(g["db_i"]),
                "dW_xC":  tl(g["dW_xC"]), "dW_hC": tl(g["dW_hC"]), "db_C": tl(g["db_C"]),
                "dW_xo":  tl(g["dW_xo"]), "dW_ho": tl(g["dW_ho"]), "db_o": tl(g["db_o"]),
                "norm_total": float(np.linalg.norm(np.concatenate([
                    g["dW_xf"].flatten(), g["dW_hf"].flatten(),
                    g["dW_xi"].flatten(), g["dW_hi"].flatten(),
                    g["dW_xC"].flatten(), g["dW_hC"].flatten(),
                    g["dW_xo"].flatten(), g["dW_ho"].flatten(),
                ]))),
            } for L_idx, g in enumerate(bwd["layer_grads"])],
        },
        "updated": {
            "layers": [{
                "L":        L_idx + 1,
                **{k: tl(v) for k, v in U.items()}
            } for L_idx, U in enumerate(updated_layers)],
            "W_out_new": tl(W_out_new),
        },
    }