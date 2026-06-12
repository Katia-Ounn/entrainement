"""
rnn_demo.py — Démo RNN MULTI-COUCHES (1, 2 ou 3 couches) — pédagogique.

Consomme un objet "pipeline-like" (synthétique en Partie 2) exposant
X_train, y_train, feature_cols, scaler_y. Voir backend/demos/synthetic_data.py.
"""
import os
import numpy as np
from typing import Dict, List, Optional


def _round_tensor(arr, decimals=3):
    return np.round(arr, decimals).tolist()


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


def init_demo_weights_multilayer(n_features, layers, seed=7):
    """layers: [4] ou [4, 4] ou [4, 4, 4]"""
    rng = np.random.RandomState(seed)
    layer_weights = []
    prev_dim = n_features
    for units in layers:
        scale_xh = np.sqrt(1.0 / prev_dim)
        scale_hh = np.sqrt(1.0 / units)
        layer_weights.append({
            "input_dim": prev_dim,
            "units":     units,
            "W_xh":  (rng.randn(prev_dim, units) * scale_xh).astype(np.float32),
            "W_hh":  (rng.randn(units,   units) * scale_hh).astype(np.float32),
            "b":     np.zeros(units, dtype=np.float32),
        })
        prev_dim = units

    return {
        "layers": layer_weights,
        "W_out": (rng.randn(layers[-1], 1) * 0.3).astype(np.float32),
        "b_out": np.zeros(1, dtype=np.float32),
    }


def forward_multilayer(X_batch, weights):
    B, T, F = X_batch.shape
    n_layers = len(weights["layers"])

    activations   = [[None] * T for _ in range(n_layers)]
    pre_acts      = [[None] * T for _ in range(n_layers)]
    contributions = [[None] * T for _ in range(n_layers)]

    h_prev = [np.zeros((B, layer["units"]), dtype=np.float32)
              for layer in weights["layers"]]

    for t in range(T):
        input_t = X_batch[:, t, :]
        for L, layer in enumerate(weights["layers"]):
            part_xh = input_t   @ layer["W_xh"]
            part_hh = h_prev[L] @ layer["W_hh"]
            part_b  = layer["b"]
            z_t = part_xh + part_hh + part_b
            h_t = np.tanh(z_t)

            pre_acts[L][t]      = z_t
            activations[L][t]   = h_t
            contributions[L][t] = {
                "input_t": input_t,
                "h_prev":  h_prev[L],
                "part_xh": part_xh,
                "part_hh": part_hh,
                "part_b":  np.tile(part_b, (B, 1)),
            }

            h_prev[L] = h_t
            input_t = h_t

    H_last = activations[-1][T - 1]
    Y_pred = H_last @ weights["W_out"] + weights["b_out"]

    return {
        "activations":   activations,
        "pre_acts":      pre_acts,
        "contributions": contributions,
        "Y_pred":        Y_pred,
        "H_last":        H_last,
        "n_layers":      n_layers,
    }


def compute_loss_and_backward_multilayer(X_batch, weights, forward, y_true):
    B, T, F = X_batch.shape
    n_layers = forward["n_layers"]
    y_true = y_true.reshape(-1, 1)
    y_pred = forward["Y_pred"]

    per_seq_error = (y_pred - y_true)
    per_seq_loss  = per_seq_error ** 2
    loss          = float(per_seq_loss.mean())

    dY = (2.0 / B) * per_seq_error
    dW_out = forward["H_last"].T @ dY
    db_out = dY.sum(axis=0)
    dH_top = dY @ weights["W_out"].T

    layer_grads = [{
        "dW_xh": np.zeros_like(L["W_xh"]),
        "dW_hh": np.zeros_like(L["W_hh"]),
        "db":    np.zeros_like(L["b"]),
    } for L in weights["layers"]]

    dH_per_layer = [
        [np.zeros((B, L["units"]), dtype=np.float32) for _ in range(T)]
        for L in weights["layers"]
    ]
    dH_per_layer[-1][T - 1] = dH_top.copy()

    for L_idx in reversed(range(n_layers)):
        layer = weights["layers"][L_idx]
        units = layer["units"]
        dh_future = np.zeros((B, units), dtype=np.float32)
        dInput_per_t = [None] * T

        for t in reversed(range(T)):
            dh_t = dH_per_layer[L_idx][t] + dh_future
            h_t  = forward["activations"][L_idx][t]
            dz_t = dh_t * (1.0 - h_t ** 2)

            X_t   = forward["contributions"][L_idx][t]["input_t"]
            H_prv = forward["contributions"][L_idx][t]["h_prev"]

            layer_grads[L_idx]["dW_xh"] += X_t.T   @ dz_t
            layer_grads[L_idx]["dW_hh"] += H_prv.T @ dz_t
            layer_grads[L_idx]["db"]    += dz_t.sum(axis=0)

            dInput_per_t[t] = dz_t @ layer["W_xh"].T
            dh_future = dz_t @ layer["W_hh"].T

        if L_idx > 0:
            for t in range(T):
                dH_per_layer[L_idx - 1][t] = dInput_per_t[t]

    return {
        "loss":          loss,
        "per_seq_error": per_seq_error,
        "per_seq_loss":  per_seq_loss,
        "dY":            dY,
        "dW_out":        dW_out,
        "db_out":        db_out,
        "layer_grads":   layer_grads,
        "dH_per_layer":  dH_per_layer,
    }


def run_demo(pipeline, layers=[4], batch_size=4, seq_length=3,
             learning_rate=0.1, seed=7):
    batch = build_demo_batch(pipeline, batch_size, seq_length, n_features_show=4)
    X = batch["X_batch"]
    y = batch["y_batch_norm"]
    B, T, F = X.shape

    weights = init_demo_weights_multilayer(F, layers=layers, seed=seed)
    forward = forward_multilayer(X, weights)
    bwd = compute_loss_and_backward_multilayer(X, weights, forward, y)

    updated_layers = []
    for L_idx, L in enumerate(weights["layers"]):
        g = bwd["layer_grads"][L_idx]
        updated_layers.append({
            "W_xh_new": L["W_xh"] - learning_rate * g["dW_xh"],
            "W_hh_new": L["W_hh"] - learning_rate * g["dW_hh"],
            "b_new":    L["b"]    - learning_rate * g["db"],
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
        "weights": {
            "layers": [{
                "L":         L_idx + 1,
                "input_dim": layer["input_dim"],
                "units":     layer["units"],
                "W_xh":      tl(layer["W_xh"]),
                "W_hh":      tl(layer["W_hh"]),
                "b":         tl(layer["b"]),
            } for L_idx, layer in enumerate(weights["layers"])],
            "W_out": tl(weights["W_out"]),
            "b_out": tl(weights["b_out"]),
        },
        "forward_steps": [
            [
                {
                    "L":       L_idx + 1,
                    "t":       t + 1,
                    "input_t": tl(forward["contributions"][L_idx][t]["input_t"]),
                    "h_prev":  tl(forward["contributions"][L_idx][t]["h_prev"]),
                    "part_xh": tl(forward["contributions"][L_idx][t]["part_xh"]),
                    "part_hh": tl(forward["contributions"][L_idx][t]["part_hh"]),
                    "part_b":  tl(forward["contributions"][L_idx][t]["part_b"]),
                    "Z_t":     tl(forward["pre_acts"][L_idx][t]),
                    "H_t":     tl(forward["activations"][L_idx][t]),
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
                "L":          L_idx + 1,
                "dW_xh":      tl(g["dW_xh"]),
                "dW_hh":      tl(g["dW_hh"]),
                "db":         tl(g["db"]),
                "norm_dW_xh": float(np.linalg.norm(g["dW_xh"])),
                "norm_dW_hh": float(np.linalg.norm(g["dW_hh"])),
                "norm_db":    float(np.linalg.norm(g["db"])),
            } for L_idx, g in enumerate(bwd["layer_grads"])],
            "dH_per_layer": [
                [tl(dh) for dh in layer_dHs]
                for layer_dHs in bwd["dH_per_layer"]
            ],
        },
        "updated": {
            "layers": [{
                "L":        L_idx + 1,
                "W_xh_new": tl(L_new["W_xh_new"]),
                "W_hh_new": tl(L_new["W_hh_new"]),
                "b_new":    tl(L_new["b_new"]),
            } for L_idx, L_new in enumerate(updated_layers)],
            "W_out_new": tl(W_out_new),
        },
    }