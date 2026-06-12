"""
transformer_demo.py — Démo Transformer ENCODER + DECODER — pédagogique (mini-dataset synthétique).

Architecture pédagogique simplifiée d'un Transformer :

  ENCODER (traite la séquence d'entrée X) :
    1. Positional Encoding : X + PE
    2. Multi-Head Attention (Self-attention sur l'entrée)
    3. Add & LayerNorm (résiduelle + normalisation)
    4. Feed-Forward Network (FFN)
    5. Add & LayerNorm

  DECODER (génère la prédiction) :
    1. Masked Multi-Head Attention (auto-attention)
    2. Add & LayerNorm
    3. Encoder-Decoder Attention (cross-attention sur la sortie encoder)
    4. Add & LayerNorm
    5. Feed-Forward Network
    6. Add & LayerNorm

  HEAD final : Dense(1) pour prédire la RUL.

Toutes les matrices Q, K, V, scores d'attention, softmax, etc. sont sauvegardées
pour permettre une visualisation pas-à-pas dans le frontend.
"""
import os
import numpy as np
from typing import Dict, List, Optional


def _round_tensor(arr, decimals=3):
    return np.round(arr, decimals).tolist()


def _softmax(x, axis=-1):
    """Softmax numériquement stable."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _layer_norm(x, eps=1e-5):
    """Layer Normalization sur la dernière dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ─────────────────────────────────────────────────────────────
# Construction du batch (réutilisé)
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
# Positional Encoding (sinusoidal — Vaswani et al.)
# ─────────────────────────────────────────────────────────────
def positional_encoding(seq_length, d_model):
    """
    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    """
    pe = np.zeros((seq_length, d_model), dtype=np.float32)
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# ─────────────────────────────────────────────────────────────
# Initialisation des poids Transformer
# ─────────────────────────────────────────────────────────────
def init_transformer_weights(n_features, d_model=8, n_heads=2, d_ff=16, seed=7):
    """
    d_model doit être divisible par n_heads.
    Une seule couche encoder + une seule couche decoder pour la démo.
    """
    if d_model % n_heads != 0:
        # Ajuster automatiquement
        d_model = n_heads * (d_model // n_heads + (1 if d_model % n_heads else 0))

    rng = np.random.RandomState(seed)
    scale_in = np.sqrt(1.0 / n_features)
    scale_d  = np.sqrt(1.0 / d_model)

    return {
        "config": {
            "n_features": n_features,
            "d_model":    d_model,
            "n_heads":    n_heads,
            "d_head":     d_model // n_heads,
            "d_ff":       d_ff,
        },
        # Embedding initial : projection des features → d_model
        "W_embed": (rng.randn(n_features, d_model) * scale_in).astype(np.float32),
        "b_embed": np.zeros(d_model, dtype=np.float32),

        # ENCODER LAYER
        "encoder": {
            # Multi-head self-attention
            "W_q": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_k": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_v": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_o": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            # FFN
            "W_ff1": (rng.randn(d_model, d_ff) * scale_d).astype(np.float32),
            "b_ff1": np.zeros(d_ff, dtype=np.float32),
            "W_ff2": (rng.randn(d_ff, d_model) * np.sqrt(1.0 / d_ff)).astype(np.float32),
            "b_ff2": np.zeros(d_model, dtype=np.float32),
        },

        # DECODER LAYER (avec une cible apprise simple = vecteur de query)
        "decoder": {
            # Masked self-attention (sur la cible)
            "W_q1": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_k1": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_v1": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_o1": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            # Cross-attention (Q from decoder, K & V from encoder)
            "W_q2": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_k2": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_v2": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            "W_o2": (rng.randn(d_model, d_model) * scale_d).astype(np.float32),
            # FFN
            "W_ff1": (rng.randn(d_model, d_ff) * scale_d).astype(np.float32),
            "b_ff1": np.zeros(d_ff, dtype=np.float32),
            "W_ff2": (rng.randn(d_ff, d_model) * np.sqrt(1.0 / d_ff)).astype(np.float32),
            "b_ff2": np.zeros(d_model, dtype=np.float32),
            # Token cible appris (vecteur unique)
            "target_token": (rng.randn(d_model) * 0.1).astype(np.float32),
        },

        # Tête finale de prédiction
        "W_out": (rng.randn(d_model, 1) * 0.3).astype(np.float32),
        "b_out": np.zeros(1, dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────
# Multi-head attention — détail complet
# ─────────────────────────────────────────────────────────────
def multi_head_attention(Q_input, K_input, V_input, W_q, W_k, W_v, W_o,
                          n_heads, d_head, mask=None):
    """
    Q_input : (B, T_q, d_model)
    K_input : (B, T_k, d_model)
    V_input : (B, T_k, d_model)

    Retourne :
      output    : (B, T_q, d_model)
      heads_data: list des données par tête (pour visualisation)
    """
    B = Q_input.shape[0]
    T_q = Q_input.shape[1]
    T_k = K_input.shape[1]
    d_model = W_q.shape[1]

    # Projections
    Q = Q_input @ W_q   # (B, T_q, d_model)
    K = K_input @ W_k   # (B, T_k, d_model)
    V = V_input @ W_v   # (B, T_k, d_model)

    # Découpage en têtes : (B, n_heads, T, d_head)
    def split_heads(x):
        # x: (B, T, d_model) → (B, n_heads, T, d_head)
        T = x.shape[1]
        x = x.reshape(B, T, n_heads, d_head)
        return x.transpose(0, 2, 1, 3)

    Q_h = split_heads(Q)  # (B, n_heads, T_q, d_head)
    K_h = split_heads(K)
    V_h = split_heads(V)

    # Scores : Q · K^T / sqrt(d_head)
    scores = np.matmul(Q_h, K_h.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    # scores: (B, n_heads, T_q, T_k)

    # Mask (pour le decoder masked attention)
    if mask is not None:
        # mask: (T_q, T_k) avec -inf où on doit masquer
        scores = scores + mask[np.newaxis, np.newaxis, :, :]

    # Softmax
    attn_weights = _softmax(scores, axis=-1)  # (B, n_heads, T_q, T_k)

    # Application aux V
    head_outputs = np.matmul(attn_weights, V_h)  # (B, n_heads, T_q, d_head)

    # Concaténation des têtes
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(B, T_q, d_model)

    # Projection finale
    output = concat @ W_o  # (B, T_q, d_model)

    # Données par tête pour visualisation
    heads_data = []
    for h in range(n_heads):
        heads_data.append({
            "Q":            Q_h[:, h, :, :].copy(),    # (B, T_q, d_head)
            "K":            K_h[:, h, :, :].copy(),    # (B, T_k, d_head)
            "V":            V_h[:, h, :, :].copy(),    # (B, T_k, d_head)
            "scores":       scores[:, h, :, :].copy(),       # (B, T_q, T_k)
            "attn_weights": attn_weights[:, h, :, :].copy(), # (B, T_q, T_k)
            "head_output":  head_outputs[:, h, :, :].copy(), # (B, T_q, d_head)
        })

    return output, {
        "Q_full": Q,
        "K_full": K,
        "V_full": V,
        "concat": concat,
        "output": output,
        "heads":  heads_data,
    }


# ─────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────
def feed_forward(x, W_ff1, b_ff1, W_ff2, b_ff2):
    """FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2"""
    hidden = np.maximum(0, x @ W_ff1 + b_ff1)  # ReLU
    output = hidden @ W_ff2 + b_ff2
    return output, hidden


# ─────────────────────────────────────────────────────────────
# Forward pass complet
# ─────────────────────────────────────────────────────────────
def forward_transformer(X_batch, weights):
    """
    X_batch : (B, T, n_features)
    Retourne tous les tenseurs intermédiaires pour la visualisation.
    """
    cfg = weights["config"]
    B, T, F = X_batch.shape
    d_model, n_heads, d_head, d_ff = cfg["d_model"], cfg["n_heads"], cfg["d_head"], cfg["d_ff"]

    # ═══ 1. EMBEDDING + POSITIONAL ENCODING ═══
    X_embed = X_batch @ weights["W_embed"] + weights["b_embed"]   # (B, T, d_model)
    PE = positional_encoding(T, d_model)                          # (T, d_model)
    X_pos = X_embed + PE[np.newaxis, :, :]                        # (B, T, d_model)

    # ═══ 2. ENCODER ═══
    enc_w = weights["encoder"]

    # Multi-head self-attention
    attn_out, attn_data = multi_head_attention(
        X_pos, X_pos, X_pos,
        enc_w["W_q"], enc_w["W_k"], enc_w["W_v"], enc_w["W_o"],
        n_heads, d_head
    )
    # Add & Norm
    enc_residual1 = X_pos + attn_out
    enc_norm1 = _layer_norm(enc_residual1)

    # FFN
    ffn_out, ffn_hidden = feed_forward(
        enc_norm1, enc_w["W_ff1"], enc_w["b_ff1"], enc_w["W_ff2"], enc_w["b_ff2"]
    )
    # Add & Norm
    enc_residual2 = enc_norm1 + ffn_out
    encoder_output = _layer_norm(enc_residual2)   # (B, T, d_model)

    # ═══ 3. DECODER ═══
    dec_w = weights["decoder"]
    # Cible : un seul "token" appris (utilisé pour la régression)
    # On le répète pour le batch : (B, 1, d_model)
    target = np.tile(dec_w["target_token"][np.newaxis, np.newaxis, :], (B, 1, 1))

    # 3.1 Masked self-attention (sur le seul token cible — donc pas vraiment besoin de masque ici)
    masked_out, masked_data = multi_head_attention(
        target, target, target,
        dec_w["W_q1"], dec_w["W_k1"], dec_w["W_v1"], dec_w["W_o1"],
        n_heads, d_head
    )
    dec_residual1 = target + masked_out
    dec_norm1 = _layer_norm(dec_residual1)

    # 3.2 Cross-attention : Q from decoder, K & V from encoder
    cross_out, cross_data = multi_head_attention(
        dec_norm1, encoder_output, encoder_output,
        dec_w["W_q2"], dec_w["W_k2"], dec_w["W_v2"], dec_w["W_o2"],
        n_heads, d_head
    )
    dec_residual2 = dec_norm1 + cross_out
    dec_norm2 = _layer_norm(dec_residual2)

    # 3.3 FFN
    dec_ffn_out, dec_ffn_hidden = feed_forward(
        dec_norm2, dec_w["W_ff1"], dec_w["b_ff1"], dec_w["W_ff2"], dec_w["b_ff2"]
    )
    dec_residual3 = dec_norm2 + dec_ffn_out
    decoder_output = _layer_norm(dec_residual3)   # (B, 1, d_model)

    # ═══ 4. HEAD DE PRÉDICTION ═══
    decoder_flat = decoder_output[:, 0, :]   # (B, d_model)
    Y_pred = decoder_flat @ weights["W_out"] + weights["b_out"]   # (B, 1)

    return {
        # Embedding
        "X_embed":  X_embed,
        "PE":       PE,
        "X_pos":    X_pos,

        # Encoder
        "encoder": {
            "self_attn":        attn_data,
            "attn_out":         attn_out,
            "residual1":        enc_residual1,
            "norm1":            enc_norm1,
            "ffn_hidden":       ffn_hidden,
            "ffn_out":          ffn_out,
            "residual2":        enc_residual2,
            "output":           encoder_output,
        },

        # Decoder
        "decoder": {
            "target":           target,
            "masked_attn":      masked_data,
            "masked_out":       masked_out,
            "residual1":        dec_residual1,
            "norm1":            dec_norm1,
            "cross_attn":       cross_data,
            "cross_out":        cross_out,
            "residual2":        dec_residual2,
            "norm2":            dec_norm2,
            "ffn_hidden":       dec_ffn_hidden,
            "ffn_out":          dec_ffn_out,
            "residual3":        dec_residual3,
            "output":           decoder_output,
        },

        # Prédiction
        "Y_pred":     Y_pred,
        "decoder_flat": decoder_flat,
    }


# ─────────────────────────────────────────────────────────────
# Loss + Backward (simplifié — on calcule juste les gradients principaux)
# ─────────────────────────────────────────────────────────────
def compute_loss_and_simple_backward(X_batch, weights, forward, y_true):
    """
    Backward simplifié : on calcule la loss + le gradient au niveau de la sortie
    + les normes globales (suffit pour la visualisation pédagogique).
    """
    B = X_batch.shape[0]
    y_true = y_true.reshape(-1, 1)
    y_pred = forward["Y_pred"]

    per_seq_error = (y_pred - y_true)
    per_seq_loss  = per_seq_error ** 2
    loss = float(per_seq_loss.mean())

    dY = (2.0 / B) * per_seq_error
    dW_out = forward["decoder_flat"].T @ dY
    db_out = dY.sum(axis=0)

    # Gradient remontant au décodeur (norme indicative)
    dDecoder = dY @ weights["W_out"].T   # (B, d_model)

    # On calcule des "normes" indicatives pour montrer qu'il y a propagation
    # (un vrai backward complet du Transformer = trop complexe pour pédagogie)
    norms = {
        "dY":      float(np.linalg.norm(dY)),
        "dW_out":  float(np.linalg.norm(dW_out)),
        "decoder": float(np.linalg.norm(dDecoder)),
        # On simule la propagation à travers le décodeur puis l'encodeur
        "dec_ffn":     float(np.linalg.norm(dDecoder)) * 0.85,
        "dec_cross":   float(np.linalg.norm(dDecoder)) * 0.65,
        "dec_masked":  float(np.linalg.norm(dDecoder)) * 0.50,
        "encoder":     float(np.linalg.norm(dDecoder)) * 0.40,
        "enc_ffn":     float(np.linalg.norm(dDecoder)) * 0.32,
        "enc_attn":    float(np.linalg.norm(dDecoder)) * 0.25,
        "embedding":   float(np.linalg.norm(dDecoder)) * 0.18,
    }

    return {
        "loss":          loss,
        "per_seq_error": per_seq_error,
        "per_seq_loss":  per_seq_loss,
        "dY":            dY,
        "dW_out":        dW_out,
        "db_out":        db_out,
        "dDecoder":      dDecoder,
        "norms":         norms,
    }


# ─────────────────────────────────────────────────────────────
# API principale
# ─────────────────────────────────────────────────────────────
def run_demo(pipeline, d_model=8, n_heads=2, d_ff=16,
             batch_size=4, seq_length=3, learning_rate=0.01, seed=7):
    batch = build_demo_batch(pipeline, batch_size, seq_length, n_features_show=4)
    X = batch["X_batch"]
    y = batch["y_batch_norm"]
    B, T, F = X.shape

    # S'assurer que d_model est divisible par n_heads
    if d_model % n_heads != 0:
        d_model = n_heads * max(2, d_model // n_heads)

    weights = init_transformer_weights(F, d_model=d_model, n_heads=n_heads, d_ff=d_ff, seed=seed)
    forward = forward_transformer(X, weights)
    bwd = compute_loss_and_simple_backward(X, weights, forward, y)

    y_pred_norm = forward["Y_pred"].flatten()
    y_pred_hours = pipeline.scaler_y.inverse_transform(
        y_pred_norm.reshape(-1, 1)
    ).flatten()

    cfg = weights["config"]

    def tl(arr, decimals=3):
        return _round_tensor(np.asarray(arr), decimals)

    # Sérialisation des données par tête
    def serialize_attention(attn_data):
        return {
            "Q_full": tl(attn_data["Q_full"]),
            "K_full": tl(attn_data["K_full"]),
            "V_full": tl(attn_data["V_full"]),
            "concat": tl(attn_data["concat"]),
            "output": tl(attn_data["output"]),
            "heads":  [
                {
                    "Q":            tl(h["Q"]),
                    "K":            tl(h["K"]),
                    "V":            tl(h["V"]),
                    "scores":       tl(h["scores"]),
                    "attn_weights": tl(h["attn_weights"]),
                    "head_output":  tl(h["head_output"]),
                }
                for h in attn_data["heads"]
            ],
        }

    return {
        "config": {
            "batch_size":    B,
            "seq_length":    T,
            "n_features":    F,
            "d_model":       cfg["d_model"],
            "n_heads":       cfg["n_heads"],
            "d_head":        cfg["d_head"],
            "d_ff":          cfg["d_ff"],
            "learning_rate": learning_rate,
            "machine_id":    getattr(pipeline, "machine_id", 99),
        },

        # Données d'entrée
        "input_tensor_3d": tl(X),
        "feat_names":      batch["feat_names"],
        "y_true_norm":     tl(y),
        "y_true_hours":    tl(batch["y_batch_hours"]),
        "slices_2d":       [tl(X[:, t, :]) for t in range(T)],

        # Poids initiaux (juste les principaux pour pas exploser le JSON)
        "weights": {
            "W_embed": tl(weights["W_embed"]),
            "encoder": {
                "W_q": tl(weights["encoder"]["W_q"]),
                "W_k": tl(weights["encoder"]["W_k"]),
                "W_v": tl(weights["encoder"]["W_v"]),
                "W_o": tl(weights["encoder"]["W_o"]),
                "W_ff1": tl(weights["encoder"]["W_ff1"]),
                "W_ff2": tl(weights["encoder"]["W_ff2"]),
            },
            "decoder": {
                "W_q1": tl(weights["decoder"]["W_q1"]),
                "W_k1": tl(weights["decoder"]["W_k1"]),
                "W_v1": tl(weights["decoder"]["W_v1"]),
                "W_q2": tl(weights["decoder"]["W_q2"]),
                "W_k2": tl(weights["decoder"]["W_k2"]),
                "W_v2": tl(weights["decoder"]["W_v2"]),
                "target_token": tl(weights["decoder"]["target_token"]),
            },
            "W_out": tl(weights["W_out"]),
        },

        # Étape 1 : embedding + PE
        "embedding": {
            "X_embed": tl(forward["X_embed"]),
            "PE":      tl(forward["PE"]),
            "X_pos":   tl(forward["X_pos"]),
        },

        # Étape 2 : ENCODER
        "encoder_steps": {
            "self_attn":  serialize_attention(forward["encoder"]["self_attn"]),
            "attn_out":   tl(forward["encoder"]["attn_out"]),
            "residual1":  tl(forward["encoder"]["residual1"]),
            "norm1":      tl(forward["encoder"]["norm1"]),
            "ffn_hidden": tl(forward["encoder"]["ffn_hidden"]),
            "ffn_out":    tl(forward["encoder"]["ffn_out"]),
            "residual2":  tl(forward["encoder"]["residual2"]),
            "output":     tl(forward["encoder"]["output"]),
        },

        # Étape 3 : DECODER
        "decoder_steps": {
            "target":      tl(forward["decoder"]["target"]),
            "masked_attn": serialize_attention(forward["decoder"]["masked_attn"]),
            "masked_out":  tl(forward["decoder"]["masked_out"]),
            "residual1":   tl(forward["decoder"]["residual1"]),
            "norm1":       tl(forward["decoder"]["norm1"]),
            "cross_attn":  serialize_attention(forward["decoder"]["cross_attn"]),
            "cross_out":   tl(forward["decoder"]["cross_out"]),
            "residual2":   tl(forward["decoder"]["residual2"]),
            "norm2":       tl(forward["decoder"]["norm2"]),
            "ffn_hidden":  tl(forward["decoder"]["ffn_hidden"]),
            "ffn_out":     tl(forward["decoder"]["ffn_out"]),
            "residual3":   tl(forward["decoder"]["residual3"]),
            "output":      tl(forward["decoder"]["output"]),
        },

        # Prédiction
        "prediction": {
            "decoder_flat": tl(forward["decoder_flat"]),
            "Y_pred_norm":  tl(forward["Y_pred"].flatten()),
            "Y_pred_hours": tl(y_pred_hours),
        },

        # Loss
        "loss_info": {
            "per_seq_error_norm": tl(bwd["per_seq_error"].flatten()),
            "per_seq_loss":       tl(bwd["per_seq_loss"].flatten()),
            "loss":               round(bwd["loss"], 6),
        },

        # Backward (simplifié : on a les gradients principaux + les normes)
        "backward": {
            "dY":      tl(bwd["dY"]),
            "dW_out":  tl(bwd["dW_out"]),
            "db_out":  tl(bwd["db_out"]),
            "dDecoder": tl(bwd["dDecoder"]),
            "norms":   bwd["norms"],
        },
    }