"""
rnn_lab.py — Simulation NumPy pure d'un RNN pour le laboratoire interactif.

Ce module simule un RNN de maintenance prédictive étape par étape :
- Forward pass avec états cachés détaillés (h_t pour chaque t et chaque couche)
- Calcul de la loss (MSE)
- Backward pass avec gradients (chain rule codée manuellement)
- Scénarios pédagogiques : séquence stable vs dégradation progressive

Tout est en NumPy pour un contrôle total et des formules LaTeX qui correspondent
EXACTEMENT au code exécuté.
"""

import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
# Configuration des 31 features (groupées par catégorie)
# ─────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "capteurs_bruts": {
        "features": ["volt", "rotate", "pressure", "vibration"],
        "color":    "#4fc3f7",
        "icon":     "⚡",
    },
    "rolling_3h_mean": {
        "features": ["volt_mean_3h", "rotate_mean_3h", "pressure_mean_3h", "vibration_mean_3h"],
        "color":    "#81c784",
        "icon":     "📈",
    },
    "rolling_3h_std": {
        "features": ["volt_std_3h", "rotate_std_3h", "pressure_std_3h", "vibration_std_3h"],
        "color":    "#ffb74d",
        "icon":     "📊",
    },
    "rolling_24h_mean": {
        "features": ["volt_mean_24h", "rotate_mean_24h", "pressure_mean_24h", "vibration_mean_24h"],
        "color":    "#ce93d8",
        "icon":     "📉",
    },
    "rolling_24h_std": {
        "features": ["volt_std_24h", "rotate_std_24h", "pressure_std_24h", "vibration_std_24h"],
        "color":    "#f06292",
        "icon":     "📐",
    },
    "composants_age": {
        "features": ["comp1_age", "comp2_age", "comp3_age", "comp4_age"],
        "color":    "#aed581",
        "icon":     "⚙️",
    },
    "erreurs": {
        "features": ["error1", "error2", "error3", "error4", "error5"],
        "color":    "#ff8a65",
        "icon":     "⚠️",
    },
    "machine_info": {
        "features": ["model_encoded", "machine_age_years"],
        "color":    "#90a4ae",
        "icon":     "🏭",
    },
}

# Liste plate des 31 features dans l'ordre utilisé par le modèle
ALL_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group["features"])


# ─────────────────────────────────────────────────────────────
# Génération des scénarios pédagogiques
# ─────────────────────────────────────────────────────────────
def generate_scenario(scenario_type: str, lookback: int = 24, seed: int = 42) -> np.ndarray:
    """
    Génère une séquence de 24 pas de temps × 31 features.

    scenario_type:
      - 'stable'      : Machine en bon état. Valeurs quasi-constantes. RUL élevée.
      - 'degradation' : Dégradation progressive. Vibrations ↑, température ↑, erreurs.
    """
    np.random.seed(seed)
    n_features = len(ALL_FEATURES)
    x = np.zeros((lookback, n_features), dtype=np.float32)

    # Index utiles dans ALL_FEATURES
    idx_volt      = ALL_FEATURES.index("volt")
    idx_rotate    = ALL_FEATURES.index("rotate")
    idx_pressure  = ALL_FEATURES.index("pressure")
    idx_vibration = ALL_FEATURES.index("vibration")

    idx_vib_mean_3h  = ALL_FEATURES.index("vibration_mean_3h")
    idx_vib_std_3h   = ALL_FEATURES.index("vibration_std_3h")
    idx_vib_mean_24h = ALL_FEATURES.index("vibration_mean_24h")
    idx_vib_std_24h  = ALL_FEATURES.index("vibration_std_24h")

    idx_volt_mean_3h  = ALL_FEATURES.index("volt_mean_3h")
    idx_volt_std_3h   = ALL_FEATURES.index("volt_std_3h")

    idx_error1 = ALL_FEATURES.index("error1")
    idx_error2 = ALL_FEATURES.index("error2")

    idx_comp1 = ALL_FEATURES.index("comp1_age")
    idx_comp2 = ALL_FEATURES.index("comp2_age")
    idx_comp3 = ALL_FEATURES.index("comp3_age")
    idx_comp4 = ALL_FEATURES.index("comp4_age")

    idx_machine_age = ALL_FEATURES.index("machine_age_years")

    if scenario_type == "stable":
        # Valeurs normales, faible bruit
        for t in range(lookback):
            noise = np.random.normal(0, 0.05, n_features).astype(np.float32)
            x[t, idx_volt]      = 0.5 + noise[0]       # ≈ 170V normalisé
            x[t, idx_rotate]    = 0.5 + noise[1]       # ≈ 450 RPM
            x[t, idx_pressure]  = 0.5 + noise[2]       # ≈ 100 psi
            x[t, idx_vibration] = 0.3 + noise[3]       # basses vibrations

            # Rolling features stables
            x[t, idx_vib_mean_3h]  = 0.3 + noise[4] * 0.3
            x[t, idx_vib_std_3h]   = 0.1 + abs(noise[5] * 0.2)
            x[t, idx_vib_mean_24h] = 0.3 + noise[6] * 0.2
            x[t, idx_vib_std_24h]  = 0.1 + abs(noise[7] * 0.1)

            x[t, idx_volt_mean_3h] = 0.5 + noise[8] * 0.2
            x[t, idx_volt_std_3h]  = 0.1 + abs(noise[9] * 0.1)

            # Âge composants (croissance linéaire lente)
            x[t, idx_comp1] = 0.2 + t * 0.005
            x[t, idx_comp2] = 0.3 + t * 0.005
            x[t, idx_comp3] = 0.15 + t * 0.005
            x[t, idx_comp4] = 0.25 + t * 0.005

            x[t, idx_machine_age] = 0.4  # machine d'âge moyen

    elif scenario_type == "degradation":
        # Dégradation progressive : déclenchement à t=15
        for t in range(lookback):
            noise = np.random.normal(0, 0.05, n_features).astype(np.float32)

            if t < 15:
                # Début : état quasi normal
                x[t, idx_volt]      = 0.5 + noise[0]
                x[t, idx_rotate]    = 0.5 + noise[1]
                x[t, idx_pressure]  = 0.5 + noise[2]
                x[t, idx_vibration] = 0.35 + noise[3]
                x[t, idx_vib_mean_3h]  = 0.35
                x[t, idx_vib_std_3h]   = 0.12
                x[t, idx_vib_mean_24h] = 0.35
                x[t, idx_vib_std_24h]  = 0.12
            else:
                # À partir de t=15 : dégradation rapide
                progress = (t - 14) / 10.0  # 0.1 à 1.0 sur 10 pas
                x[t, idx_volt]      = 0.5 + progress * 0.3 + noise[0]
                x[t, idx_rotate]    = 0.5 + progress * 0.2 + noise[1]
                x[t, idx_pressure]  = 0.5 - progress * 0.15 + noise[2]  # chute de pression
                x[t, idx_vibration] = 0.35 + progress * 0.55 + noise[3]  # VIBRATIONS ↑↑

                # Rolling 3h réagit rapidement (court terme)
                x[t, idx_vib_mean_3h] = 0.35 + progress * 0.5
                x[t, idx_vib_std_3h]  = 0.12 + progress * 0.6   # std explose
                # Rolling 24h lag
                x[t, idx_vib_mean_24h] = 0.35 + progress * 0.2
                x[t, idx_vib_std_24h]  = 0.12 + progress * 0.3

                x[t, idx_volt_mean_3h] = 0.5 + progress * 0.25
                x[t, idx_volt_std_3h]  = 0.1 + progress * 0.4

                # Erreurs apparaissent
                if t >= 18:
                    x[t, idx_error1] = 1.0
                if t >= 21:
                    x[t, idx_error2] = 1.0

            # Âge composants toujours croissant
            x[t, idx_comp1] = 0.7 + t * 0.01
            x[t, idx_comp2] = 0.8 + t * 0.01
            x[t, idx_comp3] = 0.65 + t * 0.01
            x[t, idx_comp4] = 0.75 + t * 0.01

            x[t, idx_machine_age] = 0.7  # machine plus âgée

    # Clipper entre 0 et 1
    x = np.clip(x, 0.0, 1.0)
    return x


# ─────────────────────────────────────────────────────────────
# Initialisation des poids du RNN (Xavier)
# ─────────────────────────────────────────────────────────────
def initialize_weights(n_features: int, layer_units: List[int], seed: int = 42) -> Dict:
    """
    Initialise les poids pour un RNN multi-couches avec Xavier initialization.

    Pour chaque couche L :
      W_xh^(L) : (n_input, n_units) — entrée → hidden
      W_hh^(L) : (n_units, n_units) — récurrence
      b^(L)    : (n_units,)         — biais

    La dernière couche ajoute :
      W_hy : (n_last_units, 1) — hidden → output
      b_y  : (1,)
    """
    rng = np.random.RandomState(seed)
    weights = {"layers": []}

    prev_dim = n_features
    for L, units in enumerate(layer_units):
        limit_xh = np.sqrt(6.0 / (prev_dim + units))
        limit_hh = np.sqrt(6.0 / (units + units))
        layer = {
            "units":    units,
            "input_dim": prev_dim,
            "W_xh":     rng.uniform(-limit_xh, limit_xh, (prev_dim, units)).astype(np.float32),
            "W_hh":     rng.uniform(-limit_hh, limit_hh, (units, units)).astype(np.float32),
            "b":        np.zeros(units, dtype=np.float32),
        }
        weights["layers"].append(layer)
        prev_dim = units

    # Couche de sortie (hidden final → 1 scalaire RUL)
    limit_y = np.sqrt(6.0 / (prev_dim + 1))
    weights["W_hy"] = rng.uniform(-limit_y, limit_y, (prev_dim, 1)).astype(np.float32)
    weights["b_y"]  = np.zeros(1, dtype=np.float32)

    return weights


# ─────────────────────────────────────────────────────────────
# Forward pass avec tous les états cachés sauvegardés
# ─────────────────────────────────────────────────────────────
def forward_pass(x: np.ndarray, weights: Dict) -> Dict:
    """
    x : (T, n_features) — séquence d'entrée
    weights : dict renvoyé par initialize_weights

    Renvoie un dict complet avec :
      - h_history[L][t] : état caché de la couche L au temps t (shape: units)
      - pre_activations[L][t] : valeur avant tanh (pour backprop)
      - y_pred : prédiction finale (scalaire)
    """
    T = x.shape[0]
    n_layers = len(weights["layers"])

    # Stockage des états cachés et pré-activations
    h_history       = [[None] * T for _ in range(n_layers)]
    pre_activations = [[None] * T for _ in range(n_layers)]

    # État initial h_{-1} = 0 pour chaque couche
    h_prev = [np.zeros(layer["units"], dtype=np.float32) for layer in weights["layers"]]

    # Déroulement temporel
    for t in range(T):
        input_t = x[t]  # entrée de la 1ère couche au temps t

        for L, layer in enumerate(weights["layers"]):
            # Pré-activation : z = W_xh @ input + W_hh @ h_prev + b
            z_t = input_t @ layer["W_xh"] + h_prev[L] @ layer["W_hh"] + layer["b"]
            h_t = np.tanh(z_t)

            pre_activations[L][t] = z_t
            h_history[L][t]       = h_t

            # La sortie de cette couche devient l'entrée de la suivante
            input_t = h_t
            h_prev[L] = h_t

    # Couche de sortie sur le dernier h_T de la dernière couche
    h_final = h_history[-1][T - 1]
    y_pred  = h_final @ weights["W_hy"] + weights["b_y"]

    return {
        "h_history":       h_history,
        "pre_activations": pre_activations,
        "y_pred":          float(y_pred[0]),
        "h_final":         h_final,
        "T":               T,
        "n_layers":        n_layers,
    }


# ─────────────────────────────────────────────────────────────
# Backward pass (chain rule codée manuellement)
# ─────────────────────────────────────────────────────────────
def backward_pass(x: np.ndarray, weights: Dict, forward: Dict, y_true: float) -> Dict:
    """
    Calcule les gradients de la loss MSE par rapport à tous les poids.

    Loss = (y_true - y_pred)^2

    Étapes :
      1. dL/dy_pred = -2 * (y_true - y_pred)
      2. Gradients couche de sortie (W_hy, b_y)
      3. Rétropropagation temporelle (BPTT) pour chaque couche RNN
      4. Rétropropagation entre couches empilées
    """
    T        = forward["T"]
    n_layers = forward["n_layers"]
    h_history       = forward["h_history"]
    pre_activations = forward["pre_activations"]
    y_pred   = forward["y_pred"]
    h_final  = forward["h_final"]

    # Gradients à retourner
    grads = {
        "layers": [{
            "dW_xh": np.zeros_like(layer["W_xh"]),
            "dW_hh": np.zeros_like(layer["W_hh"]),
            "db":    np.zeros_like(layer["b"]),
        } for layer in weights["layers"]],
        "dW_hy": np.zeros_like(weights["W_hy"]),
        "db_y":  np.zeros_like(weights["b_y"]),
    }

    # ── 1. Gradient de la loss MSE ──
    # L = (y_true - y_pred)^2  →  dL/dy_pred = -2 * (y_true - y_pred)
    dL_dy = -2.0 * (y_true - y_pred)
    loss  = (y_true - y_pred) ** 2

    # ── 2. Couche de sortie ──
    # y_pred = h_final @ W_hy + b_y
    grads["dW_hy"] = np.outer(h_final, np.array([dL_dy])).astype(np.float32)
    grads["db_y"]  = np.array([dL_dy], dtype=np.float32)

    # Gradient remontant vers h_final
    # dy/dh_final = W_hy  →  dL/dh_final = dL/dy * W_hy.T
    dL_dh_top = (dL_dy * weights["W_hy"].flatten()).astype(np.float32)

    # ── 3. Rétropropagation temporelle (BPTT) par couche ──
    # On part de la couche du haut et on descend
    dL_dh_next_layer = [None] * T  # gradient entrant depuis la couche supérieure
    dL_dh_next_layer[T - 1] = dL_dh_top.copy()
    for t in range(T - 1):
        dL_dh_next_layer[t] = np.zeros_like(dL_dh_top)

    for L in reversed(range(n_layers)):
        layer = weights["layers"][L]
        units = layer["units"]

        # Gradient qui descend depuis la couche L+1 (ou sortie si L est la dernière)
        # Pour la dernière couche, seul t=T-1 a un gradient direct (via y_pred)
        # Pour les couches intermédiaires, chaque t a un gradient venant de la couche supérieure

        # dL/dh^(L)_t vient de 2 sources :
        #   (a) la couche supérieure L+1 au même t  (stocké dans dL_dh_next_layer[t])
        #   (b) le futur temporel : dL/dh^(L)_{t+1} via W_hh

        dL_dh_future = np.zeros(units, dtype=np.float32)

        # Pour propager vers la couche L-1 (si elle existe)
        dL_dinput_per_t = [None] * T  # pour chaque t : gradient vers l'entrée (= h^(L-1)_t)

        for t in reversed(range(T)):
            # Gradient total arrivant à h^(L)_t
            dL_dh_t = dL_dh_next_layer[t] + dL_dh_future

            # Traversée du tanh : dh/dz = 1 - tanh(z)^2 = 1 - h^2
            h_t  = h_history[L][t]
            dh_dz = 1.0 - h_t ** 2
            dL_dz = dL_dh_t * dh_dz

            # Gradients sur les poids
            # z = input @ W_xh + h_prev @ W_hh + b
            if L == 0:
                input_t = x[t]
            else:
                input_t = h_history[L - 1][t]

            if t > 0:
                h_prev_t = h_history[L][t - 1]
            else:
                h_prev_t = np.zeros(units, dtype=np.float32)

            grads["layers"][L]["dW_xh"] += np.outer(input_t, dL_dz)
            grads["layers"][L]["dW_hh"] += np.outer(h_prev_t, dL_dz)
            grads["layers"][L]["db"]    += dL_dz

            # Gradient vers l'entrée (pour propager à la couche L-1)
            dL_dinput_per_t[t] = dL_dz @ layer["W_xh"].T

            # Gradient vers le passé temporel
            dL_dh_future = dL_dz @ layer["W_hh"].T

        # Préparer dL_dh_next_layer pour la couche L-1
        if L > 0:
            for t in range(T):
                dL_dh_next_layer[t] = dL_dinput_per_t[t]

    # Normes des gradients (pour analyse pédagogique)
    grad_norms = {
        "layers": [{
            "dW_xh_norm": float(np.linalg.norm(grads["layers"][L]["dW_xh"])),
            "dW_hh_norm": float(np.linalg.norm(grads["layers"][L]["dW_hh"])),
            "db_norm":    float(np.linalg.norm(grads["layers"][L]["db"])),
        } for L in range(n_layers)],
        "dW_hy_norm": float(np.linalg.norm(grads["dW_hy"])),
        "db_y_norm":  float(np.linalg.norm(grads["db_y"])),
    }

    return {
        "grads":      grads,
        "grad_norms": grad_norms,
        "loss":       float(loss),
        "dL_dy":      float(dL_dy),
        "error":      float(y_true - y_pred),
    }


# ─────────────────────────────────────────────────────────────
# API principale : simulation complète pour le frontend
# ─────────────────────────────────────────────────────────────
def simulate_rnn_lab(
    layer_units: List[int],
    scenario: str = "degradation",
    lookback: int = 24,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    Point d'entrée principal appelé par la route FastAPI /api/rnn_lab/simulate.

    Renvoie un JSON complet avec :
      - config        : paramètres de la simulation
      - scenario      : type et description du scénario
      - input         : séquence x (T × 31) et groupes de features
      - weights       : tous les poids initialisés
      - forward       : états cachés à chaque t, prédiction
      - loss_info     : loss MSE, erreur
      - backward      : gradients et leurs normes
      - update        : nouvelles valeurs des poids après une étape de SGD
    """
    n_features = len(ALL_FEATURES)

    # Vérifier la configuration
    if not all(1 <= u <= 256 for u in layer_units):
        raise ValueError("Chaque couche doit avoir entre 1 et 256 neurones")
    if not 1 <= len(layer_units) <= 3:
        raise ValueError("Entre 1 et 3 couches supportées")

    # 1. Générer les données
    x = generate_scenario(scenario, lookback=lookback, seed=seed)

    # 2. RUL cible (simule un label supervisé)
    if scenario == "stable":
        y_true = 250.0  # RUL élevée en heures
    else:
        y_true = 12.0   # Panne imminente

    # 3. Initialiser les poids
    weights = initialize_weights(n_features, layer_units, seed=seed)

    # 4. Forward pass
    forward = forward_pass(x, weights)

    # 5. Dénormaliser la prédiction (simule un MinMaxScaler sur RUL)
    # Si y_pred ∈ [-1, 1] ≈ tanh, on la mappe à [0, 400h]
    y_pred_raw = forward["y_pred"]
    y_pred_hours = max(0.0, (y_pred_raw + 1.0) / 2.0 * 400.0)

    # 6. Backward pass (avec la cible en espace "tanh")
    # On convertit y_true en espace [-1, 1] pour rester cohérent
    y_true_normalized = (y_true / 400.0) * 2.0 - 1.0
    backward = backward_pass(x, weights, forward, y_true_normalized)

    # 7. Mise à jour des poids (une étape de SGD pour animer)
    updated_weights = _sgd_step(weights, backward["grads"], learning_rate)

    # ─── Sérialisation pour JSON ───
    def serialize_layer(layer):
        return {
            "units":     layer["units"],
            "input_dim": layer["input_dim"],
            "W_xh_shape": list(layer["W_xh"].shape),
            "W_hh_shape": list(layer["W_hh"].shape),
            # On renvoie un échantillon 8×8 max pour l'affichage heatmap
            "W_xh_sample": layer["W_xh"][:8, :8].tolist(),
            "W_hh_sample": layer["W_hh"][:8, :8].tolist(),
            "b_sample":    layer["b"][:8].tolist(),
            "W_xh_stats": _tensor_stats(layer["W_xh"]),
            "W_hh_stats": _tensor_stats(layer["W_hh"]),
        }

    def serialize_grads(grads_layer):
        return {
            "dW_xh_sample": grads_layer["dW_xh"][:8, :8].tolist(),
            "dW_hh_sample": grads_layer["dW_hh"][:8, :8].tolist(),
            "db_sample":    grads_layer["db"][:8].tolist(),
            "dW_xh_stats": _tensor_stats(grads_layer["dW_xh"]),
            "dW_hh_stats": _tensor_stats(grads_layer["dW_hh"]),
        }

    # États cachés : on envoie les activations par (L, t) pour animation
    # Chaque h_t est un vecteur de taille units — on envoie tout
    h_history_serialized = [
        [h.tolist() for h in forward["h_history"][L]]
        for L in range(len(layer_units))
    ]

    return {
        "config": {
            "layer_units":   layer_units,
            "n_layers":      len(layer_units),
            "lookback":      lookback,
            "n_features":    n_features,
            "learning_rate": learning_rate,
        },
        "scenario": {
            "type":        scenario,
            "description": (
                "Machine en bon état. Valeurs stables. RUL élevée."
                if scenario == "stable" else
                "Dégradation progressive à partir de t=15. Vibrations et température en hausse."
            ),
            "y_true_hours":      y_true,
            "trigger_timestep":  None if scenario == "stable" else 15,
        },
        "input": {
            "x":              x.tolist(),  # (T, 31)
            "feature_names":  ALL_FEATURES,
            "feature_groups": FEATURE_GROUPS,
        },
        "weights": {
            "layers": [serialize_layer(L) for L in weights["layers"]],
            "W_hy_sample": weights["W_hy"][:8].tolist(),
            "W_hy_stats":  _tensor_stats(weights["W_hy"]),
            "b_y":         weights["b_y"].tolist(),
        },
        "forward": {
            "h_history":      h_history_serialized,
            "y_pred_raw":     y_pred_raw,
            "y_pred_hours":   y_pred_hours,
        },
        "loss_info": {
            "y_true_normalized": y_true_normalized,
            "y_pred_normalized": y_pred_raw,
            "y_true_hours":      y_true,
            "y_pred_hours":      y_pred_hours,
            "error_normalized":  backward["error"],
            "error_hours":       y_true - y_pred_hours,
            "loss":              backward["loss"],
            "dL_dy":             backward["dL_dy"],
            "mae":               float(abs(y_true - y_pred_hours)),
            "rmse":              float(abs(y_true - y_pred_hours)),  # 1 échantillon
        },
        "backward": {
            "grads_per_layer": [serialize_grads(g) for g in backward["grads"]["layers"]],
            "dW_hy_sample":    backward["grads"]["dW_hy"][:8].tolist(),
            "db_y":            backward["grads"]["db_y"].tolist(),
            "grad_norms":      backward["grad_norms"],
        },
        "update": {
            "learning_rate": learning_rate,
            "layers": [{
                "W_xh_delta_sample":
                    (updated_weights["layers"][L]["W_xh"][:8, :8] -
                     weights["layers"][L]["W_xh"][:8, :8]).tolist(),
                "W_xh_new_sample":
                    updated_weights["layers"][L]["W_xh"][:8, :8].tolist(),
            } for L in range(len(layer_units))],
        },
    }


def _sgd_step(weights: Dict, grads: Dict, lr: float) -> Dict:
    """Une étape de SGD : W_new = W_old - lr * dW."""
    new_weights = {"layers": []}
    for L, layer in enumerate(weights["layers"]):
        g = grads["layers"][L]
        new_weights["layers"].append({
            "units":     layer["units"],
            "input_dim": layer["input_dim"],
            "W_xh": layer["W_xh"] - lr * g["dW_xh"],
            "W_hh": layer["W_hh"] - lr * g["dW_hh"],
            "b":    layer["b"]    - lr * g["db"],
        })
    new_weights["W_hy"] = weights["W_hy"] - lr * grads["dW_hy"]
    new_weights["b_y"]  = weights["b_y"]  - lr * grads["db_y"]
    return new_weights


def _tensor_stats(tensor: np.ndarray) -> Dict:
    """Statistiques résumées pour afficher les tenseurs."""
    flat = tensor.flatten()
    return {
        "min":  float(flat.min()),
        "max":  float(flat.max()),
        "mean": float(flat.mean()),
        "std":  float(flat.std()),
        "shape": list(tensor.shape),
    }


# ─────────────────────────────────────────────────────────────
# Test rapide en standalone
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Test — Laboratoire RNN")
    result = simulate_rnn_lab(
        layer_units=[64, 32],
        scenario="degradation",
        lookback=24,
    )
    print(f"\n✓ Simulation terminée")
    print(f"  Scenario     : {result['scenario']['type']}")
    print(f"  Description  : {result['scenario']['description']}")
    print(f"  Couches      : {result['config']['layer_units']}")
    print(f"  y_pred       : {result['forward']['y_pred_hours']:.2f}h")
    print(f"  y_true       : {result['loss_info']['y_true_hours']:.2f}h")
    print(f"  Erreur       : {result['loss_info']['error_hours']:.2f}h")
    print(f"  Loss (MSE)   : {result['loss_info']['loss']:.6f}")
    print(f"\n  Normes des gradients :")
    for L, gn in enumerate(result['backward']['grad_norms']['layers']):
        print(f"    Couche {L+1} : |dW_xh|={gn['dW_xh_norm']:.4f} "
              f"|dW_hh|={gn['dW_hh_norm']:.4f} |db|={gn['db_norm']:.4f}")