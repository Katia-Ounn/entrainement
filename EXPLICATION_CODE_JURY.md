# 📖 Explication du code pour la soutenance (PFE Cevital — Maintenance prédictive)

Ce document explique **ligne par ligne**, en français, les parties clés du code.
Objectif : pouvoir expliquer au jury **ce qu'on a fait** et **pourquoi**.

---

## 🗺️ Vue d'ensemble du pipeline

```
Données brutes GMAO (failure.csv + equipment.csv)
        │
        ▼
1) compute_features()      → calcule le RUL (cible) + les features (DSLF, MTBF, rolling…)
        │
        ▼
2) prepare_sequences()     → split temporel 70/15/15 + normalisation + fenêtres glissantes
        │
        ▼
3) Entraînement (LSTM/GRU) → manuel / AutoML (évaluation)  OU  full (déploiement)
        │
        ▼
4) next_failures()         → prédit le RUL de chaque composant → date de prochaine panne
```

**Le RUL** (Remaining Useful Life / Durée de vie restante) = **nombre de jours jusqu'à la prochaine panne**. C'est ce que le modèle apprend à prédire (problème de **régression**).

---

## 1️⃣ Le calcul du RUL — `calculate_rul_v1()`

> Fichier : `backend/pipelines/cevital_pipeline.py`
> Rôle : pour chaque jour d'un composant, calculer combien de jours le séparent de sa **prochaine** panne.

```python
def calculate_rul_v1(df_comp):
    # On travaille composant par composant, trié par date croissante
    df_comp = df_comp.sort_values("date").copy().reset_index(drop=True)

    # Liste des dates où ce composant est tombé en panne / a été maintenu
    failure_dates   = df_comp[df_comp["failure"] == 1]["date"].values
    maint_dates_arr = df_comp[df_comp["maintenance"] == 1]["date"].values

    # Si le composant n'a jamais de panne → on ne peut pas calculer de RUL
    if len(failure_dates) == 0:
        return pd.DataFrame()

    end_of_year = timeline_end       # borne : fin de la période de données
    ruls = []
    for _, row in df_comp.iterrows():   # on parcourt chaque jour
        d = row["date"]

        # Pannes et maintenances DÉJÀ arrivées à la date d (le passé)
        past_fails  = failure_dates[failure_dates <= d]
        past_maints = maint_dates_arr[maint_dates_arr <= d]

        # Cas spécial : le composant est ACTUELLEMENT en panne (pas encore réparé)
        if len(past_fails) > 0:
            last_fail  = past_fails[-1]                                   # dernière panne
            last_maint = past_maints[-1] if len(past_maints) > 0 else None # dernière maint.
            # Si la dernière panne est postérieure à la dernière maintenance
            # → le composant est encore en panne → RUL = 0
            if last_maint is None or last_fail > last_maint:
                ruls.append(0)
                continue

        # Cas normal : on cherche la PROCHAINE panne (dans le futur)
        future_fails = failure_dates[failure_dates > d]
        if len(future_fails) > 0:
            # RUL = nombre de jours entre aujourd'hui (d) et la prochaine panne
            rul = int((pd.Timestamp(future_fails[0]) - d).days)
        else:
            # Pas de panne future connue → RUL = jours jusqu'à la fin des données
            rul = (end_of_year - d).days
        ruls.append(rul)

    df_comp["RUL"] = ruls   # on ajoute la colonne cible
    return df_comp
```

**À dire au jury :**
- Le RUL est la **variable cible** : pour chaque jour, c'est le **nombre de jours avant la prochaine panne** du composant.
- On gère le cas « composant déjà en panne non réparé » → RUL = 0 (urgence maximale).
- C'est calculé **par composant** car chaque équipement a son propre historique.

---

## 2️⃣ Le Feature Engineering — `add_features()`

> Rôle : créer les **variables explicatives** (features) que le modèle utilisera pour prédire le RUL. Tout est calculé **par composant** et **sans regarder le futur** (`.shift(1)`).

```python
def add_features(df_comp):
    df_comp = df_comp.sort_values("date").copy().reset_index(drop=True)

    # 1) FENÊTRES ROULANTES : nb de pannes / maintenances sur les 7, 30, 90 derniers jours
    for w in [7, 30, 90]:
        df_comp[f"pannes_{w}j"] = (
            df_comp["failure"]
              .shift(1)                       # .shift(1) = on EXCLUT le jour courant (pas de fuite du futur)
              .rolling(w, min_periods=0)      # fenêtre glissante de w jours
              .sum().fillna(0).astype(int)    # somme = combien de pannes dans la fenêtre
        )
        df_comp[f"maint_{w}j"] = (
            df_comp["maintenance"].shift(1).rolling(w, min_periods=0).sum().fillna(0).astype(int)
        )

    # 2) DSLF — Days Since Last Failure (jours depuis la dernière panne)
    last_fail = pd.Timestamp(f"{self.year - 1}-12-31")  # init : avant le début des données
    dslf = []
    for _, row in df_comp.iterrows():
        if row["failure"] == 1:        # si panne aujourd'hui → on remet le compteur à cette date
            last_fail = row["date"]
        dslf.append((row["date"] - last_fail).days)   # nb de jours écoulés depuis la dernière panne
    df_comp["DSLF"] = dslf

    # 3) DSLM — Days Since Last Maintenance (même logique pour la maintenance)
    last_maint = pd.Timestamp(f"{self.year - 1}-12-31")
    dslm = []
    for _, row in df_comp.iterrows():
        if row["maintenance"] == 1:
            last_maint = row["date"]
        dslm.append((row["date"] - last_maint).days)
    df_comp["DSLM"] = dslm

    # 4) MTBF rolling — Mean Time Between Failures (temps moyen entre les 3 dernières pannes)
    mtbf_col, gap_list, last_fail_date, current_mtbf = [], [], None, np.nan
    for _, row in df_comp.iterrows():
        if row["failure"] == 1:
            if last_fail_date is not None:
                gap = (row["date"] - last_fail_date).days   # écart avec la panne précédente
                gap_list.append(gap)
                # moyenne des 3 derniers écarts → fiabilité récente du composant
                current_mtbf = round(sum(gap_list[-3:]) / len(gap_list[-3:]), 1)
            last_fail_date = row["date"]
        mtbf_col.append(current_mtbf)
    df_comp["MTBF_rolling"] = mtbf_col
    df_comp["has_mtbf"]     = df_comp["MTBF_rolling"].notna().astype(int)  # 1 si on a déjà un MTBF
    df_comp["MTBF_rolling"] = df_comp["MTBF_rolling"].fillna(0)

    # 6) SAISONNALITÉ — encodage cyclique du mois (sin/cos)
    df_comp["month"]     = df_comp["date"].dt.month
    df_comp["month_sin"] = np.sin(2 * np.pi * df_comp["month"] / 12)   # décembre (12) proche de janvier (1)
    df_comp["month_cos"] = np.cos(2 * np.pi * df_comp["month"] / 12)

    # 7) RATIO DSLF / MTBF — feature dérivée : « le composant a-t-il dépassé sa durée de vie habituelle ? »
    df_comp["dslf_mtbf_ratio"] = df_comp["DSLF"] / (df_comp["MTBF_rolling"] + 1)  # +1 évite la division par 0

    return df_comp
```

**À dire au jury :**
- `.shift(1)` est **crucial** : on ne regarde **jamais** le jour courant ni le futur → pas de **fuite de données** (data leakage).
- **DSLF** (jours depuis la dernière panne) et **MTBF** (temps moyen entre pannes) sont les features les plus parlantes : plus DSLF s'approche du MTBF, plus une panne est probable → le `dslf_mtbf_ratio` capture ça.
- Le mois est encodé en **sin/cos** pour que le modèle comprenne que décembre et janvier sont « proches » (cyclique).

---

## 3️⃣ La préparation des séquences — `prepare_sequences()`

> Rôle : transformer le tableau de features en **séquences temporelles** prêtes pour le LSTM, avec un **split temporel** et une **normalisation sans fuite**.

```python
def prepare_sequences(self, lookback=21, current_max_rul=30,
                      val_ratio=0.15, test_ratio=0.15, weight_factor=4.0, ...):
    df = self.df_export.copy()
    df = df.sort_values([self.COMP_COL, "date"]).reset_index(drop=True)

    # PLAFONNEMENT du RUL : on borne à current_max_rul (ex. 30 jours).
    # Au-delà, peu importe la valeur exacte → on se concentre sur le court terme (l'urgence).
    self.current_max_rul = current_max_rul
    self.lookback        = lookback
    df["RUL"] = np.clip(df["RUL"], 0, current_max_rul)

    # POIDS D'ÉCHANTILLON : on donne plus d'importance aux cas proches de la panne (RUL faible)
    # w = 1 + factor × (1 − RUL/MAX)  → RUL=0 (panne imminente) pèse le plus
    df["sample_weight"] = 1.0 + weight_factor * (1.0 - df["RUL"] / current_max_rul)

    # ENCODAGE du composant en entier (pour la couche Embedding du modèle)
    le = LabelEncoder()
    df["comp_idx"] = le.fit_transform(df[self.COMP_COL].astype(str))
    self.num_classes_comp  = int(df["comp_idx"].nunique())
    self._comp_name_to_idx = {str(c): int(i) for i, c in enumerate(le.classes_)}

    # ── SPLIT TEMPOREL (et non aléatoire !) ──────────────────────────
    # On coupe par DATE : le passé pour entraîner, le futur pour tester.
    d_min       = df["date"].min()
    n_days      = (df["date"].max() - d_min).days
    train_ratio = 1.0 - val_ratio - test_ratio
    split_train = d_min + pd.Timedelta(days=int(n_days * train_ratio))           # fin du train
    split_val   = d_min + pd.Timedelta(days=int(n_days * (train_ratio + val_ratio)))  # fin de la val

    df_train = df[df["date"] <  split_train]                       # 70% le plus ancien
    df_val   = df[(df["date"] >= split_train) & (df["date"] < split_val)]  # 15% intermédiaire
    df_test  = df[df["date"] >= split_val]                         # 15% le plus récent

    # ── NORMALISATION MinMax — fit SUR TRAIN UNIQUEMENT (pas de fuite) ──
    self.scaler_x = MinMaxScaler()
    self.scaler_y = MinMaxScaler()
    X_train_s = self.scaler_x.fit_transform(df_train[self.FEATURE_COLS])  # fit + transform sur train
    X_val_s   = self.scaler_x.transform(df_val[self.FEATURE_COLS])        # transform seulement (mêmes bornes)
    X_test_s  = self.scaler_x.transform(df_test[self.FEATURE_COLS])
    y_train_s = self.scaler_y.fit_transform(df_train[[self.TARGET_COL]])  # on normalise aussi la cible RUL
    y_val_s   = self.scaler_y.transform(df_val[[self.TARGET_COL]])
    y_test_s  = self.scaler_y.transform(df_test[[self.TARGET_COL]])

    # ── FENÊTRES GLISSANTES : transformer en séquences (N, lookback, n_features) ──
    def create_sequences(X_s, y_s, df_meta):
        X_num, X_cat, y_seq, sw_seq = [], [], [], []
        for comp in df_meta[self.COMP_COL].unique():   # composant par composant
            mask = df_meta[self.COMP_COL].values == comp
            X_c  = X_s[mask]                            # features de ce composant
            y_c  = y_s[mask]                            # RUL de ce composant
            n    = int(mask.sum())
            if n <= lookback:                          # pas assez d'historique → on saute
                continue
            for i in range(n - lookback):
                X_num.append(X_c[i : i + lookback])    # fenêtre = lookback jours consécutifs
                X_cat.append(cidx_c[i : i + lookback]) # idx composant (pour l'embedding)
                y_seq.append(y_c[i + lookback][0])     # cible = RUL du jour JUSTE APRÈS la fenêtre
        return np.array(X_num), np.array(X_cat), np.array(y_seq), ...

    # On crée les séquences pour les 3 jeux
    Xn_tr, Xc_tr, ytr, wtr = create_sequences(X_train_s, y_train_s, df_train)
    Xn_va, Xc_va, yva, wva = create_sequences(X_val_s,   y_val_s,   df_val)
    Xn_te, Xc_te, yte, wte = create_sequences(X_test_s,  y_test_s,  df_test)
```

**À dire au jury (TRÈS important) :**
- **Split TEMPOREL, pas aléatoire** : en série temporelle, on entraîne sur le **passé** et on teste sur le **futur**. Un split aléatoire tricherait (le modèle verrait le futur).
- **Normalisation `fit` sur le train seulement** : on évite que les statistiques du test « fuient » dans l'entraînement (data leakage).
- **Fenêtre glissante (`lookback`)** : chaque exemple = `lookback` jours consécutifs → la cible est le RUL du **jour suivant**. Le modèle apprend « vu les N derniers jours, quel est l'état du composant ? ».
- **`sample_weight`** : les jours proches d'une panne (RUL faible) pèsent plus → le modèle se concentre sur ce qui est **critique** (détecter les pannes imminentes).
- **Plafond du RUL (30 j)** : on borne la cible car prédire « panne dans 200 jours » n'a pas d'intérêt opérationnel ; on veut le **court terme**.

---

## 4️⃣ L'entraînement de DÉPLOIEMENT — `train_full()`

> Fichier : `backend/full_trainer.py` (créé pour mettre le modèle en production).
> Idée : une fois le meilleur modèle trouvé (avec split, pour l'évaluer), on le **ré-entraîne sur 100% des données** pour qu'il soit le plus performant possible en production.

```python
class CevitalFullTrainer(CevitalTuner):   # hérite de l'entraîneur normal (réutilise tout)

    async def train_full(self, architecture, embedding_dim, num_layers, units,
                         dropout_rates, learning_rate, epochs, batch_size):
        p = self.pipeline

        # 1) CONCATÉNER train + val + test → on utilise TOUTES les données
        X_num = self._concat(p.X_train_num,  p.X_val_num,  p.X_test_num)
        X_cmp = self._concat(p.X_train_comp, p.X_val_comp, p.X_test_comp)
        y_all = self._concat(p.y_train,      p.y_val,      p.y_test)
        w_all = self._concat(p.w_train,      p.w_val,      p.w_test)

        # 2) Construire le modèle avec les MÊMES hyperparamètres que le champion
        model = build_model_cevital_manual(architecture, self.lookback, self.n_features,
                    self.num_classes_comp, embedding_dim, num_layers, units,
                    dropout_rates, learning_rate)

        # 3) Entraîner pour `epochs` époques EXACTES (= meilleure époque trouvée à l'évaluation),
        #    SANS validation et SANS EarlyStopping (plus de jeu de validation puisqu'on prend tout)
        cb_ws = WebSocketCallback(asyncio.get_event_loop(), self._send_fn, epochs)
        history = await self._fit_in_thread(
            model, x_inputs=[X_num, X_cmp], y=y_all, sample_weight=w_all,
            val_data=None, epochs=epochs, batch_size=batch_size, callbacks=[cb_ws])

        # 4) Métriques calculées sur l'ensemble (indicatif — le modèle a vu ces données)
        y_pred = self.pipeline.predict_with_safety(model, X_num, X_cmp)
        y_true = self.pipeline.scaler_y.inverse_transform(y_all.reshape(-1, 1)).flatten()
        metrics = self._compute_metrics(y_true, y_pred)
        return { "model": model, "metrics": metrics, ... }
```

**À dire au jury :**
- **Méthodologie standard en ML** : on **évalue** d'abord avec un split (pour des métriques honnêtes), **puis** on ré-entraîne le modèle final sur **100% des données** pour le déploiement (plus de données = meilleur modèle).
- **Le nombre d'époques** n'est pas choisi au hasard : c'est la **meilleure époque** (val_loss minimal) trouvée pendant l'entraînement normal avec EarlyStopping → on évite ainsi le sur-apprentissage **sans** avoir besoin d'un jeu de validation.

---

## 5️⃣ La prédiction de la PROCHAINE PANNE — `next_failures()`

> Fichier : `backend/main.py` (endpoint API).
> Rôle : pour **chaque composant**, prendre sa fenêtre la plus récente et prédire sa prochaine panne.

```python
@app.get("/api/experiments/{exp_id}/next_failures")
def get_next_failures(exp_id, db):
    # Charger le pipeline (features) + le modèle entraîné
    pipe  = _prepare_pipeline_sync(exp.dataset_id)
    df    = pipe.df_export
    # compile=False : pour PRÉDIRE on n'a pas besoin de la loss → évite l'erreur de loss custom
    model = load_model(str(model_path), compile=False)

    lookback    = int(pipe.lookback)
    scaler_x    = pipe.scaler_x
    comp_to_idx = pipe._comp_name_to_idx

    df = df.sort_values([COMP, "date"])
    windows_num, windows_cat, meta = [], [], []
    for comp, g in df.groupby(COMP):          # composant par composant
        if len(g) < lookback:                 # pas assez d'historique → on saute
            continue
        last_rows = g.iloc[-lookback:]        # ← LES `lookback` DERNIERS JOURS (la fenêtre la plus récente)
        Xn = scaler_x.transform(last_rows[feature_cols].values)  # mêmes scalers qu'à l'entraînement
        windows_num.append(Xn)
        windows_cat.append(np.full((lookback,), comp_to_idx.get(str(comp), 0)))
        # On garde l'historique des pannes passées (pour le graphe)
        meta.append({ "comp": comp, "last_date": g["date"].iloc[-1], "past_failures": [...] })

    # Prédiction en lot pour tous les composants
    preds = model.predict([np.stack(windows_num), np.stack(windows_cat)]).flatten()
    # On dé-normalise (retour en jours) et on borne au plafond RUL
    ruls  = np.clip(scaler_y.inverse_transform(preds.reshape(-1,1)).flatten(), 0, max_rul)

    results = []
    for mrow, rul in zip(meta, ruls):
        rul_days  = int(round(float(rul)))
        last_dt   = pd.to_datetime(mrow["last_date"])
        # PROCHAINE PANNE = dernière date connue + RUL prédit
        next_fail = (last_dt + pd.Timedelta(days=rul_days)).strftime("%Y-%m-%d")
        results.append({ "comp": ..., "predicted_rul": rul_days,
                         "predicted_next_failure": next_fail, ... })

    results.sort(key=lambda r: r["predicted_rul"])  # le plus urgent en premier
    return { "components": results, ... }
```

**À dire au jury :**
- À l'inférence, pour chaque composant on prend sa **fenêtre la plus récente** (`lookback` derniers jours) → le modèle sort un **RUL** → **prochaine panne = aujourd'hui + RUL**.
- On utilise **exactement les mêmes scalers** que ceux de l'entraînement (chargés du pipeline) → cohérence.
- `compile=False` : pour prédire, on n'a pas besoin de la fonction de perte → ça évite l'erreur de chargement de la loss personnalisée (`asymmetric_rul_loss`).
- Les composants sont **triés par urgence** (RUL croissant) → on voit en premier ceux qui vont tomber en panne le plus tôt.

---

## 🎯 Points clés à retenir pour le jury

| Concept | Pourquoi c'est important |
|---|---|
| **RUL = jours avant la prochaine panne** | C'est la cible de régression — le cœur de la maintenance prédictive |
| **Split TEMPOREL** (passé→futur) | Évite la triche : on teste sur des données futures jamais vues |
| **Normalisation fit sur train seul** | Évite la fuite de données (data leakage) |
| **Fenêtres glissantes (lookback)** | Le modèle apprend des **dynamiques temporelles**, pas d'un instant isolé |
| **Features métier** (DSLF, MTBF, ratio) | Encodent la **connaissance maintenance** (fiabilité, usure) |
| **sample_weight** | Priorise la détection des **pannes imminentes** (cas critiques) |
| **Modèle à 2 entrées** (numérique + embedding composant) | Apprend à la fois la dynamique ET l'identité de chaque composant |
| **Déploiement sur 100% des données** | Pratique standard : évaluer puis ré-entraîner sur tout pour la production |
| **Prédiction = dernière fenêtre → RUL → date** | Transforme une prédiction ML en **information opérationnelle** (date de panne) |
