"""
cevital_pipeline.py
═══════════════════════════════════════════════════════════════════════════
Pipeline complet CEVITAL — Conversion exacte du notebook Pipeline_PFE_Cevital_CHAMPION.ipynb

Auteur     : PFE Master 2 Génie Logiciel
Cible ML   : Régression RUL (Remaining Useful Life) en JOURS
Entrées    : failure1.csv + equipment_clean.csv
Sortie     : Dataset_V1 (23 colonnes) + tenseurs LSTM/GRU prêts (avec embedding composant)

PHASES :
  Phase 1 : EDA Brute (failure1.csv)              → compute_eda_raw()
  Phase 2 : Feature Engineering (8 étapes)        → compute_features()
  Phase 3 : EDA Features Créées                   → compute_eda_features()
  Phase 4 : Prétraitement LSTM (avec current_max_rul dynamique) → prepare_sequences()
  Bonus   : Fusion temporelle de datasets         → merge_new_data()
  Bonus   : Prédiction sécurisée                  → predict_with_safety()
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


class CevitalPipeline:
    """
    Pipeline complet pour les données GMAO Cevital.
    Toutes les étapes du notebook PFE sont encapsulées ici.

    Utilisation typique :
        pipe = CevitalPipeline(config={"year": 2023, "min_failures": 2})
        pipe.load_raw_data("failure1.csv", "equipment_clean.csv")
        eda_raw = pipe.compute_eda_raw()
        feat_info = pipe.compute_features()
        eda_feat = pipe.compute_eda_features()
        seq_info = pipe.prepare_sequences(lookback=30, current_max_rul=30)
        # → utiliser pipe.X_train, pipe.y_train, pipe.w_train, etc.
    """

    PIPELINE_ID = "cevital"
    PIPELINE_NAME = "CEVITAL GMAO"
    PIPELINE_DESCRIPTION = (
        "Pipeline GMAO CEVITAL — Régression RUL (jours) basé sur les données "
        "failure + equipment. Inclut embedding composant et séquençage pondéré."
    )

    # Features utilisées pour le modèle LSTM/GRU (NE PAS toucher l'ordre)
    FEATURE_COLS = [
        "comp_level",
        "pannes_7j", "pannes_30j", "pannes_90j",
        "maint_7j",  "maint_30j",  "maint_90j",
        "DSLF",      "DSLM",
    ]

    TARGET_COL  = "RUL"
    COMP_COL    = "failure_comp"

    # Colonnes du Dataset_V1 (fichier final exporté)
    EXPORT_COLS = [
        "date", "machineID_num", "machineID", "machineID_level",
        "comp_num", "failure_comp", "comp_level",
        "failure", "maintenance",
        "pannes_7j", "pannes_30j", "pannes_90j",
        "maint_7j", "maint_30j", "maint_90j",
        "DSLF", "DSLM", "MTBF_rolling", "has_mtbf",
        "month_sin", "month_cos", "dslf_mtbf_ratio",
        "RUL",
    ]

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # Paramètres pilotés depuis l'extérieur (UI ou main.py)
        self.year         = cfg.get("year", 2023)
        self.min_failures = cfg.get("min_failures", 2)
        self.alert_days   = cfg.get("alert_days", 10)
        self.random_state = cfg.get("random_state", 42)

        # ─── État ────────────────────────────────────────────────
        self.df_fail:    Optional[pd.DataFrame] = None  # raw failure.csv
        self.df_equip:   Optional[pd.DataFrame] = None  # raw equipment.csv
        self.df_panel:   Optional[pd.DataFrame] = None  # après panel composant × jour
        self.df_rul:     Optional[pd.DataFrame] = None  # après calcul RUL
        self.df_final:   Optional[pd.DataFrame] = None  # après feature engineering
        self.df_export:  Optional[pd.DataFrame] = None  # Dataset_V1 exporté

        # ─── Tenseurs LSTM/GRU (après prepare_sequences) ────────
        self.X_train_num: Optional[np.ndarray] = None  # (n, lookback, n_features)
        self.X_train_comp: Optional[np.ndarray] = None # (n,) indices composant
        self.X_test_num:  Optional[np.ndarray] = None
        self.X_test_comp: Optional[np.ndarray] = None
        self.y_train:     Optional[np.ndarray] = None
        self.y_test:      Optional[np.ndarray] = None
        self.w_train:     Optional[np.ndarray] = None  # sample_weights
        self.w_test:      Optional[np.ndarray] = None

        # ─── Scalers et métadonnées ─────────────────────────────
        self.scaler_x: Optional[MinMaxScaler] = None
        self.scaler_y: Optional[MinMaxScaler] = None
        self.num_classes_comp: int = 0
        self.lookback: int = 30
        self.current_max_rul: int = 30  # NOUVEAU : variable pilotée par UI

        # ─── État global ─────────────────────────────────────────
        self.is_ready: bool = False
        self.phases_completed: List[str] = []

    # ═══════════════════════════════════════════════════════════════
    # NORMALISATION FORMAT GMAO (nouveau export → format pipeline)
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        """Détecte le format du CSV : 'new' (export GMAO) ou 'old' (format pipeline)."""
        new_cols = {'date_declaration', 'equipment_code', 'equipment_level', 'parent_code'}
        old_cols = {'WOWO_DECLARATION_DATE', 'WOWO_EQUIPMENT', 'WOWO_EQUIPMENT_LEVEL'}
        if new_cols.issubset(set(df.columns)):
            return 'new'
        if old_cols.issubset(set(df.columns)):
            return 'old'
        return 'unknown'

    @staticmethod
    def normalize_gmao_export(df: pd.DataFrame):
        """
        Normalise un export GMAO (nouveau format) vers le format pipeline.
        Retourne (df_fail_normalized, df_equip_derived).
        Le df_equip est dérivé directement du fichier — pas besoin d'equipment.csv séparé.
        """
        COLUMN_MAP = {
            'date_declaration':      'WOWO_DECLARATION_DATE',
            'date_fin':              'WOWO_END_DATE',
            'date_creation':         'WOWO_CREATION_DATE',
            'equipment_code':        'WOWO_EQUIPMENT',
            'equipment_level':       'WOWO_EQUIPMENT_LEVEL',
            'parent_code':           'failure_parent_code',
            'parent_level':          'failure_parent_level',
            'type_travail':          'WOWO_JOB_CLASS',
            'cout_total':            'WOWO_TOTAL_COST',
            'system_equipment':      'WOWO_SYSTEM_EQUIPMENT',
            'equipment_description': 'WOWO_DESCRIPTION',
            'action_entity':         'WOWO_ACTION_ENTITY',
            'source':                'source',
        }
        df_fail = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        # Dériver equipment.csv depuis les colonnes présentes
        equip_cols = {
            'WOWO_EQUIPMENT':       'EREQ_CODE',
            'WOWO_EQUIPMENT_LEVEL': 'EREQ_LEVEL',
            'failure_parent_code':  'EREQ_PARENT_EQUIPMENT',
            'failure_parent_level': 'EREQ_PARENT_LEVEL',
        }
        avail = {k: v for k, v in equip_cols.items() if k in df_fail.columns}
        df_equip = df_fail[list(avail.keys())].drop_duplicates().rename(columns=avail)

        return df_fail, df_equip

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 — CHARGEMENT + EDA BRUTE
    # ═══════════════════════════════════════════════════════════════
    def load_raw_data(self, failure_path: str, equipment_path: str) -> Dict:
        """
        Charge les deux CSV nécessaires : failure et equipment.
        Retourne les métadonnées (nb lignes, période, etc.).
        """
        if not Path(failure_path).exists():
            raise FileNotFoundError(f"Fichier introuvable : {failure_path}")
        if not Path(equipment_path).exists():
            raise FileNotFoundError(f"Fichier introuvable : {equipment_path}")

        self.df_fail = pd.read_csv(failure_path, encoding="utf-8-sig")
        self.df_equip = pd.read_csv(equipment_path, encoding="utf-8-sig")

        # Conversion dates
        for col in ["WOWO_DECLARATION_DATE", "WOWO_END_DATE", "WOWO_CREATION_DATE"]:
            if col in self.df_fail.columns:
                self.df_fail[col] = pd.to_datetime(self.df_fail[col], errors="coerce")

        # Année + mois + durée (utiles pour l'EDA)
        self.df_fail["annee"] = self.df_fail["WOWO_DECLARATION_DATE"].dt.year
        self.df_fail["mois"]  = self.df_fail["WOWO_DECLARATION_DATE"].dt.month
        self.df_fail["duree"] = (
            self.df_fail["WOWO_END_DATE"] - self.df_fail["WOWO_DECLARATION_DATE"]
        ).dt.days

        self.phases_completed.append("load_raw_data")

        return {
            "failure_n_rows":   int(len(self.df_fail)),
            "failure_n_cols":   int(len(self.df_fail.columns)),
            "equipment_n_rows": int(len(self.df_equip)),
            "year_filter":      int(self.year),
            "date_min": self.df_fail["WOWO_DECLARATION_DATE"].min().isoformat()
                        if self.df_fail["WOWO_DECLARATION_DATE"].notna().any() else None,
            "date_max": self.df_fail["WOWO_DECLARATION_DATE"].max().isoformat()
                        if self.df_fail["WOWO_DECLARATION_DATE"].notna().any() else None,
            "n_failures_year":  int((self.df_fail["annee"] == self.year).sum()),
        }

    def compute_eda_raw(self) -> Dict:
        """
        Phase 1 : EDA sur les données brutes (failure1.csv).
        Reproduit les 6 SECTIONS du notebook (cell 6, 8, 10, 12, 14, 16) avec
        TOUS les graphes — pour parité jury.
        """
        if self.df_fail is None:
            raise RuntimeError("Charge d'abord les données avec load_raw_data()")

        df = self.df_fail
        df23 = df[df["annee"] == self.year].copy()
        df34 = df23[df23["WOWO_EQUIPMENT_LEVEL"].isin([3.0, 4.0])].copy()

        # Labels métier des colonnes (notebook cell 6)
        COLS_UTILES_LABELS = {
            "WOWO_DECLARATION_DATE": "Date de panne",
            "WOWO_END_DATE":         "Date fin réparation",
            "WOWO_EQUIPMENT":        "Code composant",
            "WOWO_EQUIPMENT_LEVEL":  "Niveau hiérarchique",
            "failure_parent_code":   "Code machine mère",
            "failure_parent_level":  "Niveau machine mère",
            "WOWO_JOB_CLASS":        "Type maintenance",
            "WOWO_TOTAL_COST":       "Coût OT",
        }

        # ── Section 1 : Qualité des données ─────────────────────
        cols_utiles = [
            "WOWO_DECLARATION_DATE", "WOWO_END_DATE", "WOWO_EQUIPMENT",
            "WOWO_EQUIPMENT_LEVEL", "failure_parent_code", "failure_parent_level",
            "WOWO_JOB_CLASS", "WOWO_TOTAL_COST",
        ]
        existing_cols = [c for c in cols_utiles if c in df.columns]
        missing = df[existing_cols].isnull().sum().astype(int).to_dict()
        missing_pct = (df[existing_cols].isnull().mean() * 100).round(1).to_dict()

        # ── Section 2 : Distribution par niveau hiérarchique ───
        niveau_dist = df23["WOWO_EQUIPMENT_LEVEL"].value_counts().sort_index().to_dict()
        niveau_dist = {str(k): int(v) for k, v in niveau_dist.items()}

        # ── Section 3 : Distribution temporelle ─────────────────
        pannes_par_mois = df34.groupby("mois").size().reindex(range(1, 13), fill_value=0)
        pannes_mensuel = [int(v) for v in pannes_par_mois.values]

        # Niveaux séparés
        nv3_mois = df23[df23["WOWO_EQUIPMENT_LEVEL"] == 3.0].groupby("mois").size().reindex(range(1, 13), fill_value=0)
        nv4_mois = df23[df23["WOWO_EQUIPMENT_LEVEL"] == 4.0].groupby("mois").size().reindex(range(1, 13), fill_value=0)

        # Pannes cumulées
        df34_sorted = df34.sort_values("WOWO_DECLARATION_DATE")
        df34_sorted = df34_sorted.copy()
        df34_sorted["cumul"] = range(1, len(df34_sorted) + 1)
        cumul_dates = df34_sorted["WOWO_DECLARATION_DATE"].dt.strftime("%Y-%m-%d").tolist()
        cumul_values = df34_sorted["cumul"].tolist()

        # ── Section 4 : Top composants ──────────────────────────
        top_composants = (df34["WOWO_EQUIPMENT"]
                          .value_counts()
                          .head(15)
                          .to_dict())
        top_composants = {str(k): int(v) for k, v in top_composants.items()}

        # ── Section 5 : Type maintenance + coût ─────────────────
        if "WOWO_JOB_CLASS" in df34.columns:
            job_class_dist = df34["WOWO_JOB_CLASS"].value_counts().to_dict()
            job_class_dist = {str(k): int(v) for k, v in job_class_dist.items()}
        else:
            job_class_dist = {}

        cout_stats = {}
        if "WOWO_TOTAL_COST" in df34.columns:
            cost_clean = pd.to_numeric(df34["WOWO_TOTAL_COST"], errors="coerce").dropna()
            if len(cost_clean) > 0:
                cout_stats = {
                    "mean":   float(cost_clean.mean()),
                    "median": float(cost_clean.median()),
                    "max":    float(cost_clean.max()),
                    "total":  float(cost_clean.sum()),
                    "count":  int(len(cost_clean)),
                }

        # ── Section 6 : Durée réparation ────────────────────────
        duree_stats = {}
        duree_clean = df34["duree"].dropna()
        if len(duree_clean) > 0:
            duree_stats = {
                "mean":   float(duree_clean.mean()),
                "median": float(duree_clean.median()),
                "max":    float(duree_clean.max()),
                "p95":    float(duree_clean.quantile(0.95)),
                "count":  int(len(duree_clean)),
            }

        # ── Section 2.a : Pie répartition niveaux TOUT équipement (year filter) ──
        niveau_dist_all = df23["WOWO_EQUIPMENT_LEVEL"].value_counts().sort_index().to_dict()
        niveau_dist_all = {str(k): int(v) for k, v in niveau_dist_all.items() if pd.notna(k)}
        niveau_na = int(df23["WOWO_EQUIPMENT_LEVEL"].isna().sum())

        # ── Section 3.d : Pannes par jour de semaine (niveaux 3+4) ────
        if not df34.empty:
            df34_ = df34.copy()
            df34_["jour_sem"] = df34_["WOWO_DECLARATION_DATE"].dt.dayofweek
            pannes_jour_sem = df34_.groupby("jour_sem").size().reindex(range(7), fill_value=0).astype(int).tolist()
        else:
            pannes_jour_sem = [0] * 7

        # ── Section 4 : Distribution par composant (histogramme + catégories) ──
        # ⚠️ INVARIANT À RESPECTER :
        #   composants_categories["1"] + ["2"] + ["3_4"] + ["5_plus"]
        #   ≡ df34["WOWO_EQUIPMENT"].nunique()  (= composants_uniques exposé plus bas)
        #   ≡ comp_modelisables (≥2) + composants_categories["1"]
        # On expose les 3 valeurs côte à côte → le frontend peut afficher
        # "Total contrôle = X" pour transparence jury.
        composants_categories = {"1": 0, "2": 0, "3_4": 0, "5_plus": 0}
        composants_hist = {"bins": [], "counts": []}
        composants_total_check = 0
        if not df34.empty:
            pannes_per_comp = df34.groupby("WOWO_EQUIPMENT").size()
            composants_categories = {
                "1":      int((pannes_per_comp == 1).sum()),
                "2":      int((pannes_per_comp == 2).sum()),
                "3_4":    int(((pannes_per_comp >= 3) & (pannes_per_comp <= 4)).sum()),
                "5_plus": int((pannes_per_comp >= 5).sum()),
            }
            composants_total_check = sum(composants_categories.values())
            counts_arr, bins_arr = np.histogram(pannes_per_comp.values, bins=20)
            composants_hist = {
                "bins":   [round(float(b), 2) for b in bins_arr.tolist()],
                "counts": [int(c) for c in counts_arr.tolist()],
            }
            pannes_per_comp_stats = {
                "min":    int(pannes_per_comp.min()),
                "max":    int(pannes_per_comp.max()),
                "mean":   float(pannes_per_comp.mean()),
                "median": float(pannes_per_comp.median()),
            }
        else:
            pannes_per_comp_stats = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

        # ── Section 5 : Coût par mois ──
        if not df34.empty and "WOWO_TOTAL_COST" in df34.columns:
            cost_clean34 = pd.to_numeric(df34["WOWO_TOTAL_COST"], errors="coerce").fillna(0)
            cout_par_mois = df34.assign(_cost=cost_clean34).groupby("mois")["_cost"].sum().reindex(range(1, 13), fill_value=0).astype(float).tolist()
            cost_total_year = float(cost_clean34.sum())
        else:
            cout_par_mois = [0.0] * 12
            cost_total_year = 0.0

        # ── Section 6 : Résumé pour le rapport PFE (cell 16) ──
        type_dom = max(job_class_dist.items(), key=lambda kv: kv[1])[0] if job_class_dist else "—"
        type_dom_n = job_class_dist.get(type_dom, 0)
        pannes_per_comp_for_resume = df34.groupby("WOWO_EQUIPMENT").size() if not df34.empty else pd.Series(dtype=int)
        comp_modelisables = int((pannes_per_comp_for_resume >= self.min_failures).sum())
        meme_jour_pct = float((df34["duree"] == 0).mean() * 100) if not df34.empty else 0.0
        mois_max_n = max(pannes_mensuel) if pannes_mensuel else 0
        mois_max_idx = int(np.argmax(pannes_mensuel)) if pannes_mensuel else 0
        mois_min_n = min(pannes_mensuel) if pannes_mensuel else 0
        mois_min_idx = int(np.argmin(pannes_mensuel)) if pannes_mensuel else 0

        # Composant avec le max de pannes (notebook cell 12 print line 7)
        composant_max_pannes = "—"
        composant_max_n      = 0
        if not df34.empty:
            comp_counts = df34.groupby("WOWO_EQUIPMENT").size()
            composant_max_pannes = str(comp_counts.idxmax())
            composant_max_n      = int(comp_counts.max())

        # % préventif (notebook cell 14 print line 3)
        preventif_n = int(job_class_dist.get("PREVEN", 0))
        preventif_pct = float((preventif_n / max(len(df34), 1)) * 100)

        self.phases_completed.append("compute_eda_raw")

        return {
            "overview": {
                "total_ot":            int(len(df)),
                "total_ot_year":       int(len(df23)),
                "total_ot_niveaux_34": int(len(df34)),
                "composants_uniques":  int(df34["WOWO_EQUIPMENT"].nunique()),
                "machines_meres":      int(df34["WOWO_SYSTEM_EQUIPMENT"].nunique()),
                "comp_modelisables":   comp_modelisables,
                "min_failures_seuil":  int(self.min_failures),
            },
            "cols_utiles_labels": COLS_UTILES_LABELS,  # 🆕 labels métier
            "quality": {
                "missing":     missing,
                "missing_pct": missing_pct,
                "total_rows":  int(len(df)),
            },
            # Section 2
            "niveau_distribution":     niveau_dist,
            "niveau_distribution_all": niveau_dist_all,  # 🆕
            "niveau_na":               niveau_na,         # 🆕
            # Section 3
            "pannes_mensuel":       pannes_mensuel,
            "pannes_mensuel_niv3":  [int(v) for v in nv3_mois.values],
            "pannes_mensuel_niv4":  [int(v) for v in nv4_mois.values],
            "pannes_jour_sem":      pannes_jour_sem,    # 🆕
            "pannes_cumulees": {
                "dates":  cumul_dates[:500],
                "values": cumul_values[:500],
            },
            # Section 4
            "top_composants":         top_composants,
            "composants_categories": composants_categories,  # 🆕
            "composants_total_check": int(composants_total_check),  # 🆕 invariant check
            "composants_hist":       composants_hist,         # 🆕
            "pannes_per_comp_stats": pannes_per_comp_stats,   # 🆕
            # Section 5
            "job_class_dist":   job_class_dist,
            "cout_stats":       cout_stats,
            "cout_par_mois":    cout_par_mois,                # 🆕
            "cout_total_year":  cost_total_year,              # 🆕
            "duree_stats":      duree_stats,
            "preventif_n":      preventif_n,                  # 🆕
            "preventif_pct":    round(preventif_pct, 1),      # 🆕
            "mois_min":         int(mois_min_idx + 1),        # 🆕
            "mois_min_n":       int(mois_min_n),              # 🆕
            "composant_max_pannes": composant_max_pannes,     # 🆕
            "composant_max_n":      composant_max_n,          # 🆕
            # Section 6 - Résumé PFE
            "resume_pfe": {                                   # 🆕
                "ot_year":            int(len(df23)),
                "ot_niveaux_34":      int(len(df34)),
                "composants_uniques": int(df34["WOWO_EQUIPMENT"].nunique()) if not df34.empty else 0,
                "comp_modelisables":  comp_modelisables,
                "machines_meres":     int(df34["WOWO_SYSTEM_EQUIPMENT"].nunique()) if not df34.empty else 0,
                "mois_max":           int(mois_max_idx + 1),
                "mois_max_n":         int(mois_max_n),
                "type_dominant":      type_dom,
                "type_dominant_n":    int(type_dom_n),
                "cout_total_MDA":     round(cost_total_year / 1e6, 2),
                "meme_jour_pct":      round(meme_jour_pct, 1),
            },
        }

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2 — FEATURE ENGINEERING (8 ÉTAPES)
    # ═══════════════════════════════════════════════════════════════
    def compute_features(self, progress_callback=None) -> Dict:
        """
        Phase 2 : Création des features dérivées (Dataset_V1).
        Suit fidèlement les 8 étapes du notebook (cellules 19-37).

        Args:
            progress_callback : fonction(step_id, step_label, data) appelée après
                chaque étape. Permet à l'UI de rendre progressivement.
                step_id ∈ {"filter_year","hierarchy","select_comps","lookup",
                            "panel","rul","rolling","export"}.
        """
        def _emit(step_id, step_label, data):
            if progress_callback is not None:
                try:
                    progress_callback(step_id, step_label, data)
                except Exception:
                    pass

        if self.df_fail is None or self.df_equip is None:
            raise RuntimeError("Charge d'abord les données avec load_raw_data()")

        df_fail  = self.df_fail
        df_equip = self.df_equip

        # ───────────────── ÉTAPE 2 : Filtrage année ─────────────────
        df_fail_year = df_fail[
            df_fail["WOWO_DECLARATION_DATE"].dt.year == self.year
        ].copy()

        # 🆕 Diagnostic : distribution des niveaux d'équipement (toutes années)
        level_dist_all = (
            df_fail["WOWO_EQUIPMENT_LEVEL"].value_counts().sort_index()
            if "WOWO_EQUIPMENT_LEVEL" in df_fail.columns else pd.Series(dtype=int)
        )
        level_dist_year = (
            df_fail_year["WOWO_EQUIPMENT_LEVEL"].value_counts().sort_index()
            if "WOWO_EQUIPMENT_LEVEL" in df_fail_year.columns else pd.Series(dtype=int)
        )
        n_ot_year_34 = int(
            df_fail_year[df_fail_year["WOWO_EQUIPMENT_LEVEL"].isin([3.0, 4.0])].shape[0]
        ) if "WOWO_EQUIPMENT_LEVEL" in df_fail_year.columns else 0

        _emit("filter_year", "Filtrage année", {
            "year":                     int(self.year),
            "n_rows_total":             int(len(df_fail)),
            "n_ot_year":                int(len(df_fail_year)),
            "n_ot_year_levels_34":      n_ot_year_34,
            "level_distribution_all":   {
                str(int(k)) if pd.notna(k) else "NaN": int(v)
                for k, v in level_dist_all.items()
            },
            "level_distribution_year":  {
                str(int(k)) if pd.notna(k) else "NaN": int(v)
                for k, v in level_dist_year.items()
            },
        })

        # ───────────── ÉTAPE 3 : Hiérarchie machineID ───────────────
        eq_dict = df_equip.set_index("EREQ_CODE")[
            ["EREQ_LEVEL", "EREQ_PARENT_EQUIPMENT", "EREQ_DESCRIPTION"]
        ].to_dict("index")

        def get_machineID(comp_code, comp_level, parent_code, parent_level):
            if comp_level in [1, 2]:
                return None, None
            if comp_level == 3:
                if pd.notna(parent_level) and int(parent_level) == 2:
                    desc = eq_dict.get(parent_code, {}).get("EREQ_DESCRIPTION", "")
                    return parent_code, desc
                return None, None
            if comp_level == 4:
                if pd.isna(parent_code) or parent_code not in eq_dict:
                    return None, None
                grandparent = eq_dict[parent_code].get("EREQ_PARENT_EQUIPMENT")
                if pd.isna(grandparent) or grandparent not in eq_dict:
                    return None, None
                if eq_dict[grandparent]["EREQ_LEVEL"] == 2:
                    desc = eq_dict[grandparent].get("EREQ_DESCRIPTION", "")
                    return grandparent, desc
                return None, None
            return None, None

        result = df_fail_year.apply(
            lambda r: pd.Series(get_machineID(
                r["WOWO_EQUIPMENT"], r["WOWO_EQUIPMENT_LEVEL"],
                r["failure_parent_code"], r["failure_parent_level"]
            )), axis=1
        )
        df_fail_year[["machineID", "machineID_desc"]] = result
        fail_ok = df_fail_year[df_fail_year["machineID"].notna()].copy()

        _emit("hierarchy", "Hiérarchie machineID", {
            "n_pannes_with_machine":    int(len(fail_ok)),
            "n_pannes_orphelines":      int(len(df_fail_year) - len(fail_ok)),
            "n_machines_meres":         int(fail_ok["machineID"].nunique()),
        })

        # ──────────── ÉTAPE 4 : Sélection des composants ────────────
        comp_counts_all = fail_ok.groupby("WOWO_EQUIPMENT").size()
        n_comps_unique_34 = int(len(comp_counts_all))
        comps_ok = comp_counts_all[comp_counts_all >= self.min_failures].index
        fail_ok = fail_ok[fail_ok["WOWO_EQUIPMENT"].isin(comps_ok)].copy()

        ref_comp = fail_ok.groupby("WOWO_EQUIPMENT").agg(
            machineID      = ("machineID", "first"),
            machineID_desc = ("machineID_desc", "first"),
            comp_desc      = ("WOWO_EQUIPMENT_DESCRIPTION", "first"),
            comp_level     = ("WOWO_EQUIPMENT_LEVEL", "first"),
            machine_root   = ("WOWO_SYSTEM_EQUIPMENT", "first"),
        ).reset_index().rename(columns={"WOWO_EQUIPMENT": "failure_comp"})

        all_comps = sorted(comps_ok)

        _emit("select_comps", "Sélection composants", {
            "n_composants_uniques_34":      n_comps_unique_34,
            "n_composants_modelisables":    int(len(all_comps)),
            "min_failures":                 int(self.min_failures),
            "n_machines_meres_modelisees":  int(fail_ok["machineID"].nunique()),
            "n_ot_apres_min_failures":      int(len(fail_ok)),
        })

        # ──────────── ÉTAPE 5 : Lookup maintenance V1 ───────────────
        fail_ok["maintenance_date"] = fail_ok["WOWO_END_DATE"]
        fail_ok["duree_reparation"] = (
            fail_ok["maintenance_date"] - fail_ok["WOWO_DECLARATION_DATE"]
        ).dt.days

        fail_lookup, maint_lookup_v1 = {}, {}
        for _, row in fail_ok.iterrows():
            comp = row["WOWO_EQUIPMENT"]
            fail_date = row["WOWO_DECLARATION_DATE"].date()
            maint_date = row["maintenance_date"].date()
            job_class = row.get("WOWO_JOB_CLASS", None)

            if comp not in fail_lookup:
                fail_lookup[comp] = {}
            fail_lookup[comp][fail_date] = job_class

            if comp not in maint_lookup_v1:
                maint_lookup_v1[comp] = set()
            maint_lookup_v1[comp].add(maint_date)

        _emit("lookup", "Lookup maintenance", {
            "n_composants_indexes":   int(len(fail_lookup)),
            "n_dates_panne_distinct": int(sum(len(v) for v in fail_lookup.values())),
            "n_dates_maint_distinct": int(sum(len(v) for v in maint_lookup_v1.values())),
        })

        # ──────────── ÉTAPE 6 : Panel composant × jour ──────────────
        timeline = pd.date_range(
            start=f"{self.year}-01-01",
            end  =f"{self.year}-12-31",
            freq ="D"
        )

        rows = []
        for comp in all_comps:
            r = ref_comp.set_index("failure_comp").loc[comp]
            fail_dates_c = fail_lookup.get(comp, {})
            maint_dates_c = maint_lookup_v1.get(comp, set())

            for day in timeline:
                d = day.date()
                rows.append({
                    "date":            day,
                    "machine_root":    r["machine_root"],
                    "machineID":       r["machineID"],
                    "machineID_desc":  r["machineID_desc"],
                    "failure_comp":    comp,
                    "comp_level":      r["comp_level"],
                    "comp_desc":       r["comp_desc"],
                    "failure":         1 if d in fail_dates_c else 0,
                    "WOWO_JOB_CLASS":  fail_dates_c.get(d, None),
                    "maintenance":    1 if d in maint_dates_c else 0,
                })

        df_panel = pd.DataFrame(rows)
        df_panel = df_panel.sort_values(["failure_comp", "date"]).reset_index(drop=True)
        self.df_panel = df_panel

        _emit("panel", "Panel composant × jour", {
            "n_rows_panel":           int(len(df_panel)),
            "n_jours":                int(len(timeline)),
            "n_composants":           int(len(all_comps)),
            "n_failures_in_panel":    int(df_panel["failure"].sum()),
            "n_maintenances_in_panel": int(df_panel["maintenance"].sum()),
            "preview_panel":          df_panel.head(5).to_dict(orient="records"),
        })

        # ──────────── ÉTAPE 7 : Calcul du RUL V1 ────────────────────
        def calculate_rul_v1(df_comp):
            df_comp = df_comp.sort_values("date").copy().reset_index(drop=True)
            failure_dates = df_comp[df_comp["failure"] == 1]["date"].values
            maint_dates_arr = df_comp[df_comp["maintenance"] == 1]["date"].values
            if len(failure_dates) == 0:
                return pd.DataFrame()

            end_of_year = pd.Timestamp(f"{self.year}-12-31")
            ruls = []
            for _, row in df_comp.iterrows():
                d = row["date"]
                past_fails  = failure_dates[failure_dates <= d]
                past_maints = maint_dates_arr[maint_dates_arr <= d]

                if len(past_fails) > 0:
                    last_fail  = past_fails[-1]
                    last_maint = past_maints[-1] if len(past_maints) > 0 else None
                    if last_maint is None or last_fail > last_maint:
                        ruls.append(0)
                        continue

                future_fails = failure_dates[failure_dates > d]
                if len(future_fails) > 0:
                    rul = int((pd.Timestamp(future_fails[0]) - d).days)
                else:
                    rul = (end_of_year - d).days
                ruls.append(rul)

            df_comp["RUL"] = ruls
            return df_comp

        # ⚠️ Bugfix pandas 2.2+ : `groupby.apply` retire désormais la colonne
        # de groupage du DataFrame passé à `func`. Pour rester compatible
        # avec toutes versions, on fait la boucle manuellement et on restaure
        # `failure_comp` si la fonction ne l'a pas conservée.
        _dfs_rul = []
        for _comp_name, _df_comp in df_panel.groupby("failure_comp", sort=False):
            _df_comp = _df_comp.copy()
            if "failure_comp" not in _df_comp.columns:
                _df_comp["failure_comp"] = _comp_name
            _res = calculate_rul_v1(_df_comp)
            if not isinstance(_res, pd.DataFrame) or _res.empty:
                continue
            if "failure_comp" not in _res.columns:
                _res["failure_comp"] = _comp_name
            _dfs_rul.append(_res)
        df_rul = (pd.concat(_dfs_rul, ignore_index=True)
                  if _dfs_rul else df_panel.iloc[0:0].copy())
        self.df_rul = df_rul

        # 🆕 Diagnostic RUL : nombre de pannes / seuils (en valeurs, pas qu'en %)
        rul_n_le_5  = int((df_rul["RUL"] <= 5).sum())  if "RUL" in df_rul.columns else 0
        rul_n_le_10 = int((df_rul["RUL"] <= 10).sum()) if "RUL" in df_rul.columns else 0
        rul_n_le_30 = int((df_rul["RUL"] <= 30).sum()) if "RUL" in df_rul.columns else 0
        rul_n_le_90 = int((df_rul["RUL"] <= 90).sum()) if "RUL" in df_rul.columns else 0

        _emit("rul", "Calcul du RUL V1", {
            "n_rows":     int(len(df_rul)),
            "rul_min":    int(df_rul["RUL"].min())   if not df_rul.empty else None,
            "rul_max":    int(df_rul["RUL"].max())   if not df_rul.empty else None,
            "rul_mean":   float(df_rul["RUL"].mean()) if not df_rul.empty else None,
            "rul_median": float(df_rul["RUL"].median()) if not df_rul.empty else None,
            # 🆕 Comptes par seuil (l'utilisateur ne veut pas que des %)
            "n_rul_le_5":  rul_n_le_5,
            "n_rul_le_10": rul_n_le_10,
            "n_rul_le_30": rul_n_le_30,
            "n_rul_le_90": rul_n_le_90,
        })

        # ──────────── ÉTAPE 8 : Feature Engineering ─────────────────
        def add_features(df_comp):
            df_comp = df_comp.sort_values("date").copy().reset_index(drop=True)

            # 1. Fenêtres roulantes
            for w in [7, 30, 90]:
                df_comp[f"pannes_{w}j"] = (
                    df_comp["failure"].shift(1).rolling(w, min_periods=0).sum().fillna(0).astype(int)
                )
                df_comp[f"maint_{w}j"] = (
                    df_comp["maintenance"].shift(1).rolling(w, min_periods=0).sum().fillna(0).astype(int)
                )

            # 2. DSLF (Days Since Last Failure)
            last_fail = pd.Timestamp(f"{self.year - 1}-12-31")
            dslf = []
            for _, row in df_comp.iterrows():
                if row["failure"] == 1:
                    last_fail = row["date"]
                dslf.append((row["date"] - last_fail).days)
            df_comp["DSLF"] = dslf

            # 3. DSLM (Days Since Last Maintenance)
            last_maint = pd.Timestamp(f"{self.year - 1}-12-31")
            dslm = []
            for _, row in df_comp.iterrows():
                if row["maintenance"] == 1:
                    last_maint = row["date"]
                dslm.append((row["date"] - last_maint).days)
            df_comp["DSLM"] = dslm

            # 4. MTBF rolling (moyenne sur 3 dernières pannes)
            mtbf_col, gap_list, last_fail_date, current_mtbf = [], [], None, np.nan
            for _, row in df_comp.iterrows():
                if row["failure"] == 1:
                    if last_fail_date is not None:
                        gap = (row["date"] - last_fail_date).days
                        gap_list.append(gap)
                        current_mtbf = round(sum(gap_list[-3:]) / len(gap_list[-3:]), 1)
                    last_fail_date = row["date"]
                mtbf_col.append(current_mtbf)
            df_comp["MTBF_rolling"] = mtbf_col
            df_comp["has_mtbf"] = df_comp["MTBF_rolling"].notna().astype(int)
            df_comp["MTBF_rolling"] = df_comp["MTBF_rolling"].fillna(0)

            # 5. Saisonnalité (encodage cyclique du mois)
            df_comp["month"] = df_comp["date"].dt.month
            df_comp["month_sin"] = np.sin(2 * np.pi * df_comp["month"] / 12)
            df_comp["month_cos"] = np.cos(2 * np.pi * df_comp["month"] / 12)

            # 6. Ratio DSLF / MTBF (feature dérivée)
            df_comp["dslf_mtbf_ratio"] = df_comp["DSLF"] / (df_comp["MTBF_rolling"] + 1)

            return df_comp

        # ⚠️ Bugfix pandas 2.2+ — même problème qu'à l'étape précédente.
        _dfs_final = []
        for _comp_name, _df_comp in df_rul.groupby("failure_comp", sort=False):
            _df_comp = _df_comp.copy()
            if "failure_comp" not in _df_comp.columns:
                _df_comp["failure_comp"] = _comp_name
            _res = add_features(_df_comp)
            if not isinstance(_res, pd.DataFrame) or _res.empty:
                continue
            if "failure_comp" not in _res.columns:
                _res["failure_comp"] = _comp_name
            _dfs_final.append(_res)
        df_final = (pd.concat(_dfs_final, ignore_index=True)
                    if _dfs_final else df_rul.iloc[0:0].copy())

        _emit("rolling", "Feature Engineering (rolling, DSLF, DSLM, …)", {
            "n_rows":         int(len(df_final)),
            "feature_cols":   list(self.FEATURE_COLS),
            "rolling_windows": [7, 30, 90],
            "extra_features": ["DSLF", "DSLM", "MTBF_rolling",
                                "month_sin", "month_cos", "dslf_mtbf_ratio"],
        })

        # ───── Numérotation et niveaux (pour visualisation) ─────────
        machine_ids = sorted(df_final["machineID"].unique())
        df_final["machineID_num"] = df_final["machineID"].map({m: i+1 for i, m in enumerate(machine_ids)})
        comp_ids = sorted(df_final["failure_comp"].unique())
        df_final["comp_num"] = df_final["failure_comp"].map({c: i+1 for i, c in enumerate(comp_ids)})
        df_final["machineID_level"] = 2

        self.df_final = df_final
        self.df_export = df_final[self.EXPORT_COLS].copy()

        self.phases_completed.append("compute_features")

        _emit("export", "Finalisation Dataset_V1", {
            "n_rows":         int(len(self.df_export)),
            "n_cols":         int(self.df_export.shape[1]),
            "n_failures":     int(self.df_export["failure"].sum()),
            "n_maintenances": int(self.df_export["maintenance"].sum()),
            "n_composants":   int(self.df_export["failure_comp"].nunique()),
            "n_machines":     int(self.df_export["machineID"].nunique()),
        })

        return {
            "n_rows":          int(len(self.df_export)),
            "n_failures":      int(self.df_export["failure"].sum()),
            "n_maintenances":  int(self.df_export["maintenance"].sum()),
            "n_composants":    int(self.df_export["failure_comp"].nunique()),
            "n_machines":      int(self.df_export["machineID"].nunique()),
            "n_features":      len(self.FEATURE_COLS),
            "feature_cols":    self.FEATURE_COLS,
            "period_start":    self.df_export["date"].min().isoformat(),
            "period_end":      self.df_export["date"].max().isoformat(),
            "rul_stats": {
                "min":    int(self.df_export["RUL"].min()),
                "max":    int(self.df_export["RUL"].max()),
                "mean":   float(self.df_export["RUL"].mean()),
                "median": float(self.df_export["RUL"].median()),
            },
            # 🆕 Comptes par seuil RUL (le user veut les VALEURS pas que les %)
            "n_rul_le_5":     rul_n_le_5,
            "n_rul_le_10":    rul_n_le_10,
            "n_rul_le_30":    rul_n_le_30,
            "n_rul_le_90":    rul_n_le_90,
            # 🆕 Stats de filtrage (le user veut les voir affichées)
            "year":                       int(self.year),
            "n_ot_year":                  int(len(df_fail_year)),
            "n_ot_year_levels_34":        n_ot_year_34,
            "n_composants_uniques_34":    n_comps_unique_34,
            "n_composants_modelisables":  int(len(all_comps)),
            "n_machines_meres":           int(df_final["machineID"].nunique()),
            "min_failures":               int(self.min_failures),
            "level_distribution_all":     {
                str(int(k)) if pd.notna(k) else "NaN": int(v)
                for k, v in level_dist_all.items()
            },
            "level_distribution_year":    {
                str(int(k)) if pd.notna(k) else "NaN": int(v)
                for k, v in level_dist_year.items()
            },
            # Aperçu pour affichage frontend (premiers + lignes panne)
            "preview_panel":   self.df_panel.head(5).to_dict(orient="records"),
            "preview_final":   self.df_export.head(10).to_dict(orient="records"),
        }

    def export_dataset_v1(self, output_path: str) -> str:
        """Sauvegarde le Dataset_V1 final en CSV téléchargeable."""
        if self.df_export is None:
            raise RuntimeError("Lance compute_features() d'abord")
        self.df_export.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")
        return output_path

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3 — EDA SUR LES FEATURES CRÉÉES
    # ═══════════════════════════════════════════════════════════════
    def compute_eda_features(self) -> Dict:
        """
        Phase 3 : Stats descriptives + distribution RUL + corrélations.
        Reproduit les analyses du notebook (sections 3.1-3.3).
        """
        if self.df_export is None:
            raise RuntimeError("Lance compute_features() d'abord")

        df = self.df_export

        # ── 3.1 Aperçu général ─────────────────────────────────
        overview = {
            "n_rows":     int(len(df)),
            "n_cols":     int(df.shape[1]),
            "n_comp":     int(df["failure_comp"].nunique()),
            "n_pannes":   int(df["failure"].sum()),
            "rul_zero":   int((df["RUL"] == 0).sum()),
            "rul_zero_pct": float((df["RUL"] == 0).mean() * 100),
            "period_start": df["date"].min().isoformat(),
            "period_end":   df["date"].max().isoformat(),
        }

        # Stats numériques
        num_cols = ["DSLF", "DSLM", "MTBF_rolling", "has_mtbf",
                    "month_sin", "month_cos", "dslf_mtbf_ratio", "RUL"]
        stats = df[num_cols].describe().T.round(3).to_dict(orient="index")
        # Sérialiser les Timestamp
        for col, st in stats.items():
            stats[col] = {k: float(v) for k, v in st.items()}

        # ── 3.2 Distribution du RUL ────────────────────────────
        rul_pos = df[df["RUL"] > 0]["RUL"]
        rul_hist, rul_bins = np.histogram(rul_pos, bins=50)

        # ECDF
        sorted_rul = np.sort(rul_pos.values)
        ecdf = (np.arange(1, len(sorted_rul) + 1) / len(sorted_rul)) * 100

        # Alerte / Sain
        alert_pct = float((df["RUL"] <= self.alert_days).mean() * 100)

        # ── 3.3 Corrélations features avec RUL (15 features du notebook) ──
        feature_cols_full = ["failure", "maintenance",
                             "pannes_7j", "pannes_30j", "pannes_90j",
                             "maint_7j", "maint_30j", "maint_90j",
                             "DSLF", "DSLM", "MTBF_rolling", "has_mtbf",
                             "month_sin", "month_cos", "dslf_mtbf_ratio"]
        # Garder uniquement les colonnes présentes dans df_export
        feature_cols_full = [c for c in feature_cols_full if c in df.columns]
        corr_matrix = df[feature_cols_full + ["RUL"]].corr().round(3)
        corr_with_rul = corr_matrix["RUL"].drop("RUL").to_dict()
        corr_with_rul = {k: float(v) for k, v in corr_with_rul.items()}

        # Matrice complète sérialisée + ordre conservé
        corr_full = {
            row: {col: float(corr_matrix.loc[row, col]) for col in corr_matrix.columns}
            for row in corr_matrix.index
        }
        corr_features_order = list(corr_matrix.columns)  # 🆕 ordre des lignes/colonnes

        # ── 🆕 Distribution RUL par catégorie (notebook cell 42) ──
        rul_categories = [
            {"label": "RUL = 0 (réparation)",
             "n": int((df["RUL"] == 0).sum()),
             "pct": round(float((df["RUL"] == 0).mean() * 100), 1),
             "color": "error"},
            {"label": f"RUL 1-{self.alert_days}j (alerte)",
             "n": int(((df["RUL"] > 0) & (df["RUL"] <= self.alert_days)).sum()),
             "pct": round(float(((df["RUL"] > 0) & (df["RUL"] <= self.alert_days)).mean() * 100), 1),
             "color": "warning"},
            {"label": f"RUL {self.alert_days + 1}-90j",
             "n": int(((df["RUL"] > self.alert_days) & (df["RUL"] <= 90)).sum()),
             "pct": round(float(((df["RUL"] > self.alert_days) & (df["RUL"] <= 90)).mean() * 100), 1),
             "color": "info"},
            {"label": "RUL > 90j",
             "n": int((df["RUL"] > 90).sum()),
             "pct": round(float((df["RUL"] > 90).mean() * 100), 1),
             "color": "success"},
        ]

        self.phases_completed.append("compute_eda_features")

        return {
            "overview":     overview,
            "stats":        stats,
            "rul_distribution": {
                "bins":   rul_bins.tolist(),
                "counts": rul_hist.tolist(),
                "mean":   float(rul_pos.mean()),
                "median": float(rul_pos.median()),
            },
            "ecdf": {
                "rul":  sorted_rul[::max(1, len(sorted_rul)//500)].tolist(),  # sample
                "pct":  ecdf[::max(1, len(ecdf)//500)].tolist(),
            },
            "alert_balance": {
                "alert_pct":    alert_pct,
                "healthy_pct":  100 - alert_pct,
                "threshold":    self.alert_days,
            },
            "corr_with_rul":      corr_with_rul,
            "corr_matrix":        corr_full,
            "corr_features_order": corr_features_order,  # 🆕
            "rul_categories":      rul_categories,        # 🆕
        }

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4 — PRÉTRAITEMENT LSTM/GRU (avec current_max_rul DYNAMIQUE)
    # ═══════════════════════════════════════════════════════════════
    def prepare_sequences(
        self,
        lookback:        int   = 30,
        current_max_rul: int   = 30,
        test_ratio:      float = 0.20,
        weight_factor:   float = 15.0,
        healthy_sample_frac: float = 0.30,
        progress_callback=None,
    ) -> Dict:
        """
        Phase 4 : Prétraitement LSTM/GRU complet (cellules 47-53).

        Args:
            lookback        : taille de la fenêtre temporelle (configurable UI)
            current_max_rul : plafond du RUL — pilotable depuis l'UI (NOUVEAU)
            test_ratio      : proportion test (0.20 par défaut)
            weight_factor   : amplification du poids sur les RUL faibles (×15 par défaut)
            healthy_sample_frac : proportion d'échantillons sains gardés (0.30 par défaut)
            progress_callback : fonction(step_id, step_label, data) appelée après
                chaque étape. Permet à l'UI de rendre progressivement.
                step_id ∈ {"config", "balance", "split", "normalize", "sequence", "weights"}.
        """
        def _emit(step_id, step_label, data):
            if progress_callback is not None:
                try:
                    progress_callback(step_id, step_label, data)
                except Exception:
                    # On ne casse JAMAIS le pipeline pour un souci d'UI
                    pass

        if self.df_export is None:
            raise RuntimeError("Lance compute_features() d'abord")

        df = self.df_export.copy()
        df = df.sort_values(["failure_comp", "date"]).reset_index(drop=True)

        # ── Application de current_max_rul (NOUVEAU — pilotable UI) ──
        self.current_max_rul = current_max_rul
        self.lookback = lookback
        df["RUL"] = df["RUL"].apply(lambda x: min(x, current_max_rul))

        _emit("config", "Configuration appliquée", {
            "lookback":        int(lookback),
            "current_max_rul": int(current_max_rul),
            "weight_factor":   float(weight_factor),
            "test_ratio":      float(test_ratio),
            "healthy_sample_frac": float(healthy_sample_frac),
            "n_rows_initial":  int(len(df)),
            "features":        list(self.FEATURE_COLS),
            "target":          self.TARGET_COL,
        })

        # ── Équilibrage sain/dégradé ────────────────────────────────
        mask_sain = df["RUL"] >= current_max_rul
        n_sain_avant = int(mask_sain.sum())
        n_degrad     = int((~mask_sain).sum())
        df_degrad = df[~mask_sain]
        df_sain_reduit = df[mask_sain].sample(frac=healthy_sample_frac, random_state=self.random_state)
        df = pd.concat([df_degrad, df_sain_reduit]).sort_values(["failure_comp", "date"]).reset_index(drop=True)

        # Indexation composants (pour embedding)
        df["comp_idx"] = df["failure_comp"].astype("category").cat.codes
        self.num_classes_comp = int(df["comp_idx"].nunique())

        # ── 🆕 Phase 5 : tracker nom_composant → idx_embedding ─────
        # Sauvegardé tel quel dans `comp_mapping.json` lors de l'export ZIP,
        # pour permettre une réutilisation du modèle hors plateforme.
        # Logique métier inchangée — c'est juste un attribut auxiliaire.
        self._comp_name_to_idx = {
            str(name): int(idx)
            for name, idx in zip(df[self.COMP_COL].astype(str), df["comp_idx"].astype(int))
        }

        _emit("balance", "Équilibrage sain/dégradé", {
            "n_sain_avant":         n_sain_avant,
            "n_degrad":             n_degrad,
            "n_sain_apres":         int(len(df_sain_reduit)),
            "n_rows_after_balance": int(len(df)),
            "num_classes_comp":     int(self.num_classes_comp),
            "healthy_sample_frac":  float(healthy_sample_frac),
        })

        # ── Split par composant ─────────────────────────────────────
        all_components = df[self.COMP_COL].unique()
        np.random.seed(self.random_state)
        np.random.shuffle(all_components)
        train_size = int(len(all_components) * (1 - test_ratio))
        train_comps, test_comps = all_components[:train_size], all_components[train_size:]

        df_train = df[df[self.COMP_COL].isin(train_comps)].reset_index(drop=True)
        df_test  = df[df[self.COMP_COL].isin(test_comps)].reset_index(drop=True)

        _emit("split", "Split par composant", {
            "n_train_comps":              int(len(train_comps)),
            "n_test_comps":               int(len(test_comps)),
            "n_train_rows_after_balance": int(len(df_train)),
            "n_test_rows_after_balance":  int(len(df_test)),
            "test_ratio":                 float(test_ratio),
        })

        # ── Normalisation (MinMaxScaler) ────────────────────────────
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X_train_s = self.scaler_x.fit_transform(df_train[self.FEATURE_COLS])
        X_test_s  = self.scaler_x.transform(df_test[self.FEATURE_COLS])
        y_train_s = self.scaler_y.fit_transform(df_train[[self.TARGET_COL]])
        y_test_s  = self.scaler_y.transform(df_test[[self.TARGET_COL]])

        # Aperçus raw + normalisé pour Section 4
        preview_X_raw_n = []
        preview_y_raw_n = []
        preview_X_norm  = []
        if not df_train.empty:
            preview_X_raw_n = df_train[self.FEATURE_COLS].head(5).round(3).to_dict("records")
            preview_y_raw_n = df_train[self.TARGET_COL].head(5).astype(int).tolist()
            preview_X_norm  = X_train_s[:5].round(4).tolist()

        _emit("normalize", "Normalisation MinMax", {
            "features":              list(self.FEATURE_COLS),
            "preview_X_raw":         preview_X_raw_n,
            "preview_y_raw":         preview_y_raw_n,
            "preview_normalized_X":  preview_X_norm,
        })

        # ── Séquençage avec poids renforcés ─────────────────────────
        def create_sequences_weighted(X_s, y_s, df_meta, lb, wf):
            X_num, X_comp, ys, weights = [], [], [], []
            for comp in df_meta[self.COMP_COL].unique():
                mask = df_meta[self.COMP_COL].values == comp
                X_c, y_c = X_s[mask], y_s[mask]
                if mask.sum() <= lb:
                    continue
                c_idx = df_meta.loc[mask, "comp_idx"].values[0]
                for i in range(len(X_c) - lb):
                    X_num.append(X_c[i:i+lb])
                    X_comp.append(c_idx)
                    val_y = y_c[i+lb][0]
                    ys.append(val_y)
                    # Poids = 1 + (1 - y_normalisé) × weight_factor
                    weights.append(1.0 + (1.0 - val_y) * wf)
            return np.array(X_num), np.array(X_comp), np.array(ys), np.array(weights)

        Xn_tr, Xc_tr, ytr, wtr = create_sequences_weighted(X_train_s, y_train_s, df_train, lookback, weight_factor)
        Xn_te, Xc_te, yte, wte = create_sequences_weighted(X_test_s,  y_test_s,  df_test,  lookback, weight_factor)

        # ── Sauvegarde des tenseurs ─────────────────────────────────
        self.X_train_num,  self.X_train_comp = Xn_tr, Xc_tr
        self.X_test_num,   self.X_test_comp  = Xn_te, Xc_te
        self.y_train, self.y_test = ytr, yte
        self.w_train, self.w_test = wtr, wte
        # Garder les df de test pour pouvoir construire le tableau des dates de pannes
        self._df_test = df_test
        self.is_ready = True

        self.phases_completed.append("prepare_sequences")

        # 🆕 Séquence COMPLÈTE (lookback rows) — pour la viz "tableau par t"
        sequence_full_raw_s  = []
        sequence_full_norm_s = []
        sequence_meta_s      = {}
        if len(df_train) >= lookback and self.scaler_x is not None:
            seq_raw_df = df_train[self.FEATURE_COLS].iloc[:lookback].copy()
            sequence_full_raw_s = seq_raw_df.round(3).to_dict("records")
            sequence_full_norm_s = X_train_s[:lookback].round(4).tolist()
            comp_first = str(df_train[self.COMP_COL].iloc[0])
            sequence_meta_s = {
                "comp":          comp_first,
                "lookback":      int(lookback),
                "n_features":    len(self.FEATURE_COLS),
                "date_start":    str(df_train["date"].iloc[0])            if "date" in df_train.columns else None,
                "date_end":      str(df_train["date"].iloc[lookback - 1]) if "date" in df_train.columns and len(df_train) > lookback - 1 else None,
                "date_target":   str(df_train["date"].iloc[lookback])     if "date" in df_train.columns and len(df_train) > lookback else None,
                "y_target_raw":  int(df_train[self.TARGET_COL].iloc[lookback]) if len(df_train) > lookback else None,
                "y_target_norm": float(y_train_s[lookback][0])            if len(y_train_s) > lookback else None,
            }

        _emit("sequence", "Séquençage temporel", {
            "X_train_num_shape":  list(Xn_tr.shape),
            "X_train_comp_shape": list(Xc_tr.shape),
            "X_test_num_shape":   list(Xn_te.shape),
            "X_test_comp_shape":  list(Xc_te.shape),
            "y_train_shape":      list(ytr.shape),
            "y_test_shape":       list(yte.shape),
            "num_classes_comp":   int(self.num_classes_comp),
            "lookback":           int(lookback),
            "features":           list(self.FEATURE_COLS),
            "sequence_full_raw":  sequence_full_raw_s,
            "sequence_full_norm": sequence_full_norm_s,
            "sequence_meta":      sequence_meta_s,
        })

        # ── Diagnostics (formule inchangée — read-only) ──────────────
        if len(ytr) > 0 and self.scaler_y is not None:
            raw_rul_train = self.scaler_y.inverse_transform(
                ytr.reshape(-1, 1)
            ).flatten()
            n_weights_15 = int(np.isclose(wtr, 15.0, atol=1e-6).sum())
            n_weights_1  = int(np.isclose(wtr, 1.0,  atol=1e-6).sum())
            mask_crit    = raw_rul_train <= 5.0
            preview_weights_critical = wtr[mask_crit][:10].tolist() if mask_crit.any() else []

            # 🆕 Stats globales pour visualisation jury
            weight_stats = {
                "min":    float(wtr.min()),
                "max":    float(wtr.max()),
                "mean":   float(wtr.mean()),
                "std":    float(wtr.std()),
                "n_total": int(len(wtr)),
                "n_above_2":  int((wtr > 2.0).sum()),
                "n_above_8":  int((wtr > 8.0).sum()),
                "n_above_14": int((wtr > 14.0).sum()),
            }
            # Histogramme des poids — 12 bins
            hist_counts, hist_bins = np.histogram(
                wtr, bins=12, range=(1.0, max(2.0, float(wtr.max())))
            )
            weight_histogram = {
                "bins":   [round(float(b), 2) for b in hist_bins.tolist()],
                "counts": [int(c) for c in hist_counts.tolist()],
            }
        else:
            n_weights_15 = 0
            n_weights_1  = 0
            preview_weights_critical = []
            weight_stats = None
            weight_histogram = None

        # 🆕 Aperçu RAW (avant normalisation) — 5 premières lignes de df_train
        preview_X_raw = []
        preview_y_raw = []
        if not df_train.empty:
            preview_X_raw = (
                df_train[self.FEATURE_COLS].head(5).round(3).to_dict("records")
            )
            preview_y_raw = df_train[self.TARGET_COL].head(5).astype(int).tolist()

        # 🆕 Séquence COMPLÈTE (lookback rows) — pour la viz "tableau par t"
        # On prend la première séquence du training set : rows 0..lookback-1
        sequence_full_raw = []
        sequence_full_norm = []
        sequence_meta = {}
        if len(df_train) >= lookback and self.scaler_x is not None:
            seq_raw_df = df_train[self.FEATURE_COLS].iloc[:lookback].copy()
            sequence_full_raw = seq_raw_df.round(3).to_dict("records")
            sequence_full_norm = X_train_s[:lookback].round(4).tolist()
            # méta : composant + date + RUL cible (la valeur juste après les lookback rows)
            comp_first = str(df_train[self.COMP_COL].iloc[0])
            sequence_meta = {
                "comp":         comp_first,
                "lookback":     int(lookback),
                "n_features":   len(self.FEATURE_COLS),
                "date_start":   str(df_train["date"].iloc[0])           if "date" in df_train.columns else None,
                "date_end":     str(df_train["date"].iloc[lookback - 1]) if "date" in df_train.columns and len(df_train) > lookback - 1 else None,
                "date_target":  str(df_train["date"].iloc[lookback])     if "date" in df_train.columns and len(df_train) > lookback else None,
                "y_target_raw": int(df_train[self.TARGET_COL].iloc[lookback]) if len(df_train) > lookback else None,
                "y_target_norm": float(y_train_s[lookback][0]) if len(y_train_s) > lookback else None,
            }

        _emit("weights", "Poids d'entraînement", {
            "weight_factor":            float(weight_factor),
            "weight_stats":             weight_stats,
            "weight_histogram":         weight_histogram,
            "preview_weights_critical": preview_weights_critical,
            "n_weights_15":             n_weights_15,
            "n_weights_1":              n_weights_1,
        })

        return {
            "lookback":        int(lookback),
            "current_max_rul": int(current_max_rul),
            "weight_factor":   float(weight_factor),
            "num_classes_comp": int(self.num_classes_comp),
            "X_train_num_shape":  list(Xn_tr.shape),
            "X_train_comp_shape": list(Xc_tr.shape),
            "X_test_num_shape":   list(Xn_te.shape),
            "X_test_comp_shape":  list(Xc_te.shape),
            "y_train_shape":  list(ytr.shape),
            "y_test_shape":   list(yte.shape),
            "n_train_comps":  int(len(train_comps)),
            "n_test_comps":   int(len(test_comps)),
            "features":       self.FEATURE_COLS,
            "target":         self.TARGET_COL,
            # Aperçus pour visualisation UI
            "preview_normalized_X": Xn_tr[0, :5, :].tolist() if len(Xn_tr) else [],
            "preview_weights":      wtr[:10].tolist() if len(wtr) else [],
            # ─── Diagnostics demandés ─────────────────────────────
            "n_weights_15":           n_weights_15,
            "n_weights_1":            n_weights_1,
            "preview_weights_critical": preview_weights_critical,
            # 🆕 Analytics pour visualisation pédagogique jury
            "weight_stats":     weight_stats,
            "weight_histogram": weight_histogram,
            "preview_X_raw":    preview_X_raw,
            "preview_y_raw":    preview_y_raw,
            "n_train_rows_after_balance": int(len(df_train)),
            "n_test_rows_after_balance":  int(len(df_test)),
            # 🆕 Séquence complète (toutes les valeurs t=0..lookback-1) pour la viz
            "sequence_full_raw":  sequence_full_raw,    # list of dicts (lookback rows)
            "sequence_full_norm": sequence_full_norm,   # 2D list (lookback × n_features)
            "sequence_meta":      sequence_meta,
        }

    # ═══════════════════════════════════════════════════════════════
    # POST-TRAITEMENT — Prédiction sécurisée avec clip dynamique
    # ═══════════════════════════════════════════════════════════════
    def predict_with_safety(
        self,
        model,
        X_num: np.ndarray,
        X_comp: np.ndarray,
        current_max_rul: Optional[int] = None,
    ) -> np.ndarray:
        """
        Prédit et applique un clip dynamique [0, current_max_rul].
        Si current_max_rul n'est pas fourni, utilise self.current_max_rul.
        Retourne y_pred_days (valeurs en jours, dénormalisées et clippées).
        """
        if self.scaler_y is None:
            raise RuntimeError("Pipeline non prêt — lance prepare_sequences() d'abord")
        max_rul = current_max_rul if current_max_rul is not None else self.current_max_rul
        raw = model.predict([X_num, X_comp])
        prediction_inverse = self.scaler_y.inverse_transform(raw).flatten()
        prediction_finale = np.clip(prediction_inverse, 0, max_rul)
        return prediction_finale

    # ═══════════════════════════════════════════════════════════════
    # FUSION TEMPORELLE — Pour le réentrainement par l'admin
    # ═══════════════════════════════════════════════════════════════
    def merge_new_data(self, new_failure_csv: str) -> Dict:
        """
        Fusionne un nouveau fichier failure avec les données existantes,
        sur l'axe temporel + par composant.

        Logique :
        - Le nouveau fichier doit suivre la même structure que failure1.csv
        - Pour chaque composant existant : concaténation temporelle
        - Pour les nouveaux composants : intégration dans le panel
        - Recalcul automatique du RUL et des features

        Args:
            new_failure_csv : chemin vers le nouveau CSV failure
        """
        if self.df_fail is None:
            raise RuntimeError("Charge d'abord les données initiales avec load_raw_data()")

        # Charger le nouveau CSV
        new_df = pd.read_csv(new_failure_csv, encoding="utf-8-sig")
        for col in ["WOWO_DECLARATION_DATE", "WOWO_END_DATE", "WOWO_CREATION_DATE"]:
            if col in new_df.columns:
                new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
        new_df["annee"] = new_df["WOWO_DECLARATION_DATE"].dt.year
        new_df["mois"]  = new_df["WOWO_DECLARATION_DATE"].dt.month
        new_df["duree"] = (new_df["WOWO_END_DATE"] - new_df["WOWO_DECLARATION_DATE"]).dt.days

        # Vérifier compatibilité (mêmes colonnes essentielles)
        required_cols = ["WOWO_DECLARATION_DATE", "WOWO_END_DATE",
                         "WOWO_EQUIPMENT", "WOWO_EQUIPMENT_LEVEL",
                         "WOWO_SYSTEM_EQUIPMENT", "failure_parent_code", "failure_parent_level"]
        missing = [c for c in required_cols if c not in new_df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le nouveau CSV : {missing}")

        # Stats du nouveau dataset
        new_stats = {
            "n_rows_new":       int(len(new_df)),
            "date_min_new":     new_df["WOWO_DECLARATION_DATE"].min().isoformat(),
            "date_max_new":     new_df["WOWO_DECLARATION_DATE"].max().isoformat(),
            "n_composants_new": int(new_df["WOWO_EQUIPMENT"].nunique()),
        }

        # Composants existants vs nouveaux
        existing_comps = set(self.df_fail["WOWO_EQUIPMENT"].unique())
        new_comps_in_file = set(new_df["WOWO_EQUIPMENT"].unique())
        truly_new_comps = new_comps_in_file - existing_comps
        common_comps    = new_comps_in_file & existing_comps

        # Fusion
        merged = pd.concat([self.df_fail, new_df], ignore_index=True)
        merged = merged.sort_values(["WOWO_EQUIPMENT", "WOWO_DECLARATION_DATE"]).reset_index(drop=True)
        # Détecte le nouveau range d'années
        years_in_merged = sorted(merged["annee"].dropna().unique().astype(int).tolist())

        # Met à jour df_fail (le pipeline va recalculer tout le reste sur appel)
        self.df_fail = merged

        # Reset des phases postérieures (il faudra refaire feature engineering + prétraitement)
        self.df_panel = self.df_rul = self.df_final = self.df_export = None
        self.is_ready = False
        self.phases_completed = ["load_raw_data"]

        return {
            "before": {
                "n_rows":         int(len(self.df_fail) - len(new_df)),
                "n_composants":   len(existing_comps),
            },
            "new": new_stats,
            "after_merge": {
                "n_rows_total":     int(len(merged)),
                "n_composants_total": int(merged["WOWO_EQUIPMENT"].nunique()),
                "common_components": len(common_comps),
                "new_components":    len(truly_new_comps),
                "list_new_components": sorted(truly_new_comps)[:20],  # 20 premiers
                "years_covered":     years_in_merged,
            },
            "next_steps": [
                "Lance compute_features() pour recalculer Dataset_V1",
                "Lance prepare_sequences() pour préparer les tenseurs",
                "Réentraîne le modèle avec les nouvelles données",
            ],
        }

    # ═══════════════════════════════════════════════════════════════
    # UTILITAIRES
    # ═══════════════════════════════════════════════════════════════
    def get_test_dataframe(self) -> pd.DataFrame:
        """Récupère le DataFrame de test (utile pour le tableau dates pannes)."""
        if not hasattr(self, "_df_test"):
            raise RuntimeError("Lance prepare_sequences() d'abord")
        return self._df_test

    def get_info(self) -> Dict:
        """Métadonnées générales du pipeline."""
        return {
            "id":   self.PIPELINE_ID,
            "name": self.PIPELINE_NAME,
            "description": self.PIPELINE_DESCRIPTION,
            "is_ready":      self.is_ready,
            "phases_done":   self.phases_completed,
            "year":          self.year,
            "min_failures":  self.min_failures,
            "n_composants":  int(self.df_export["failure_comp"].nunique()) if self.df_export is not None else 0,
            "current_max_rul": self.current_max_rul,
            "lookback":      self.lookback,
            "num_classes_comp": self.num_classes_comp,
        }
