"""
main.py — Serveur FastAPI Cevital (Phase 1).

Routes implémentées :
  ✅ /api/pipelines                          — registry des pipelines
  ✅ /api/datasets/upload                    — uploader 2 CSV (failure + equipment)
  ✅ /api/datasets [GET/DELETE]              — CRUD lecture
  ✅ /api/datasets/{id}/eda_raw              — Phase 1 du pipeline (EDA brute)
  ✅ /api/datasets/{id}/features             — Phase 2 (Feature Engineering)
  ✅ /api/datasets/{id}/download_v1          — Télécharger Dataset_V1.csv
  ✅ /api/datasets/{id}/eda_features         — Phase 3 (EDA sur les features)
  ✅ /api/datasets/{id}/preprocessing        — Phase 4 (Tenseurs LSTM/GRU)
  ✅ /api/datasets/{id}/analyze_merge        — Preview fusion temporelle
  ✅ /api/datasets/{id}/merge                — Fusion temporelle effective
  ✅ /api/train/manual                       — Entraînement manuel (BackgroundTask)
  ✅ /api/train/auto                         — AutoML Bayésien (BackgroundTask)
  ✅ /api/experiments [CRUD]                 — Leaderboard / suppression
  ✅ /api/experiments/{id}/details           — Détails complets (5 onglets ModelDetails)
  ✅ /api/experiments/{id}/retrain           — Pré-remplir formulaire avec hyperparams
  ✅ /api/experiments/{id}/recompute_classification — Slider seuil
  ⏳ /api/experiments/{id}/export            — Phase 5 (ZIP complet)
  ✅ /api/rnn_demo/run, lstm_demo/run, transformer_demo/run — démos pédagogiques
  ✅ /ws/ingestion, /ws/{exp_id}             — WebSockets (scaffolding Phase 3)
"""
from __future__ import annotations

import os
import io
import json
import shutil
import asyncio
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd
import joblib
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    Depends, HTTPException, BackgroundTasks,
    UploadFile, File, Form, Body, Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# ─── BDD ────────────────────────────────────────────────────────
from models import (
    init_db, get_db, SessionLocal,
    Dataset, DatasetStatus,
    Experiment, TrainingStatus,
)

# ─── Architecture pipelines ────────────────────────────────────
from pipelines import CevitalPipeline, PIPELINE_REGISTRY
from pipelines.registry import list_pipelines

# ─── Démos pédagogiques (autonomes via mini-dataset synthétique) ─
from demos import get_synthetic_pipeline
from demos.rnn_demo         import run_demo as run_rnn_demo
from demos.lstm_demo        import run_demo as run_lstm_demo
from demos.transformer_demo import run_demo as run_transformer_demo


# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR   = BASE_DIR / "models"
EXPORTS_DIR  = BASE_DIR / "exports"
for _d in (DATASETS_DIR, MODELS_DIR, EXPORTS_DIR):
    _d.mkdir(exist_ok=True)


app = FastAPI(
    title="Plateforme PdM Cevital — PFE Master 2",
    description="Maintenance prédictive Cevital (RUL jours) + démos pédagogiques.",
    version="3.1.0-phase1",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()


def _fix_dataset_paths():
    """
    Corrige automatiquement les chemins absolus stockés en BDD.
    Utile quand la BDD est partagée entre machines (git clone / binôme).
    Les chemins sont reconstruits à partir de DATASETS_DIR de la machine actuelle.
    """
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).all()
        for ds in datasets:
            folder = DATASETS_DIR / str(ds.id)
            if not folder.exists():
                continue  # dossier absent → on ne touche pas
            changed = False

            # Reconstruire folder_path
            new_folder = str(folder)
            if ds.folder_path != new_folder:
                ds.folder_path = new_folder
                changed = True

            # Reconstruire failure_path
            f_path = folder / "failure.csv"
            if f_path.exists() and str(f_path) != ds.failure_path:
                ds.failure_path = str(f_path)
                changed = True

            # Reconstruire equipment_path
            e_path = folder / "equipment.csv"
            if e_path.exists() and str(e_path) != ds.equipment_path:
                ds.equipment_path = str(e_path)
                changed = True

            # Reconstruire v1_path
            v1_path = folder / "dataset_v1.csv"
            if v1_path.exists() and str(v1_path) != ds.v1_path:
                ds.v1_path = str(v1_path)
                changed = True

            if changed:
                db.add(ds)

        db.commit()
        print(f"[startup] Chemins datasets corrigés pour {len(datasets)} dataset(s).")
    except Exception as e:
        print(f"[startup] Erreur fix_dataset_paths: {e}")
    finally:
        db.close()

_fix_dataset_paths()


# ═══════════════════════════════════════════════════════════════
# Pipeline cache (un instance par dataset)
# ═══════════════════════════════════════════════════════════════
_PIPELINE_CACHE: Dict[int, CevitalPipeline] = {}
_PIPELINE_LOCK = threading.Lock()


def _detect_dominant_year(df) -> Optional[int]:
    """
    Détecte l'année dominante dans df_fail.
    Critère : année avec le PLUS de pannes de niveau 3/4 (équipements composants
    qu'on veut modéliser). Permet d'éviter le piège du défaut `year=2023` du
    pipeline lorsque le CSV uploadé couvre une autre année.
    """
    if df is None or df.empty or "annee" not in df.columns:
        return None
    # On privilégie les niveaux 3/4 (= composants — la cible du modèle)
    if "WOWO_EQUIPMENT_LEVEL" in df.columns:
        df34 = df[df["WOWO_EQUIPMENT_LEVEL"].isin([3.0, 4.0])]
        if not df34.empty and df34["annee"].notna().any():
            return int(df34["annee"].dropna().mode().iloc[0])
    # Fallback : année dominante toutes lignes confondues
    years = df["annee"].dropna()
    if years.empty:
        return None
    return int(years.mode().iloc[0])


def _get_pipeline(dataset_id: int, db: Session) -> CevitalPipeline:
    """
    Récupère le pipeline depuis le cache, ou l'instancie + recharge les CSV.

    ⚠️ Auto-détecte l'année dominante du CSV. Le notebook hardcode `year=2023`
    en défaut, mais si le CSV uploadé couvre une autre année, tout le pipeline
    devient silencieusement inopérant (filtre vide → dataframe vide → KeyError
    sur `failure_comp` plus loin). On corrige ce piège côté API en réinstanciant
    le pipeline avec la bonne année.
    """
    with _PIPELINE_LOCK:
        if dataset_id in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[dataset_id]
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise HTTPException(404, f"Dataset {dataset_id} introuvable")

        # ✅ Toujours reconstruire les chemins depuis DATASETS_DIR local
        # → évite les chemins absolus d'une autre machine stockés en BDD
        folder       = DATASETS_DIR / str(dataset_id)
        failure_path = folder / "failure.csv"
        equipment_path = folder / "equipment.csv"

        if not failure_path.exists():
            raise HTTPException(400, f"failure.csv introuvable dans {folder}")
        if not equipment_path.exists():
            raise HTTPException(400, f"equipment.csv introuvable dans {folder}")

        # Mettre à jour la BDD avec les chemins locaux corrects
        ds.folder_path    = str(folder)
        ds.failure_path   = str(failure_path)
        ds.equipment_path = str(equipment_path)
        v1 = folder / "dataset_v1.csv"
        if v1.exists():
            ds.v1_path = str(v1)
        db.add(ds); db.commit()

        # 1. Première instanciation pour détecter l'année dominante
        pipe = CevitalPipeline()
        pipe.load_raw_data(str(failure_path), str(equipment_path))

        # 2. Si année dominante ≠ défaut, réinstancier proprement
        dominant = _detect_dominant_year(pipe.df_fail)
        if dominant is not None and dominant != pipe.year:
            print(f"[pipeline] auto-detected year={dominant} (default {pipe.year}) — reinstantiating")
            pipe = CevitalPipeline(config={"year": dominant})
            pipe.load_raw_data(str(failure_path), str(equipment_path))

        _PIPELINE_CACHE[dataset_id] = pipe
        return pipe


def _invalidate_pipeline(dataset_id: int):
    with _PIPELINE_LOCK:
        _PIPELINE_CACHE.pop(dataset_id, None)


def _prepare_pipeline_sync(dataset_id: int):
    """Prépare le pipeline (chargement brut + compute_features + prepare_sequences).

    SYNCHRONE et coûteux (pandas/numpy) — conçu pour tourner dans un thread via
    asyncio.to_thread() afin de NE PAS bloquer la boucle asyncio. Sans ça, le
    WebSocket ne peut pas se connecter pendant la préparation (cause de
    « Erreur WebSocket » côté frontend). Ouvre sa propre session DB car les
    sessions SQLAlchemy ne sont pas thread-safe.
    """
    db = next(get_db())
    try:
        ds   = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        pipe = _get_pipeline(dataset_id, db)
        if pipe.X_train_num is None:
            if not ds.preproc_config:
                raise RuntimeError("Aucune config prétraitement — relance /preprocessing")
            pc = dict(ds.preproc_config)   # copie défensive
            pc.pop("healthy_sample_frac", None)
            feature_cols_override = pc.pop("feature_cols", None)
            if pipe.df_export is None:
                pipe.compute_features()
            if feature_cols_override:
                available = list(pipe.df_export.columns) if pipe.df_export is not None else []
                missing   = [f for f in feature_cols_override if f not in available]
                if not missing:
                    pipe.FEATURE_COLS = list(feature_cols_override)
            pipe.prepare_sequences(**pc)
        return pipe
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# Gestionnaire WebSocket
# ═══════════════════════════════════════════════════════════════
class ConnectionManager:
    def __init__(self):
        self.training:  dict[int, WebSocket]      = {}
        self.queues:    dict[int, asyncio.Queue]  = {}
        self.ingestion: Optional[WebSocket]       = None

    async def connect_training(self, experiment_id: int, ws: WebSocket):
        await ws.accept()
        self.training[experiment_id] = ws
        self.queues[experiment_id]   = asyncio.Queue()

    async def connect_ingestion(self, ws: WebSocket):
        await ws.accept()
        self.ingestion = ws

    def disconnect_training(self, experiment_id: int):
        self.training.pop(experiment_id, None)
        q = self.queues.pop(experiment_id, None)
        if q is not None:
            try:
                q.put_nowait(None)   # sentinel — réveille le pump task
            except Exception:
                pass

    def disconnect_ingestion(self):
        self.ingestion = None

    async def send_training(self, experiment_id: int, data: dict):
        q = self.queues.get(experiment_id)
        if q is not None:
            await q.put(data)
        else:
            print(f"[WS] ⚠ message perdu exp={experiment_id} type={data.get('type')} — aucun WS connecté", flush=True)


manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════
# Health + Pipelines registry
# ═══════════════════════════════════════════════════════════════
@app.get("/health")
def health_check():
    return {
        "status":  "ok",
        "phase":   "1",
        "version": app.version,
        "ts":      datetime.utcnow().isoformat(),
    }


@app.get("/api/pipelines")
def get_pipelines():
    return {
        "pipelines": list_pipelines(),
        "default":   CevitalPipeline.PIPELINE_ID,
    }


# ═══════════════════════════════════════════════════════════════
# 1.4.1 — DATASETS CRUD
# ═══════════════════════════════════════════════════════════════
def _validate_csv_columns(path: Path, required: list[str], file_label: str):
    """
    Vérifie qu'un CSV contient les colonnes requises (essaie sep=',' puis ';').
    Lève HTTPException 400 avec message explicite sinon — évite les bugs
    silencieux type "EREQ_CODE not in columns" rencontrés quand l'utilisateur
    uploade un mauvais fichier ou avec un mauvais séparateur.
    """
    try:
        df_head = pd.read_csv(path, encoding="utf-8-sig", nrows=1)
    except Exception as e:
        raise HTTPException(400, f"{file_label} : illisible en CSV ({e})")

    # Si une seule colonne géante avec des `;` dedans, on retente avec ';'
    if df_head.shape[1] == 1 and ";" in (df_head.columns[0] if df_head.columns.size else ""):
        try:
            df_head = pd.read_csv(path, encoding="utf-8-sig", nrows=1, sep=";")
            # Si on a pu lire correctement avec ;, on ré-écrit le fichier en ,
            df_full = pd.read_csv(path, encoding="utf-8-sig", sep=";")
            df_full.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[upload] {file_label}: converti ; → , automatiquement", flush=True)
        except Exception:
            raise HTTPException(
                400,
                f"{file_label} : séparateur CSV non standard. Utilise une virgule ',' "
                f"(ou un point-virgule ';' — auto-converti)."
            )

    cols = list(df_head.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        raise HTTPException(
            400,
            f"{file_label} invalide : colonnes manquantes {missing}. "
            f"Colonnes trouvées : {cols[:15]}"
        )


@app.post("/api/datasets/upload")
async def upload_dataset(
    name:            str        = Form(...),
    failure_file:    UploadFile = File(...),
    equipment_file:  UploadFile = File(...),
    db:              Session    = Depends(get_db),
):
    """Crée un Dataset en BDD + sauvegarde les 2 CSV dans backend/datasets/{id}/."""
    if db.query(Dataset).filter(Dataset.name == name).first():
        raise HTTPException(409, f"Un dataset nommé '{name}' existe déjà")

    ds = Dataset(name=name, folder_path="", status=DatasetStatus.UPLOADED)
    db.add(ds); db.commit(); db.refresh(ds)

    folder = DATASETS_DIR / str(ds.id)
    folder.mkdir(parents=True, exist_ok=True)
    ds.folder_path = str(folder)

    failure_path   = folder / "failure.csv"
    equipment_path = folder / "equipment.csv"
    for upload, dest in ((failure_file, failure_path),
                         (equipment_file, equipment_path)):
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    # ✅ Lot A.1 — Validateurs (évitent l'upload du mauvais fichier)
    try:
        _validate_csv_columns(
            failure_path,
            required=["WOWO_DECLARATION_DATE", "WOWO_END_DATE",
                      "WOWO_EQUIPMENT", "WOWO_EQUIPMENT_LEVEL",
                      "failure_parent_code", "failure_parent_level"],
            file_label="failure.csv",
        )
        _validate_csv_columns(
            equipment_path,
            required=["EREQ_CODE", "EREQ_LEVEL", "EREQ_PARENT_EQUIPMENT"],
            file_label="equipment.csv",
        )
    except HTTPException:
        # Cleanup BDD + dossier si validation échoue
        shutil.rmtree(folder, ignore_errors=True)
        db.delete(ds); db.commit()
        raise

    ds.failure_path   = str(failure_path)
    ds.equipment_path = str(equipment_path)
    db.commit(); db.refresh(ds)

    return ds.to_dict()


@app.get("/api/datasets")
def list_datasets(
    status: Optional[str] = Query(None, description="Filtrer par status"),
    db: Session = Depends(get_db),
):
    q = db.query(Dataset).order_by(Dataset.created_at.desc())
    if status:
        q = q.filter(Dataset.status == status)
    return [ds.to_dict() for ds in q.all()]


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} introuvable")
    return ds.to_dict()


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} introuvable")
    # Suppression dossier disque
    if ds.folder_path and Path(ds.folder_path).exists():
        try:
            shutil.rmtree(ds.folder_path)
        except Exception:
            pass
    _invalidate_pipeline(dataset_id)
    db.delete(ds)
    db.commit()
    return {"deleted": dataset_id}


# ═══════════════════════════════════════════════════════════════
# 1.4.1b — MISE À JOUR DES DONNÉES (nouveau export GMAO)
# ═══════════════════════════════════════════════════════════════
@app.post("/api/datasets/{dataset_id}/update_data")
async def update_dataset_data(
    dataset_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ajoute de nouvelles données au failure.csv existant (Option A).
    - Le nouveau fichier contient UNIQUEMENT de nouvelles interventions (dates > failure existant)
    - Normalise les colonnes si nouveau format GMAO (date_declaration, equipment_code, ...)
    - Vérifie les doublons avant concaténation
    - Concatène avec le failure.csv existant et sauvegarde
    - Remet le dataset au statut 'uploaded' pour forcer un re-run du pipeline
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} introuvable")

    folder = DATASETS_DIR / str(dataset_id)
    folder.mkdir(exist_ok=True)

    # Lire le fichier uploadé
    content = await file.read()
    try:
        df_new = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(400, f"Impossible de lire le CSV : {e}")

    # Détecter le format et normaliser
    fmt = CevitalPipeline.detect_format(df_new)
    if fmt == 'unknown':
        raise HTTPException(400,
            f"Format non reconnu. Colonnes trouvées : {list(df_new.columns)[:10]}. "
            f"Attendu : colonnes 'date_declaration/equipment_code' (nouveau) "
            f"ou 'WOWO_DECLARATION_DATE/WOWO_EQUIPMENT' (ancien).")

    if fmt == 'new':
        df_new_norm, df_equip_new = CevitalPipeline.normalize_gmao_export(df_new)
    else:
        df_new_norm = df_new

    # Lire le failure.csv existant
    fail_path = folder / "failure.csv"
    if not fail_path.exists():
        raise HTTPException(400, "failure.csv existant introuvable — impossible de concaténer.")
    df_existing = pd.read_csv(fail_path, encoding="utf-8-sig")
    n_existing  = len(df_existing)

    # Vérifier les doublons uniquement dans les nouvelles lignes par rapport à l'existant
    # (on ne touche PAS aux lignes existantes)
    key_cols = [c for c in ["WOWO_DECLARATION_DATE", "WOWO_EQUIPMENT"]
                if c in df_existing.columns and c in df_new_norm.columns]
    if key_cols:
        existing_keys  = set(map(tuple, df_existing[key_cols].astype(str).values.tolist()))
        new_key_series = df_new_norm[key_cols].astype(str).apply(tuple, axis=1)
        is_dup         = new_key_series.isin(existing_keys)
        n_duplicates   = int(is_dup.sum())
        df_to_add      = df_new_norm[~is_dup]
    else:
        n_duplicates = 0
        df_to_add    = df_new_norm

    df_combined = pd.concat([df_existing, df_to_add], ignore_index=True)

    # Trier par date
    date_col = "WOWO_DECLARATION_DATE"
    if date_col in df_combined.columns:
        df_combined[date_col] = pd.to_datetime(df_combined[date_col], errors="coerce")
        df_combined = df_combined.sort_values(date_col).reset_index(drop=True)
        dates    = df_combined[date_col].dropna()
        date_min = dates.min().strftime("%Y-%m-%d") if len(dates) else None
        date_max = dates.max().strftime("%Y-%m-%d") if len(dates) else None
    else:
        date_min = date_max = None

    # 🆕 Backup de l'état AVANT ajout → permet d'annuler le dernier ajout.
    #    On copie le failure.csv ACTUEL (pas encore écrasé) vers failure_prev.csv.
    import shutil
    shutil.copy2(fail_path, folder / "failure_prev.csv")
    (folder / "last_update.json").write_text(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "n_before":  n_existing,
        "n_added":   int(len(df_to_add)),
        "n_after":   int(len(df_combined)),
        "date_min":  date_min,
        "date_max":  date_max,
    }, ensure_ascii=False), encoding="utf-8")

    # Sauvegarder failure.csv mis à jour
    df_combined.to_csv(fail_path, index=False)

    # equipment.csv non modifié — les composants existent déjà

    # Mettre à jour la BDD
    ds.failure_path = str(fail_path)
    ds.folder_path  = str(folder)
    ds.n_rows       = int(len(df_combined))
    ds.period_start = date_min
    ds.period_end   = date_max
    ds.status       = DatasetStatus.UPLOADED
    ds.v1_path      = None
    db.add(ds); db.commit()

    _invalidate_pipeline(dataset_id)

    return {
        "ok":          True,
        "format":      fmt,
        "n_existing":  n_existing,
        "n_new":       int(len(df_to_add)),
        "n_duplicates": n_duplicates,
        "n_total":     int(len(df_combined)),
        "date_min":    date_min,
        "date_max":    date_max,
        "n_cols":    int(len(df_combined.columns)),
        "msg":       f"✅ Données mises à jour ({len(df_combined):,} lignes · {date_min} → {date_max})",
    }


@app.get("/api/datasets/{dataset_id}/can_undo_update")
def can_undo_update(dataset_id: int):
    """Indique si le DERNIER ajout de données peut être annulé (backup présent)."""
    folder = DATASETS_DIR / str(dataset_id)
    backup = folder / "failure_prev.csv"
    if not backup.exists():
        return {"can_undo": False}
    info = {}
    meta_f = folder / "last_update.json"
    if meta_f.exists():
        try:
            info = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception:
            info = {}
    return {"can_undo": True, "info": info}


@app.post("/api/datasets/{dataset_id}/undo_update")
def undo_update(dataset_id: int, db: Session = Depends(get_db)):
    """Annule le DERNIER ajout : restaure le failure.csv d'avant l'ajout."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, f"Dataset {dataset_id} introuvable")
    folder    = DATASETS_DIR / str(dataset_id)
    backup    = folder / "failure_prev.csv"
    fail_path = folder / "failure.csv"
    if not backup.exists():
        raise HTTPException(400, "Aucun ajout à annuler (pas de sauvegarde).")

    import shutil
    shutil.copy2(backup, fail_path)                 # restaure l'état d'avant le dernier ajout
    backup.unlink(missing_ok=True)
    (folder / "last_update.json").unlink(missing_ok=True)

    # Recalcule les infos du dataset depuis le failure.csv restauré
    df = pd.read_csv(fail_path, encoding="utf-8-sig")
    date_col = "WOWO_DECLARATION_DATE"
    date_min = date_max = None
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(d):
            date_min = d.min().strftime("%Y-%m-%d")
            date_max = d.max().strftime("%Y-%m-%d")

    ds.failure_path = str(fail_path)
    ds.n_rows       = int(len(df))
    ds.period_start = date_min
    ds.period_end   = date_max
    ds.status       = DatasetStatus.UPLOADED
    ds.v1_path      = None
    db.add(ds); db.commit()

    _invalidate_pipeline(dataset_id)

    return {
        "ok":       True,
        "n_rows":   int(len(df)),
        "date_min": date_min,
        "date_max": date_max,
        "msg":      f"↩️ Dernier ajout annulé — {len(df):,} lignes ({date_min} → {date_max})",
    }


# ═══════════════════════════════════════════════════════════════
# 1.4.2 — EDA / FEATURES / PREPROCESSING (par dataset)
# ═══════════════════════════════════════════════════════════════
@app.post("/api/datasets/{dataset_id}/eda_raw")
def run_eda_raw(dataset_id: int, db: Session = Depends(get_db)):
    """Phase 1 du pipeline — EDA brute (failure1.csv)."""
    pipe = _get_pipeline(dataset_id, db)
    # ⚠️ Vérif equipment.csv avant tout — catche les datasets mal uploadés
    _check_equipment_columns(pipe)
    try:
        result = pipe.compute_eda_raw()
        # Ajout de l'année effectivement utilisée — utile au frontend pour vérification
        result["_year_used"] = int(pipe.year)
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        raise HTTPException(
            500,
            f"Erreur EDA brute : {type(e).__name__}: {e}\n--- traceback ---\n{tb}"
        )


def _check_equipment_columns(pipe: CevitalPipeline):
    """Lève HTTPException 400 si df_equip n'a pas les colonnes Cevital attendues."""
    if pipe.df_equip is None or pipe.df_equip.empty:
        raise HTTPException(400, "equipment.csv non chargé (df_equip vide).")
    required = ["EREQ_CODE", "EREQ_LEVEL", "EREQ_PARENT_EQUIPMENT"]
    missing  = [c for c in required if c not in pipe.df_equip.columns]
    if missing:
        cols_found = list(pipe.df_equip.columns)[:10]
        raise HTTPException(
            400,
            f"equipment.csv invalide : colonnes manquantes {missing}. "
            f"Colonnes trouvées : {cols_found}. "
            f"→ Supprime ce dataset (DELETE /api/datasets/<id>) et re-uploade "
            f"avec un equipment.csv contenant EREQ_CODE / EREQ_LEVEL / EREQ_PARENT_EQUIPMENT. "
            f"(Tu as peut-être uploadé un fichier de maintenance ou de pannes à la place.)"
        )


@app.post("/api/datasets/{dataset_id}/features")
def run_features(dataset_id: int, db: Session = Depends(get_db)):
    """Phase 2 — Feature Engineering (8 étapes) + export Dataset_V1.csv."""
    pipe = _get_pipeline(dataset_id, db)
    ds   = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # ─── Pré-checks défensifs ────────────────────────────────────
    if pipe.df_fail is None or pipe.df_fail.empty:
        raise HTTPException(400, "Données brutes non chargées (load_raw_data n'a rien produit).")

    # ⚠️ Vérif des colonnes equipment AVANT compute_features (évite la KeyError 'EREQ_CODE')
    _check_equipment_columns(pipe)

    df = pipe.df_fail
    available_years = sorted(df["annee"].dropna().unique().astype(int).tolist())

    df_year = df[df["annee"] == pipe.year]
    if df_year.empty:
        raise HTTPException(
            400,
            f"Aucune panne pour year={pipe.year}. Années disponibles dans ton CSV : "
            f"{available_years}. Auto-détection : recharge le dataset (DELETE + UPLOAD)."
        )

    if "WOWO_EQUIPMENT_LEVEL" not in df.columns:
        raise HTTPException(
            400,
            f"Colonne 'WOWO_EQUIPMENT_LEVEL' absente du CSV failure. "
            f"Colonnes trouvées : {list(df.columns)[:15]}…"
        )

    df_year_lvl = df_year[df_year["WOWO_EQUIPMENT_LEVEL"].isin([3.0, 4.0])]
    if df_year_lvl.empty:
        raise HTTPException(
            400,
            f"Aucun équipement de niveau 3 ou 4 pour year={pipe.year} "
            f"({len(df_year)} pannes au total mais 0 en niveau 3/4). "
            f"Le pipeline cible spécifiquement les composants (niveaux 3/4)."
        )

    # Vérifier qu'au moins un composant a `>= min_failures` pannes
    n_per_comp = df_year_lvl.groupby("WOWO_EQUIPMENT").size()
    n_ok = int((n_per_comp >= pipe.min_failures).sum())
    if n_ok == 0:
        raise HTTPException(
            400,
            f"Aucun composant n'a au moins {pipe.min_failures} pannes en {pipe.year}. "
            f"Composants niveau 3/4 trouvés : {len(n_per_comp)} (max pannes : "
            f"{int(n_per_comp.max()) if not n_per_comp.empty else 0})."
        )

    # ─── Feature Engineering proprement dit ───
    try:
        result = pipe.compute_features()
        if pipe.df_export is None or pipe.df_export.empty:
            raise RuntimeError(
                "compute_features() a renvoyé un df_export vide — "
                "vérifie la hiérarchie machineID (champs failure_parent_code / level)."
            )

        v1_path = Path(ds.folder_path) / "dataset_v1.csv"
        pipe.export_dataset_v1(str(v1_path))

        # Mise à jour BDD
        ds.v1_path        = str(v1_path)
        ds.n_rows         = int(result["n_rows"])
        ds.n_failures     = int(result["n_failures"])
        ds.n_maintenances = int(result["n_maintenances"])
        ds.n_composants   = int(result["n_composants"])
        ds.n_machines     = int(result["n_machines"])
        ds.period_start   = result["period_start"]
        ds.period_end     = result["period_end"]
        ds.status         = DatasetStatus.FEATURES_DONE
        db.commit()
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        raise HTTPException(
            500,
            f"Erreur Feature Engineering : {type(e).__name__}: {e}\n"
            f"Pipeline year={pipe.year} · années_csv={available_years}\n"
            f"--- traceback ---\n{tb}"
        )


@app.get("/api/datasets/{dataset_id}/download_v1")
def download_v1(dataset_id: int, db: Session = Depends(get_db)):
    """Télécharge le Dataset_V1.csv généré."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds or not ds.v1_path or not Path(ds.v1_path).exists():
        raise HTTPException(404, "Dataset_V1 non généré — lance d'abord /features")
    return FileResponse(
        ds.v1_path,
        filename=f"{ds.name}_V1.csv",
        media_type="text/csv",
    )


@app.post("/api/datasets/{dataset_id}/eda_features")
def run_eda_features(dataset_id: int, db: Session = Depends(get_db)):
    """Phase 3 — EDA sur le Dataset_V1 (stats descriptives + corrélations)."""
    pipe = _get_pipeline(dataset_id, db)
    try:
        # Recompute features si pas encore fait dans le pipeline en cache
        if pipe.df_export is None:
            pipe.compute_features()
        return pipe.compute_eda_features()
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        raise HTTPException(
            500,
            f"Erreur EDA features : {type(e).__name__}: {e}\n--- traceback ---\n{tb}"
        )


class PreprocessingRequest(BaseModel):
    lookback:        int   = Field(21, ge=3, le=180)
    current_max_rul: int   = Field(30, ge=5, le=365)
    weight_factor:   float = Field(4.0, ge=0.0, le=50.0)
    val_ratio:       float = Field(0.15, ge=0.05, le=0.4)
    test_ratio:      float = Field(0.15, ge=0.05, le=0.4)
    feature_cols:    Optional[List[str]] = Field(None,
                          description="Sous-ensemble de features. Si null, utilise les 11 du notebook.")


@app.post("/api/datasets/{dataset_id}/preprocessing")
def run_preprocessing(
    dataset_id: int,
    req: PreprocessingRequest = Body(...),
    db: Session = Depends(get_db),
):
    """Phase 4 — Prétraitement LSTM/GRU. Génère X_train_num, X_train_comp, …"""
    pipe = _get_pipeline(dataset_id, db)
    ds   = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    try:
        if pipe.df_export is None:
            pipe.compute_features()

        # 🆕 Lot C : override des features si l'UI en a sélectionné un sous-ensemble
        if req.feature_cols and len(req.feature_cols) > 0:
            # Vérifie que les features demandées existent dans Dataset_V1
            available = list(pipe.df_export.columns)
            missing   = [f for f in req.feature_cols if f not in available]
            if missing:
                raise HTTPException(
                    400,
                    f"Features inconnues : {missing}. Colonnes disponibles : {available}"
                )
            # Override de l'instance pipe (sans modifier la classe)
            pipe.FEATURE_COLS = list(req.feature_cols)

        result = pipe.prepare_sequences(
            lookback        = req.lookback,
            current_max_rul = req.current_max_rul,
            val_ratio       = req.val_ratio,
            test_ratio      = req.test_ratio,
            weight_factor   = req.weight_factor,
        )
        ds.preproc_config = {
            "lookback":        req.lookback,
            "current_max_rul": req.current_max_rul,
            "weight_factor":   req.weight_factor,
            "val_ratio":       req.val_ratio,
            "test_ratio":      req.test_ratio,
            "feature_cols":    list(pipe.FEATURE_COLS),
        }
        ds.status = DatasetStatus.PREPROCESSED
        db.commit()
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        raise HTTPException(
            500,
            f"Erreur prétraitement : {type(e).__name__}: {e}\n--- traceback ---\n{tb}"
        )


# ── 🆕 STREAMING : feature engineering étape-par-étape (NDJSON) ──────
# Même architecture que /preprocessing_stream : asyncio.Queue + run_in_executor
@app.post("/api/datasets/{dataset_id}/features_stream")
async def run_features_stream(dataset_id: int, db: Session = Depends(get_db)):
    """Phase 2 — version streaming. Émet 1 ligne NDJSON par étape du pipeline (8 étapes)."""
    pipe = _get_pipeline(dataset_id, db)
    ds   = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    _check_equipment_columns(pipe)

    # Pré-checks défensifs (avant le stream)
    if pipe.df_fail is None or pipe.df_fail.empty:
        raise HTTPException(400, "Données brutes non chargées.")
    available_years = sorted(pipe.df_fail["annee"].dropna().unique().astype(int).tolist())
    df_year = pipe.df_fail[pipe.df_fail["annee"] == pipe.year]
    if df_year.empty:
        raise HTTPException(400,
            f"Aucune panne pour year={pipe.year}. Années disponibles : {available_years}.")

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(step_id: str, step_label: str, data: dict):
        asyncio.run_coroutine_threadsafe(
            queue.put({
                "type":       "step",
                "step_id":    step_id,
                "step_label": step_label,
                "data":       data,
            }),
            loop,
        )

    def _run_sync():
        return pipe.compute_features(progress_callback=progress)

    async def worker():
        # 🆕 Session BDD fraîche (Depends peut être fermé avant la fin du stream)
        local_db = SessionLocal()
        try:
            await queue.put({
                "type": "step", "step_id": "started",
                "step_label": "Démarrage du pipeline",
                "data": {"message": "Backend reçu, calcul en cours…"},
            })
            result = await loop.run_in_executor(None, _run_sync)
            # Export V1 CSV + maj BDD (sur la session FRAÎCHE)
            local_ds = local_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if local_ds is not None:
                v1_path = Path(local_ds.folder_path) / "dataset_v1.csv"
                pipe.export_dataset_v1(str(v1_path))
                local_ds.v1_path        = str(v1_path)
                local_ds.n_rows         = int(result["n_rows"])
                local_ds.n_failures     = int(result["n_failures"])
                local_ds.n_maintenances = int(result["n_maintenances"])
                local_ds.n_composants   = int(result["n_composants"])
                local_ds.n_machines     = int(result["n_machines"])
                local_ds.period_start   = result["period_start"]
                local_ds.period_end     = result["period_end"]
                local_ds.status         = DatasetStatus.FEATURES_DONE
                local_db.commit()
                print(f"[features_stream] ✅ ds.id={dataset_id} status=features_done committé", flush=True)
            await queue.put({"type": "done", "data": result})
        except HTTPException as he:
            await queue.put({"type": "error", "message": str(he.detail)})
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)
            await queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            try:
                local_db.close()
            except Exception:
                pass
            await queue.put(None)

    async def generator():
        # Padding 4 KB pour pousser le 1er chunk TCP au-delà du buffering Chromium
        padding = "// " + (" " * 4090) + "\n"
        yield padding.encode("utf-8")
        await asyncio.sleep(0)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
                await asyncio.sleep(0)   # force le flush ASGI
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control":     "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ── 🆕 STREAMING : prétraitement étape-par-étape (NDJSON) ──────
# Le client consomme `application/x-ndjson` ligne par ligne. Chaque ligne est
# un événement JSON :
#   {"type":"step","step_id":"balance","step_label":"...","data":{...}}
#   {"type":"done","data":{<résultat final agrégé, comme la route classique>}}
#   {"type":"error","message":"..."}
# Le pipeline tourne dans un thread (CPU-bound), un asyncio.Queue fait le pont
# vers le générateur async qui yield les lignes au client en temps réel.
@app.post("/api/datasets/{dataset_id}/preprocessing_stream")
async def run_preprocessing_stream(
    dataset_id: int,
    req: PreprocessingRequest = Body(...),
    db: Session = Depends(get_db),
):
    """
    Phase 4 — version streaming NDJSON.

    ⚠️ Flushing forcé : Chromium bufferise les réponses fetch()+getReader()
    jusqu'à ~2 KB avant de les délivrer à JS. On envoie un padding initial
    de 4 KB pour pousser le 1er paquet TCP au-delà du seuil de buffering.

    ⚠️ DB session : FastAPI peut fermer `db` (Depends) avant la fin du stream.
    On ouvre une session FRAÎCHE via SessionLocal() à l'intérieur du worker
    pour garantir que `ds.status = PREPROCESSED` est bien committé.
    """
    pipe = _get_pipeline(dataset_id, db)
    # Vérif existence — on n'utilise pas cette `ds` au-delà
    if not db.query(Dataset).filter(Dataset.id == dataset_id).first():
        raise HTTPException(404, f"Dataset {dataset_id} introuvable")

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(step_id: str, step_label: str, data: dict):
        # Appelé depuis le thread worker → bridge thread-safe vers la loop async
        asyncio.run_coroutine_threadsafe(
            queue.put({
                "type":       "step",
                "step_id":    step_id,
                "step_label": step_label,
                "data":       data,
            }),
            loop,
        )

    def _run_sync():
        # ⚠️ Lourd : compute_features peut prendre plusieurs secondes la 1ère fois
        if pipe.df_export is None:
            pipe.compute_features()
        # Override des features si demandé par l'UI
        if req.feature_cols and len(req.feature_cols) > 0:
            available = list(pipe.df_export.columns)
            missing   = [f for f in req.feature_cols if f not in available]
            if missing:
                raise ValueError(
                    f"Features inconnues : {missing}. "
                    f"Colonnes disponibles : {available}"
                )
            pipe.FEATURE_COLS = list(req.feature_cols)
        return pipe.prepare_sequences(
            lookback          = req.lookback,
            current_max_rul   = req.current_max_rul,
            val_ratio         = req.val_ratio,
            test_ratio        = req.test_ratio,
            weight_factor     = req.weight_factor,
            progress_callback = progress,
        )

    async def worker():
        # 🆕 Session BDD fraîche, propre à ce worker — survit au cycle du Depends
        local_db = SessionLocal()
        try:
            # Signal de démarrage IMMÉDIAT (avant compute_features potentiel)
            await queue.put({
                "type": "step",
                "step_id": "started",
                "step_label": "Démarrage du pipeline",
                "data": {"message": "Backend reçu, préparation en cours…"},
            })
            result = await loop.run_in_executor(None, _run_sync)

            # Persiste sur la session FRAÎCHE
            local_ds = local_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if local_ds is not None:
                local_ds.preproc_config = {
                    "lookback":        req.lookback,
                    "current_max_rul": req.current_max_rul,
                    "weight_factor":   req.weight_factor,
                    "val_ratio":       req.val_ratio,
                    "test_ratio":      req.test_ratio,
                    "feature_cols":    list(pipe.FEATURE_COLS),
                }
                local_ds.status = DatasetStatus.PREPROCESSED
                local_db.commit()
                print(f"[preprocessing_stream] ✅ ds.id={dataset_id} status=preprocessed committé", flush=True)
            await queue.put({"type": "done", "data": result})
        except HTTPException as he:
            await queue.put({"type": "error", "message": str(he.detail)})
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)
            await queue.put({
                "type": "error",
                "message": f"{type(e).__name__}: {e}",
            })
        finally:
            try:
                local_db.close()
            except Exception:
                pass
            await queue.put(None)   # sentinel de fin

    async def generator():
        # 🔥 PADDING : 4 KB de JSON valide pour pousser le 1er chunk TCP
        # au-delà du seuil de buffering Chromium (~2 KB). On utilise un
        # event "ping" rempli d'espaces — JSON parfaitement valide, le
        # frontend l'ignore via `evt.type === 'ping'`.
        ping = {"type": "ping", "_pad": " " * 3950}
        yield (json.dumps(ping) + "\n").encode("utf-8")
        await asyncio.sleep(0)   # cède la main → flush ASGI

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                line = json.dumps(item, ensure_ascii=False) + "\n"
                yield line.encode("utf-8")
                await asyncio.sleep(0)   # 🔑 force le flush ASGI/HTTP après chaque event
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control":     "no-cache, no-transform",
            "X-Accel-Buffering": "no",        # désactive buffering nginx si présent
            "Content-Encoding":  "identity",  # interdit gzip (qui bufferise tout)
        },
    )


# ═══════════════════════════════════════════════════════════════
# 1.4.3 — FUSION (analyse + merge)
# ═══════════════════════════════════════════════════════════════
@app.post("/api/datasets/{dataset_id}/analyze_merge")
async def analyze_merge(
    dataset_id:        int,
    new_failure_file:  UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Analyse un nouveau CSV failure SANS fusionner (preview pour décision admin)."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset introuvable")

    tmp_path = Path(ds.folder_path) / "_merge_preview.csv"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(new_failure_file.file, f)
    try:
        # Crée un pipeline éphémère pour calculer le diff
        preview_pipe = CevitalPipeline()
        preview_pipe.load_raw_data(ds.failure_path, ds.equipment_path)
        # merge_new_data() lit le nouveau CSV et retourne le diff
        return preview_pipe.merge_new_data(str(tmp_path))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur analyse fusion : {e}")


@app.post("/api/datasets/{dataset_id}/merge")
async def merge_dataset(
    dataset_id:       int,
    new_failure_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Fusion temporelle effective + recalcul features → nouveau Dataset_V1."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset introuvable")

    new_path = Path(ds.folder_path) / "failure_merged.csv"
    with open(new_path, "wb") as f:
        shutil.copyfileobj(new_failure_file.file, f)

    try:
        pipe = _get_pipeline(dataset_id, db)
        merge_info = pipe.merge_new_data(str(new_path))

        # Recalcul features + dataset V1
        pipe.compute_features()
        v1_path = Path(ds.folder_path) / "dataset_v1_merged.csv"
        pipe.export_dataset_v1(str(v1_path))

        # MAJ BDD
        ds.v1_path        = str(v1_path)
        if pipe.df_export is not None:
            ds.n_rows         = int(len(pipe.df_export))
            ds.n_failures     = int(pipe.df_export["failure"].sum())
            ds.n_maintenances = int(pipe.df_export["maintenance"].sum())
            ds.n_composants   = int(pipe.df_export["failure_comp"].nunique())
            ds.n_machines     = int(pipe.df_export["machineID"].nunique())
            ds.period_start   = pipe.df_export["date"].min().isoformat()
            ds.period_end     = pipe.df_export["date"].max().isoformat()
        ds.status         = DatasetStatus.FEATURES_DONE
        ds.preproc_config = None   # à refaire après merge
        db.commit()
        return merge_info
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur fusion : {e}")


# ═══════════════════════════════════════════════════════════════
# 1.4.4 — TRAINING (manual + auto)
# ═══════════════════════════════════════════════════════════════
class TrainManualRequest(BaseModel):
    dataset_id:    int
    name:          str
    architecture:  str       = Field("LSTM", pattern="^(LSTM|GRU)$")
    embedding_dim: int       = Field(8, ge=2, le=64)
    num_layers:    int       = Field(2, ge=1, le=3)
    units:         List[int] = Field([64, 32])
    dropout_rates: List[float] = Field([0.2, 0.15])
    learning_rate: float     = Field(0.001, gt=0.0, le=0.1)
    batch_size:    int       = Field(32, ge=8, le=512)
    epochs:        int       = Field(50, ge=1, le=500)
    patience:      int       = Field(10, ge=1, le=50)
    notes:         Optional[str] = None


class TrainAutoRequest(BaseModel):
    dataset_id:    int
    name:          str
    architecture:  str = Field("LSTM", pattern="^(LSTM|GRU)$")
    max_trials:    int = Field(20, ge=2, le=100, description="n_calls pour gp_minimize (notebook : 20)")
    epochs:        int = Field(20, ge=5, le=200, description="Époques PAR FOLD CV (notebook : 20)")
    batch_size:    int = Field(32, ge=8, le=512)
    patience:      int = Field(4, ge=1, le=30, description="Patience EarlyStopping CV (notebook : 4)")
    embedding_search: List[int] = Field([4, 8, 16, 32])
    # Bornes espace de recherche — défaut = notebook PFE exact
    units_min:     int   = Field(32,  ge=8, le=512)
    units_max:     int   = Field(128, ge=8, le=512)
    units_step:    int   = Field(32,  ge=1, le=64, description="Ignoré (gp_minimize optimise en continu)")
    nb_layers_min: int   = Field(1, ge=1, le=4)
    nb_layers_max: int   = Field(1, ge=1, le=4, description="Couches LSTM (notebook : 1)")
    dropout_min:   float = Field(0.1, ge=0.0, le=0.9)
    dropout_max:   float = Field(0.4, ge=0.0, le=0.9)
    lr_choices:    List[float] = Field([1e-4, 1e-2], description="Bornes lr pour Real log-uniform")
    # Re-entraînement final du best model (notebook : EPOCHS_FIN=35, batch=64)
    final_epochs:  int = Field(35, ge=5, le=500, description="Époques entraînement final (notebook : 35)")
    notes:         Optional[str] = None


class TrainFullRequest(BaseModel):
    """Réentraînement de DÉPLOIEMENT sur 100 % des données (train+val+test).

    Mêmes hyperparamètres qu'en manuel (fixes). Le nombre d'époques utilisé est
    la *meilleure époque* de l'expérience source (val_loss minimal) si
    `source_experiment_id` est fourni ; sinon on retombe sur `epochs`.
    """
    dataset_id:    int
    name:          str
    architecture:  str       = Field("LSTM", pattern="^(LSTM|GRU)$")
    embedding_dim: int       = Field(8, ge=2, le=64)
    num_layers:    int       = Field(2, ge=1, le=3)
    units:         List[int] = Field([64, 32])
    dropout_rates: List[float] = Field([0.2, 0.15])
    learning_rate: float     = Field(0.001, gt=0.0, le=0.1)
    batch_size:    int       = Field(32, ge=8, le=512)
    epochs:        int       = Field(50, ge=1, le=500, description="Fallback si pas d'historique source")
    source_experiment_id: Optional[int] = Field(None, description="Expérience d'où récupérer la meilleure époque")
    notes:         Optional[str] = None


def _save_metadata_json(model_dir: Path, exp: Experiment,
                         pipeline: CevitalPipeline, architecture: str):
    """
    Sauvegarde `metadata.json` au format exact du notebook
    Pipeline_PFE_Cevital_CHAMPION.ipynb (cell 71).

    Schéma :
        {
          "model_type":         "LSTM" | "GRU",
          "lookback":           int,
          "max_rul":            int,
          "features":           [...],
          "levels_modelises":   [3, 4],
          "year_entrainement":  int,
          "min_failures":       int,
          "num_composants":     int,
          "embedding_dim":      int,
          "metrics_test": {
            "r2": float, "mae": float, "recall": float, "f1": float
          }
        }
    """
    hp = exp.hyperparams or {}
    metadata = {
        "model_type":         architecture,
        "lookback":           int(pipeline.lookback),
        "max_rul":            int(pipeline.current_max_rul),
        "features":           list(pipeline.FEATURE_COLS),
        "levels_modelises":   [3, 4],
        "year_entrainement":  int(pipeline.year),
        "min_failures":       int(pipeline.min_failures),
        "num_composants":     int(pipeline.num_classes_comp),
        "embedding_dim":      int(hp.get("embedding_dim", 8)),
        "metrics_test": {
            "r2":     float(exp.r2)        if exp.r2        is not None else None,
            "mae":    float(exp.mae)       if exp.mae       is not None else None,
            "recall": float(exp.recall)    if exp.recall    is not None else None,
            "f1":     float(exp.f1_score)  if exp.f1_score  is not None else None,
        },
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _save_model_artifacts(exp_id: int, pipeline: CevitalPipeline,
                          model, training_result: dict) -> Path:
    """
    Sauvegarde les artefacts dans backend/models/{exp_id}/ :
        model.keras
        scaler_x.pkl
        scaler_y.pkl
        features_list.json   (ordre exact des 9 features)
        comp_mapping.json    (nom → idx embedding)
        predictions.csv      (date, comp, y_true, y_pred, error)
    """
    model_dir = MODELS_DIR / str(exp_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modèle Keras
    model.save(model_dir / "model.keras")

    # 2. Scalers
    joblib.dump(pipeline.scaler_x, model_dir / "scaler_x.pkl")
    joblib.dump(pipeline.scaler_y, model_dir / "scaler_y.pkl")

    # 3. 🆕 Phase 5 — features_list.json (ordre exact des colonnes X_num)
    with open(model_dir / "features_list.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_cols":     list(pipeline.FEATURE_COLS),
            "n_features":       len(pipeline.FEATURE_COLS),
            "target_col":       pipeline.TARGET_COL,
            "lookback":         int(pipeline.lookback),
            "current_max_rul":  int(pipeline.current_max_rul),
            "comment": (
                "Ordre exact des colonnes à fournir au modèle dans X_num. "
                "Normaliser chaque colonne avec scaler_x avant prédiction."
            ),
        }, f, indent=2, ensure_ascii=False)

    # 4. comp_mapping.json — format PLAT (compat notebook PFE_CHAMPION cell 71)
    #    { "B3602R0092-R001": 0, "B6001T0015-EXTRUD-FO": 1, ... }
    comp_mapping = getattr(pipeline, "_comp_name_to_idx", {}) or {}
    if not comp_mapping and getattr(pipeline, "_df_test", None) is not None:
        df = pipeline._df_test
        comp_mapping = {
            str(name): int(idx)
            for name, idx in zip(df[pipeline.COMP_COL].astype(str),
                                 df.get("comp_idx", pd.Series([], dtype=int)).astype(int))
        }
    comp_mapping_flat = {str(k): int(v) for k, v in comp_mapping.items()}
    with open(model_dir / "comp_mapping.json", "w", encoding="utf-8") as f:
        json.dump(comp_mapping_flat, f, indent=2, ensure_ascii=False)

    # 5. Predictions
    y_true = np.array(training_result["y_true"])
    y_pred = np.array(training_result["y_pred"])
    n = min(len(y_true), len(y_pred))
    df_test = pipeline.get_test_dataframe()
    if df_test is not None:
        lb = pipeline.lookback
        date_col = list(df_test["date"].iloc[lb:].values[:n])
        comp_col = list(df_test[pipeline.COMP_COL].iloc[lb:].values[:n])
    else:
        date_col = []
        comp_col = []
    # Aligner date/comp sur la longueur des prédictions : en mode "full"
    # (déploiement) y_true/y_pred couvrent train+val+test → plus de lignes que le
    # seul jeu de test. On complète par "" pour éviter le mismatch de longueurs
    # (pandas exige des colonnes de même taille). Inchangé pour manuel/auto.
    date_col = (date_col + [""] * n)[:n]
    comp_col = (comp_col + [""] * n)[:n]
    df_preds = pd.DataFrame({
        "date":   date_col,
        "comp":   comp_col,
        "y_true": y_true[:n],
        "y_pred": y_pred[:n],
        "error":  np.abs(y_true[:n] - y_pred[:n]),
    })
    df_preds.to_csv(model_dir / "predictions.csv", index=False)

    return model_dir


async def _run_training_manual(exp_id: int, req: TrainManualRequest):
    """Task background : entraînement Cevital manuel."""
    from tuner import CevitalTuner
    db = next(get_db())
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        ds  = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
        exp.status = TrainingStatus.RUNNING
        db.commit()

        pipe = _get_pipeline(req.dataset_id, db)
        # Re-préparer les séquences si pas encore (ou refaite)
        if pipe.X_train_num is None:
            if not ds.preproc_config:
                raise RuntimeError("Aucune config prétraitement — relance /preprocessing")
            pc = dict(ds.preproc_config)   # copie défensive
            # feature_cols et healthy_sample_frac ne sont pas des paramètres
            # de prepare_sequences — on les retire avant de passer **pc.
            pc.pop("healthy_sample_frac", None)
            feature_cols_override = pc.pop("feature_cols", None)
            feature_cols_override = pc.pop("feature_cols", None)
            if pipe.df_export is None:
                pipe.compute_features()
            if feature_cols_override:
                # Vérifier que ces colonnes existent bien
                available = list(pipe.df_export.columns) if pipe.df_export is not None else []
                missing   = [f for f in feature_cols_override if f not in available]
                if not missing:
                    pipe.FEATURE_COLS = list(feature_cols_override)
            pipe.prepare_sequences(**pc)

        async def send_fn(payload: dict):
            await manager.send_training(exp_id, payload)

        tuner_obj = CevitalTuner(pipe, str(EXPORTS_DIR), req.name, send_fn=send_fn)
        result = await tuner_obj.train_manual(
            architecture  = req.architecture,
            embedding_dim = req.embedding_dim,
            num_layers    = req.num_layers,
            units         = req.units,
            dropout_rates = req.dropout_rates,
            learning_rate = req.learning_rate,
            epochs        = req.epochs,
            batch_size    = req.batch_size,
            patience      = req.patience,
        )

        model_dir = _save_model_artifacts(exp_id, pipe, result["model"], result)
        m = result["metrics"]
        exp.status       = TrainingStatus.COMPLETED
        exp.completed_at = datetime.utcnow()
        exp.duration_sec = result["duration_sec"]
        exp.r2           = m["r2"]
        exp.mae          = m["mae"]
        exp.rmse         = m["rmse"]
        exp.mape         = m["mape"]
        exp.accuracy     = m["accuracy"]
        exp.precision    = m["precision"]
        exp.recall       = m["recall"]
        exp.f1_score     = m["f1"]
        exp.training_history = result["training_history"]
        exp.model_dir    = str(model_dir)
        exp.hyperparams  = {
            "lookback":         pipe.lookback,
            "current_max_rul":  pipe.current_max_rul,
            "embedding_dim":    req.embedding_dim,
            "num_layers":       req.num_layers,
            "units":            req.units,
            "dropout_rates":    req.dropout_rates,
            "learning_rate":    req.learning_rate,
            "batch_size":       req.batch_size,
            "epochs":           req.epochs,
            "patience":         req.patience,
            "weight_factor":    (ds.preproc_config or {}).get("weight_factor", 15.0),
            "feature_cols":     pipe.FEATURE_COLS,
            "mode":             "manual",
        }
        # ✅ Lot A.3 — metadata.json au format notebook (cell 71)
        _save_metadata_json(model_dir, exp, pipe, req.architecture)
        db.commit()
    except Exception as e:
        traceback.print_exc()
        exp.status        = TrainingStatus.FAILED
        exp.error_message = str(e)
        db.commit()
    finally:
        db.close()


async def _run_training_auto(exp_id: int, req: TrainAutoRequest):
    """Task background : AutoML."""
    print(f"\n[AUTO] Task démarré pour exp_id={exp_id}", flush=True)
    from tuner import CevitalTuner
    db = next(get_db())
    try:
        print(f"[AUTO] DB connecté, chargement pipeline...", flush=True)
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        ds  = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
        exp.status = TrainingStatus.RUNNING
        db.commit()

        # ⚡ Préparation lourde (compute_features + prepare_sequences) déportée
        #    dans un thread → la boucle asyncio reste libre pour accepter le
        #    WebSocket pendant ce temps (sinon « Erreur WebSocket » au frontend).
        pipe = await asyncio.to_thread(_prepare_pipeline_sync, req.dataset_id)

        async def send_fn(payload: dict):
            await manager.send_training(exp_id, payload)

        print(f"[AUTO] Pipeline OK — X_train_num shape: {pipe.X_train_num.shape}", flush=True)
        tuner_obj = CevitalTuner(pipe, str(EXPORTS_DIR), req.name, send_fn=send_fn)
        print(f"[AUTO] CevitalTuner créé, lancement train_auto...", flush=True)
        result = await tuner_obj.train_auto(
            architecture     = req.architecture,
            max_trials       = req.max_trials,
            epochs           = req.epochs,
            batch_size       = req.batch_size,
            patience         = req.patience,
            embedding_search = tuple(req.embedding_search),
            units_min        = req.units_min,
            units_max        = req.units_max,
            units_step       = req.units_step,
            nb_layers_min    = req.nb_layers_min,
            nb_layers_max    = req.nb_layers_max,
            dropout_min      = req.dropout_min,
            dropout_max      = req.dropout_max,
            lr_choices       = tuple(req.lr_choices),
            final_epochs     = req.final_epochs,
        )

        model_dir = _save_model_artifacts(exp_id, pipe, result["model"], result)
        m = result["metrics"]
        exp.status       = TrainingStatus.COMPLETED
        exp.completed_at = datetime.utcnow()
        exp.duration_sec = result["duration_sec"]
        exp.r2           = m["r2"]
        exp.mae          = m["mae"]
        exp.rmse         = m["rmse"]
        exp.mape         = m["mape"]
        exp.accuracy     = m["accuracy"]
        exp.precision    = m["precision"]
        exp.recall       = m["recall"]
        exp.f1_score     = m["f1"]
        exp.total_trials     = req.max_trials
        exp.training_history = result.get("training_history")   # 🆕 courbes du re-train final
        exp.model_dir        = str(model_dir)
        exp.hyperparams      = {
            **result["best_hps"],
            "epochs_per_trial": req.epochs,
            "final_epochs":     req.final_epochs,
            "batch_size":       req.batch_size,
            "patience":         req.patience,
            "weight_factor":    (ds.preproc_config or {}).get("weight_factor", 15.0),
            "feature_cols":     pipe.FEATURE_COLS,
            "mode":             "auto",
        }
        exp.search_space = {
            "embedding_search": req.embedding_search,
            "nb_layers":        [req.nb_layers_min, req.nb_layers_max],
            "units":            [req.units_min, req.units_max, req.units_step],
            "dropout":          [req.dropout_min, req.dropout_max],
            "lr":               req.lr_choices,
            "epochs_per_trial": req.epochs,
            "final_epochs":     req.final_epochs,
        }
        # ✅ Lot A.3 — metadata.json au format notebook (cell 71)
        _save_metadata_json(model_dir, exp, pipe, req.architecture)
        db.commit()
    except Exception as e:
        traceback.print_exc()
        exp.status        = TrainingStatus.FAILED
        exp.error_message = str(e)
        db.commit()
    finally:
        db.close()


async def _run_training_full(exp_id: int, req: TrainFullRequest):
    """Task background : réentraînement DÉPLOIEMENT sur 100 % des données."""
    print(f"\n[FULL] Task démarré pour exp_id={exp_id}", flush=True)
    from full_trainer import CevitalFullTrainer
    db = next(get_db())
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        ds  = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
        exp.status = TrainingStatus.RUNNING
        db.commit()

        # Préparation pipeline dans un thread (la boucle reste libre → WS OK)
        pipe = await asyncio.to_thread(_prepare_pipeline_sync, req.dataset_id)

        async def send_fn(payload: dict):
            await manager.send_training(exp_id, payload)

        # ── Modèle source : meilleure époque (val_loss min) + métriques héritées ──
        src = None
        if req.source_experiment_id is not None:
            src = db.query(Experiment).filter(
                Experiment.id == req.source_experiment_id
            ).first()
        best_epochs = int(req.epochs)
        if src is not None and src.training_history:
            valid = [
                (int(h.get("epoch", i + 1)), float(h["val_loss"]))
                for i, h in enumerate(src.training_history)
                if h.get("val_loss") not in (None, 0, 0.0)
            ]
            if valid:
                best_epochs = min(valid, key=lambda t: t[1])[0]

        await send_fn({
            "type": "log",
            "message": (f"Meilleure époque retenue : {best_epochs} "
                        f"(depuis l'expérience #{req.source_experiment_id})")
                       if req.source_experiment_id else
                       f"Époques (fixe) : {best_epochs}",
        })

        print(f"[FULL] Pipeline OK — époques={best_epochs}, lancement train_full...", flush=True)
        trainer = CevitalFullTrainer(pipe, str(EXPORTS_DIR), req.name, send_fn=send_fn)
        result = await trainer.train_full(
            architecture  = req.architecture,
            embedding_dim = req.embedding_dim,
            num_layers    = req.num_layers,
            units         = req.units,
            dropout_rates = req.dropout_rates,
            learning_rate = req.learning_rate,
            epochs        = best_epochs,
            batch_size    = req.batch_size,
        )

        model_dir = _save_model_artifacts(exp_id, pipe, result["model"], result)
        # Métriques : on HÉRITE celles du modèle source (estimation honnête sur le
        # jeu de TEST). Les métriques d'ajustement (sur données vues à l'entraînement)
        # seraient trompeuses → on ne les stocke pas. Sans source : pas de métriques
        # (le modèle est sauvegardé quand même).
        exp.status       = TrainingStatus.COMPLETED
        exp.completed_at = datetime.utcnow()
        exp.duration_sec = result["duration_sec"]
        exp.r2        = src.r2        if src else None
        exp.mae       = src.mae       if src else None
        exp.rmse      = src.rmse      if src else None
        exp.mape      = src.mape      if src else None
        exp.accuracy  = src.accuracy  if src else None
        exp.precision = src.precision if src else None
        exp.recall    = src.recall    if src else None
        exp.f1_score  = src.f1_score  if src else None
        exp.training_history = result["training_history"]
        exp.model_dir        = str(model_dir)
        exp.hyperparams      = {
            "lookback":         pipe.lookback,
            "current_max_rul":  pipe.current_max_rul,
            "embedding_dim":    req.embedding_dim,
            "num_layers":       req.num_layers,
            "units":            req.units,
            "dropout_rates":    req.dropout_rates,
            "learning_rate":    req.learning_rate,
            "batch_size":       req.batch_size,
            "epochs":           best_epochs,
            "weight_factor":    (ds.preproc_config or {}).get("weight_factor", 15.0),
            "feature_cols":     pipe.FEATURE_COLS,
            "mode":             "full",
            "trained_on":       "all_data",
            "n_sequences":      result.get("n_sequences"),
            "source_experiment_id": req.source_experiment_id,
        }
        _save_metadata_json(model_dir, exp, pipe, req.architecture)
        db.commit()
    except Exception as e:
        traceback.print_exc()
        exp.status        = TrainingStatus.FAILED
        exp.error_message = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/api/train/manual", status_code=202)
async def train_manual(req: TrainManualRequest,
                        background_tasks: BackgroundTasks,
                        db: Session = Depends(get_db)):
    """Lance un entraînement manuel en BackgroundTask. Retourne exp_id."""
    ds = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset introuvable")
    if ds.status != DatasetStatus.PREPROCESSED:
        raise HTTPException(400, "Lance d'abord le prétraitement (/preprocessing)")

    exp = Experiment(
        name=req.name,
        architecture=req.architecture,
        mode="manual",
        status=TrainingStatus.PENDING,
        dataset_id=req.dataset_id,
        notes=req.notes,
    )
    db.add(exp); db.commit(); db.refresh(exp)

    background_tasks.add_task(_run_training_manual, exp.id, req)
    return {"experiment_id": exp.id, "status": "started"}


@app.post("/api/train/full", status_code=202)
async def train_full_endpoint(req: TrainFullRequest,
                              background_tasks: BackgroundTasks,
                              db: Session = Depends(get_db)):
    """Réentraînement de DÉPLOIEMENT sur 100 % des données. Retourne exp_id."""
    ds = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset introuvable")
    if ds.status != DatasetStatus.PREPROCESSED:
        raise HTTPException(400, "Lance d'abord le prétraitement (/preprocessing)")

    exp = Experiment(
        name=req.name,
        architecture=req.architecture,
        mode="full",
        status=TrainingStatus.PENDING,
        dataset_id=req.dataset_id,
        notes=req.notes,
    )
    db.add(exp); db.commit(); db.refresh(exp)

    background_tasks.add_task(_run_training_full, exp.id, req)
    return {"experiment_id": exp.id, "status": "started"}


_NEXT_FAIL_MODEL_CACHE: dict = {}   # exp_id -> modèle Keras chargé (évite de recharger à chaque appel)


@app.get("/api/experiments/{exp_id}/next_failures")
def get_next_failures(exp_id: int, db: Session = Depends(get_db)):
    """Prédit la PROCHAINE panne de chaque composant avec le modèle entraîné.

    Pour chaque composant : on prend sa fenêtre la plus récente (les `lookback`
    derniers jours), le modèle prédit le RUL → prochaine panne = dernière date du
    composant + RUL. Renvoie aussi les pannes passées (historique) pour la timeline.
    Endpoint synchrone (def) → exécuté dans le threadpool FastAPI (calcul lourd).
    """
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment introuvable")
    if not exp.model_dir:
        raise HTTPException(400, "Modèle non entraîné")
    model_path = Path(exp.model_dir) / "model.keras"
    if not model_path.exists():
        raise HTTPException(404, "model.keras introuvable")

    try:
        # Pipeline préparé (features + scalers + mapping composant) pour ce dataset.
        # ⚠️ 1er appel après un redémarrage = cache froid → préparation longue.
        print(f"[NEXTFAIL] exp={exp_id} — préparation pipeline (dataset {exp.dataset_id})...", flush=True)
        pipe = _prepare_pipeline_sync(exp.dataset_id)
        df = pipe.df_export
        if df is None or len(df) == 0:
            raise HTTPException(400, "Données du dataset indisponibles")

        # Modèle mis en cache par expérience → évite ~1-3 s de rechargement à chaque appel
        model = _NEXT_FAIL_MODEL_CACHE.get(exp_id)
        if model is None:
            print(f"[NEXTFAIL] exp={exp_id} — chargement modèle Keras (1ère fois)...", flush=True)
            from tensorflow.keras.models import load_model
            # compile=False : inférence seule → pas besoin de la loss custom
            # (asymmetric_rul_loss) ni de l'optimiseur. Évite le TypeError au chargement.
            model = load_model(str(model_path), compile=False)
            _NEXT_FAIL_MODEL_CACHE[exp_id] = model
        print(f"[NEXTFAIL] exp={exp_id} — pipeline+modèle OK ({len(df)} lignes), prédiction...", flush=True)

        lookback     = int(pipe.lookback)
        feature_cols = list(pipe.FEATURE_COLS)
        scaler_x     = pipe.scaler_x
        scaler_y     = pipe.scaler_y
        max_rul      = int(pipe.current_max_rul)
        comp_to_idx  = getattr(pipe, "_comp_name_to_idx", {}) or {}
        COMP         = pipe.COMP_COL

        df = df.sort_values([COMP, "date"])
        windows_num, windows_cat, meta = [], [], []
        skipped = 0
        for comp, g in df.groupby(COMP, sort=False):
            if len(g) < lookback:
                skipped += 1
                continue
            last_rows = g.iloc[-lookback:]
            Xn   = scaler_x.transform(last_rows[feature_cols].values).astype("float32")
            cidx = int(comp_to_idx.get(str(comp), 0))
            windows_num.append(Xn)
            windows_cat.append(np.full((lookback,), cidx, dtype="int32"))
            past_fail = (
                g.loc[g["failure"] == 1, "date"].astype(str).str.slice(0, 10).tolist()
                if "failure" in g.columns else []
            )
            meta.append({
                "comp":          str(comp),
                "last_date":     str(g["date"].iloc[-1])[:10],
                "past_failures": past_fail,
                "n_failures":    len(past_fail),
            })

        if not windows_num:
            return {"components": [], "max_rul": max_rul, "skipped": skipped, "n": 0}

        Xn_all = np.stack(windows_num)
        Xc_all = np.stack(windows_cat)
        preds  = model.predict([Xn_all, Xc_all], verbose=0).flatten()
        ruls   = np.clip(
            scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten(), 0, max_rul
        )

        results = []
        for mrow, rul in zip(meta, ruls):
            rul_days  = int(round(float(rul)))
            last_dt   = pd.to_datetime(mrow["last_date"])
            next_fail = (last_dt + pd.Timedelta(days=rul_days)).strftime("%Y-%m-%d")
            results.append({
                "comp":                   mrow["comp"],
                "last_date":              mrow["last_date"],
                "predicted_rul":          rul_days,
                "predicted_next_failure": next_fail,
                "past_failures":          mrow["past_failures"],
                "n_failures":             mrow["n_failures"],
            })
        results.sort(key=lambda r: r["predicted_rul"])   # plus urgent en premier
        print(f"[NEXTFAIL] exp={exp_id} — terminé ({len(results)} composants)", flush=True)
        return {
            "components": results,
            "max_rul":    max_rul,
            "n":          len(results),
            "skipped":    skipped,
            "data_end":   str(df["date"].max())[:10],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur prédiction prochaine panne : {type(e).__name__}: {e}")


@app.post("/api/train/auto", status_code=202)
async def train_auto(req: TrainAutoRequest,
                      background_tasks: BackgroundTasks,
                      db: Session = Depends(get_db)):
    """Lance un AutoML Bayésien en BackgroundTask. Retourne exp_id."""
    ds = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset introuvable")
    if ds.status != DatasetStatus.PREPROCESSED:
        raise HTTPException(400, "Lance d'abord le prétraitement (/preprocessing)")

    exp = Experiment(
        name=req.name,
        architecture=req.architecture,
        mode="auto",
        status=TrainingStatus.PENDING,
        dataset_id=req.dataset_id,
        notes=req.notes,
    )
    db.add(exp); db.commit(); db.refresh(exp)

    background_tasks.add_task(_run_training_auto, exp.id, req)
    return {"experiment_id": exp.id, "status": "started"}


# ═══════════════════════════════════════════════════════════════
# 1.4.5 — EXPERIMENTS (CRUD + details + retrain + classification)
# ═══════════════════════════════════════════════════════════════
@app.get("/api/experiments")
def list_experiments(db: Session = Depends(get_db)):
    rows = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return [exp.to_dict() for exp in rows]


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment introuvable")
    return exp.to_dict()


@app.delete("/api/experiments/{exp_id}")
def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment introuvable")
    if exp.model_dir and Path(exp.model_dir).exists():
        try:
            shutil.rmtree(exp.model_dir)
        except Exception:
            pass
    db.delete(exp); db.commit()
    return {"deleted": exp_id}


@app.get("/api/experiments/{exp_id}/details")
def get_experiment_details(exp_id: int, db: Session = Depends(get_db)):
    """Détails complets pour ModelDetails (5 onglets : Régression, Classif, Apprentissage, Prédictions, Config)."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment introuvable")

    out = exp.to_dict()

    # Predictions CSV
    preds_csv = None
    if exp.model_dir:
        p = Path(exp.model_dir) / "predictions.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                preds_csv = {
                    "y_true": df["y_true"].astype(float).tolist(),
                    "y_pred": df["y_pred"].astype(float).tolist(),
                    "dates":  df["date"].astype(str).tolist(),
                    "comp":   df["comp"].astype(str).tolist(),
                    "current_max_rul": (exp.hyperparams or {}).get("current_max_rul", 30),
                }
            except Exception:
                pass
    out["predictions"] = preds_csv

    # Files info
    files = {}
    if exp.model_dir:
        for fname in ("model.keras", "scaler_x.pkl", "scaler_y.pkl",
                      "comp_mapping.json", "predictions.csv"):
            fpath = Path(exp.model_dir) / fname
            if fpath.exists():
                files[fname] = {
                    "name":    fname,
                    "size_kb": round(fpath.stat().st_size / 1024, 1),
                }
    out["files"] = files

    return out


@app.post("/api/experiments/{exp_id}/retrain")
def retrain_experiment(exp_id: int, db: Session = Depends(get_db)):
    """Retourne les hyperparams + dataset_id pour pré-remplir le formulaire."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment introuvable")
    return {
        "hyperparams":  exp.hyperparams,
        "dataset_id":   exp.dataset_id,
        "architecture": exp.architecture,
        "mode":         exp.mode,
        "name":         f"{exp.name}_retrain",
    }


@app.get("/api/experiments/{exp_id}/recompute_classification")
def recompute_classification(
    exp_id: int,
    threshold: float = Query(10.0, ge=1.0, le=180.0),
    db: Session = Depends(get_db),
):
    """Recalcule accuracy/precision/recall/f1 + matrice de confusion avec nouveau seuil."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp or not exp.model_dir:
        raise HTTPException(404, "Experiment introuvable ou predictions absentes")

    preds_csv = Path(exp.model_dir) / "predictions.csv"
    if not preds_csv.exists():
        raise HTTPException(404, "predictions.csv absent")

    df = pd.read_csv(preds_csv)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    y_true_alert = (y_true <= threshold).astype(int)
    y_pred_alert = (y_pred <= threshold).astype(int)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )
    cm = confusion_matrix(y_true_alert, y_pred_alert, labels=[0, 1])
    return {
        "threshold":  threshold,
        "accuracy":   float(accuracy_score(y_true_alert, y_pred_alert)),
        "precision":  float(precision_score(y_true_alert, y_pred_alert, zero_division=0)),
        "recall":     float(recall_score(y_true_alert, y_pred_alert, zero_division=0)),
        "f1":         float(f1_score(y_true_alert, y_pred_alert, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": ["sain (>seuil)", "alerte (≤seuil)"],
        "n_total": int(len(y_true)),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 5 — EXPORT ZIP complet (réutilisation hors plateforme)
# ═══════════════════════════════════════════════════════════════
def _generate_readme(exp: Experiment) -> str:
    """Génère le README.md inclus dans le ZIP — snippet Python complet
    pour utiliser le modèle dans une autre app (sans dépendance Cevital)."""
    hp = exp.hyperparams or {}
    feature_cols = hp.get("feature_cols") or [
        "comp_level", "pannes_7j", "pannes_30j", "pannes_90j",
        "maint_7j",  "maint_30j",  "maint_90j",
        "DSLF",      "DSLM",
    ]
    feat_list = ", ".join(feature_cols)
    max_rul   = hp.get("current_max_rul", 30)
    lookback  = hp.get("lookback", 30)

    created  = exp.created_at.isoformat() if exp.created_at else "—"
    duration = f"{exp.duration_sec:.1f}s" if exp.duration_sec else "—"

    r2   = f"{exp.r2:.4f}"  if exp.r2   is not None else "—"
    mae  = f"{exp.mae:.2f}" if exp.mae  is not None else "—"
    rmse = f"{exp.rmse:.2f}" if exp.rmse is not None else "—"
    mape = f"{exp.mape:.2f}" if exp.mape is not None else "—"
    acc  = f"{exp.accuracy:.4f}"  if exp.accuracy  is not None else "—"
    prec = f"{exp.precision:.4f}" if exp.precision is not None else "—"
    rec  = f"{exp.recall:.4f}"    if exp.recall    is not None else "—"
    f1   = f"{exp.f1_score:.4f}"  if exp.f1_score  is not None else "—"

    hyper_json = json.dumps(hp, indent=2, ensure_ascii=False)

    return f"""# Modèle PdM Cevital — Export

## 📋 Métadonnées
- **Nom** : `{exp.name}`
- **Architecture** : `{exp.architecture}` ({exp.mode})
- **Date d'entraînement** : {created}
- **Durée** : {duration}
- **Dataset source** : id `{exp.dataset_id}`

## 📊 Métriques

### Régression (RUL en jours)
| Métrique | Valeur |
|----------|--------|
| R²       | {r2}   |
| MAE      | {mae} jours |
| RMSE     | {rmse} jours |
| MAPE     | {mape}% |

### Classification dérivée (seuil = 10 jours)
| Métrique | Valeur |
|----------|--------|
| Accuracy | {acc}  |
| Precision| {prec} |
| Recall   | {rec}  |
| F1       | {f1}   |

## 📁 Contenu du ZIP

| Fichier | Description |
|---------|-------------|
| `model.keras`         | Modèle Keras (2 inputs : `X_num` + `X_comp`) |
| `scaler_x.pkl`        | MinMaxScaler des features numériques |
| `scaler_y.pkl`        | MinMaxScaler du RUL |
| `features_list.json`  | Ordre exact des colonnes features |
| `comp_mapping.json`   | Mapping nom_composant → idx du layer Embedding |
| `config.json`         | Hyperparamètres + métadonnées |
| `metrics.json`        | Métriques de régression + classification |
| `predictions.csv`     | Prédictions sur le jeu de test |
| `README.md`           | Ce fichier |

## 🐍 Utilisation dans une application Python

```python
import json, joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ── 1. Charger les artefacts ──────────────────────────────
# compile=False : pour prédire, pas besoin de la loss custom (asymmetric_rul_loss)
model    = load_model("model.keras", compile=False)
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

with open("features_list.json") as f:
    features_info = json.load(f)
with open("comp_mapping.json") as f:
    comp_info = json.load(f)

feature_cols = features_info["feature_cols"]
lookback     = features_info["lookback"]          # = {lookback}
max_rul      = features_info["current_max_rul"]   # = {max_rul}
comp_mapping = comp_info["mapping"]

# ── 2. Préparer tes données ───────────────────────────────
# `df_recent` doit contenir AU MOINS `lookback` lignes consécutives
# pour le composant à analyser, avec ces colonnes (ordre exact) :
# {feat_list}

df_recent = pd.DataFrame(...)  # tes données
component_name = "B3602R0092-R001"
comp_idx = comp_mapping[component_name]

# ── 3. Construire les inputs (2 inputs : X_num + X_comp) ─
X_features = df_recent[feature_cols].values[-lookback:]    # (lookback, n_feat)
X_scaled   = scaler_x.transform(X_features)
X_num      = np.array([X_scaled])                          # (1, lookback, n_feat)
X_comp     = np.array([comp_idx])                          # (1,)

# ── 4. Prédire ────────────────────────────────────────────
raw_pred    = model.predict([X_num, X_comp])
pred_norm   = raw_pred[0][0]
pred_days   = scaler_y.inverse_transform([[pred_norm]])[0][0]
pred_days   = float(np.clip(pred_days, 0, max_rul))   # clip de sécurité

# ── 5. Interprétation ─────────────────────────────────────
if pred_days < 5:
    print(f"⚠️ ALERTE CRITIQUE  — RUL = {{pred_days:.1f}} j (< 5 j)")
elif pred_days < 10:
    print(f"🟠 SURVEILLANCE     — RUL = {{pred_days:.1f}} j (à planifier)")
else:
    print(f"✅ SAIN             — RUL = {{pred_days:.1f}} j")
```

## ⚙️ Hyperparamètres utilisés

```json
{hyper_json}
```

---
Généré par **PdM Platform Cevital** · PFE Master 2 Génie Logiciel.
"""


@app.get("/api/experiments/{exp_id}/export")
def export_experiment(exp_id: int, db: Session = Depends(get_db)):
    """
    Phase 5 — Génère un ZIP complet avec tous les artefacts du modèle.

    Contenu :
        model.keras, scaler_x.pkl, scaler_y.pkl,
        features_list.json, comp_mapping.json,
        config.json, metrics.json, predictions.csv, README.md
    """
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment introuvable")
    if exp.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            400,
            f"Modèle non complété (statut = {exp.status}). "
            "L'export n'est possible que pour les modèles terminés."
        )
    if not exp.model_dir or not Path(exp.model_dir).exists():
        raise HTTPException(404, f"Dossier modèle absent : {exp.model_dir}")

    model_dir = Path(exp.model_dir)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in exp.name)
    zip_path  = EXPORTS_DIR / f"exp_{exp.id}_{safe_name}.zip"

    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:

            # ── Artefacts depuis le dossier modèle (Lot A.4) ──
            #    metadata.json est inclus EN PLUS de config.json + metrics.json
            #    (format notebook PFE_CHAMPION cell 71, plat et lisible).
            for fname in ("model.keras", "scaler_x.pkl", "scaler_y.pkl",
                          "features_list.json", "comp_mapping.json",
                          "metadata.json", "predictions.csv"):
                fpath = model_dir / fname
                if fpath.exists():
                    zf.write(fpath, fname)

            # ── config.json (génération à la volée) ──
            zf.writestr("config.json", json.dumps({
                "experiment_id":   exp.id,
                "name":            exp.name,
                "architecture":    exp.architecture,
                "mode":            exp.mode,
                "dataset_id":      exp.dataset_id,
                "dataset_name":    exp.dataset.name if exp.dataset else None,
                "hyperparams":     exp.hyperparams,
                "search_space":    exp.search_space,
                "created_at":      exp.created_at.isoformat() if exp.created_at else None,
                "completed_at":    exp.completed_at.isoformat() if exp.completed_at else None,
                "duration_sec":    exp.duration_sec,
            }, indent=2, ensure_ascii=False))

            # ── metrics.json ──
            zf.writestr("metrics.json", json.dumps({
                "regression": {
                    "r2":   exp.r2,
                    "mae":  exp.mae,
                    "rmse": exp.rmse,
                    "mape": exp.mape,
                },
                "classification": {
                    "threshold_days": (exp.hyperparams or {}).get("classification_threshold", 10),
                    "accuracy":  exp.accuracy,
                    "precision": exp.precision,
                    "recall":    exp.recall,
                    "f1":        exp.f1_score,
                },
            }, indent=2, ensure_ascii=False))

            # ── README.md (snippet Python complet) ──
            zf.writestr("README.md", _generate_readme(exp))

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        raise HTTPException(500, f"Erreur génération ZIP : {type(e).__name__}: {e}")

    return FileResponse(
        path=str(zip_path),
        filename=f"{safe_name}.zip",
        media_type="application/zip",
    )


# ═══════════════════════════════════════════════════════════════
# Démos pédagogiques (inchangées vs Phase 0)
# ═══════════════════════════════════════════════════════════════
class RNNDemoRequest(BaseModel):
    layers:        List[int] = Field([4])
    batch_size:    int   = Field(4,   ge=2, le=8)
    seq_length:    int   = Field(3,   ge=2, le=6)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    seed:          int   = Field(7)


@app.post("/api/rnn_demo/run")
def rnn_demo_run(req: RNNDemoRequest):
    try:
        pipeline = get_synthetic_pipeline()
        if not (1 <= len(req.layers) <= 3):
            raise ValueError("Entre 1 et 3 couches supportées")
        if not all(2 <= u <= 16 for u in req.layers):
            raise ValueError("Chaque couche doit avoir entre 2 et 16 neurones")
        return run_rnn_demo(pipeline=pipeline, layers=req.layers,
                            batch_size=req.batch_size, seq_length=req.seq_length,
                            learning_rate=req.learning_rate, seed=req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur démo RNN : {e}")


class LSTMDemoRequest(BaseModel):
    layers:        List[int] = Field([4])
    batch_size:    int   = Field(4,   ge=2, le=8)
    seq_length:    int   = Field(3,   ge=2, le=6)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    seed:          int   = Field(7)


@app.post("/api/lstm_demo/run")
def lstm_demo_run(req: LSTMDemoRequest):
    try:
        pipeline = get_synthetic_pipeline()
        if not (1 <= len(req.layers) <= 3):
            raise ValueError("Entre 1 et 3 couches supportées")
        if not all(2 <= u <= 16 for u in req.layers):
            raise ValueError("Chaque couche doit avoir entre 2 et 16 neurones")
        return run_lstm_demo(pipeline=pipeline, layers=req.layers,
                             batch_size=req.batch_size, seq_length=req.seq_length,
                             learning_rate=req.learning_rate, seed=req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur démo LSTM : {e}")


class TransformerDemoRequest(BaseModel):
    d_model:       int   = Field(8,  ge=4, le=32)
    n_heads:       int   = Field(2,  ge=1, le=4)
    d_ff:          int   = Field(16, ge=8, le=64)
    batch_size:    int   = Field(4,  ge=2, le=8)
    seq_length:    int   = Field(3,  ge=2, le=6)
    learning_rate: float = Field(0.01, gt=0.0, le=1.0)
    seed:          int   = Field(7)


@app.post("/api/transformer_demo/run")
def transformer_demo_run(req: TransformerDemoRequest):
    try:
        pipeline = get_synthetic_pipeline()
        if req.n_heads not in [1, 2, 4]:
            raise ValueError("Nombre de têtes : 1, 2 ou 4")
        if req.d_model % req.n_heads != 0:
            raise ValueError(f"d_model ({req.d_model}) doit être divisible par n_heads ({req.n_heads})")
        return run_transformer_demo(pipeline=pipeline, d_model=req.d_model,
                                    n_heads=req.n_heads, d_ff=req.d_ff,
                                    batch_size=req.batch_size, seq_length=req.seq_length,
                                    learning_rate=req.learning_rate, seed=req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur démo Transformer : {e}")


# ═══════════════════════════════════════════════════════════════
# WebSockets (training + ingestion / preparation)
# ═══════════════════════════════════════════════════════════════
@app.websocket("/ws/ingestion")
async def websocket_ingestion(websocket: WebSocket):
    await manager.connect_ingestion(websocket)
    try:
        await websocket.send_json({
            "type": "info",
            "message": "WS prêt — scaffolding préparation (étendu en Phase 2).",
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_ingestion()


@app.websocket("/ws/{experiment_id}")
async def websocket_training(websocket: WebSocket, experiment_id: int):
    await manager.connect_training(experiment_id, websocket)
    print(f"[WS] connecté exp={experiment_id}", flush=True)

    async def _pump():
        """Lit la queue et envoie au client — seul endroit qui appelle send_json."""
        q = manager.queues.get(experiment_id)
        if q is None:
            return
        while True:
            item = await q.get()
            if item is None:          # sentinel de fin
                break
            try:
                await websocket.send_json(item)
                print(f"[WS] → envoyé exp={experiment_id} type={item.get('type')}", flush=True)
            except Exception as e:
                print(f"[WS] ✗ échec send exp={experiment_id}: {type(e).__name__}: {e}", flush=True)
                break

    pump_task = asyncio.create_task(_pump())
    try:
        await websocket.send_json({
            "type": "info",
            "message": f"WS prêt pour exp {experiment_id} (live training updates).",
        })
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"[WS] client déconnecté exp={experiment_id}", flush=True)
                break
            except Exception as e:
                print(f"[WS] receive exception exp={experiment_id}: {type(e).__name__}: {e}", flush=True)
                break
    except Exception as e:
        print(f"[WS] endpoint exception exp={experiment_id}: {type(e).__name__}: {e}", flush=True)
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        manager.disconnect_training(experiment_id)
        print(f"[WS] fermé exp={experiment_id}", flush=True)
