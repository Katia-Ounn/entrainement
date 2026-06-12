"""
models.py — Schéma BDD Cevital (SQLAlchemy + SQLite).

Phase 1 — Schéma final Cevital :

  Dataset
  ┣━━ id, name, folder_path, failure_path, equipment_path, v1_path
  ┣━━ stats (n_rows, n_failures, n_maintenances, n_composants, period_*)
  ┣━━ status ∈ {uploaded, features_done, preprocessed}
  ┗━━ created_at, metadata_json

  Experiment (référence un Dataset)
  ┣━━ id, name, architecture (LSTM|GRU), mode (manual|auto)
  ┣━━ status, created_at, completed_at, duration_sec
  ┣━━ dataset_id → Dataset.id
  ┣━━ hyperparams JSON (lookback, current_max_rul, embedding_dim, units, …)
  ┣━━ Régression : r2, mae, rmse, mape
  ┣━━ Classification (seuil=10j) : accuracy, precision, recall, f1_score
  ┣━━ training_history JSON (loss/val_loss/mae/val_mae par époque)
  ┣━━ AutoML : total_trials, best_trial_id
  ┗━━ model_dir, notes, error_message

⚠️ Les champs Azure (machine_id, mae_hours, scaler_x_path, scaler_y_path)
   ont été retirés. Phase 0 a effacé pdm_experiments.db — la BDD sera
   recréée vierge avec ce schéma au premier démarrage.
"""
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, JSON, DateTime, Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./pdm_experiments.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrainingStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class DatasetStatus:
    UPLOADED      = "uploaded"        # CSV chargés, EDA pas encore fait
    FEATURES_DONE = "features_done"   # Dataset_V1 généré
    PREPROCESSED  = "preprocessed"    # Tenseurs prêts pour l'entraînement


# ═══════════════════════════════════════════════════════════════
# DATASET — un dataset = 2 CSV uploadés (failure + equipment)
# ═══════════════════════════════════════════════════════════════
class Dataset(Base):
    __tablename__ = "datasets"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(200), unique=True, nullable=False)

    # Chemins disque
    folder_path     = Column(String(500), nullable=False)
    failure_path    = Column(String(500), nullable=True)
    equipment_path  = Column(String(500), nullable=True)
    v1_path         = Column(String(500), nullable=True)   # Dataset_V1 généré

    # Stats remplies après chaque phase
    n_rows          = Column(Integer, default=0)
    n_failures      = Column(Integer, default=0)
    n_maintenances  = Column(Integer, default=0)
    n_composants    = Column(Integer, default=0)
    n_machines      = Column(Integer, default=0)
    period_start    = Column(String(32), nullable=True)
    period_end      = Column(String(32), nullable=True)

    # Préprocessing config (lookback/current_max_rul/weight_factor) une fois
    # `prepare_sequences()` lancé. Permet à la card "Dataset" du
    # TrainingPanel de l'afficher en lecture seule.
    preproc_config  = Column(JSON, nullable=True)

    status          = Column(String(20), default=DatasetStatus.UPLOADED)
    created_at      = Column(DateTime, default=datetime.utcnow)
    metadata_json   = Column(JSON, nullable=True)

    experiments = relationship(
        "Experiment", back_populates="dataset", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "name":             self.name,
            "folder_path":      self.folder_path,
            "failure_path":     self.failure_path,
            "equipment_path":   self.equipment_path,
            "v1_path":          self.v1_path,
            "n_rows":           self.n_rows,
            "n_failures":       self.n_failures,
            "n_maintenances":   self.n_maintenances,
            "n_composants":     self.n_composants,
            "n_machines":       self.n_machines,
            "period_start":     self.period_start,
            "period_end":       self.period_end,
            "preproc_config":   self.preproc_config,
            "status":           self.status,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "metadata_json":    self.metadata_json,
        }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT — un entraînement (manuel ou AutoML) sur un dataset
# ═══════════════════════════════════════════════════════════════
class Experiment(Base):
    __tablename__ = "experiments"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(200), nullable=False)
    architecture    = Column(String(50), nullable=False)        # "LSTM" | "GRU"
    mode            = Column(String(20), nullable=False)        # "manual" | "auto"
    status          = Column(String(20), default=TrainingStatus.PENDING)

    created_at      = Column(DateTime, default=datetime.utcnow)
    completed_at    = Column(DateTime, nullable=True)
    duration_sec    = Column(Float, nullable=True)

    # Lien Dataset
    dataset_id      = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    dataset         = relationship("Dataset", back_populates="experiments")

    # Hyperparamètres complets (sérialisés en JSON pour le re-train)
    # Contenu attendu :
    #   {
    #     "lookback":         30,
    #     "current_max_rul":  30,
    #     "embedding_dim":    8,
    #     "num_layers":       2,
    #     "units":            [128, 64],
    #     "dropout_rates":    [0.2, 0.15],
    #     "learning_rate":    0.001,
    #     "batch_size":       32,
    #     "epochs":           60,
    #     "patience":         7,
    #     "weight_factor":    15.0,
    #     "feature_cols":     [...],
    #   }
    hyperparams     = Column(JSON, nullable=True)
    search_space    = Column(JSON, nullable=True)   # AutoML uniquement

    # ─── Métriques régression ────────────────────────────────
    r2              = Column(Float, nullable=True)
    mae             = Column(Float, nullable=True)   # en jours
    rmse            = Column(Float, nullable=True)   # en jours
    mape            = Column(Float, nullable=True)   # en %

    # ─── Métriques classification (seuil=10 jours par défaut) ─
    accuracy        = Column(Float, nullable=True)
    precision       = Column(Float, nullable=True)
    recall          = Column(Float, nullable=True)
    f1_score        = Column(Float, nullable=True)

    # ─── Historique d'apprentissage (Loss/MAE par époque) ──
    training_history = Column(JSON, nullable=True)

    # ─── AutoML state ────────────────────────────────────────
    total_trials    = Column(Integer, nullable=True)
    best_trial_id   = Column(Integer, nullable=True)

    # ─── Fichiers sauvegardés ────────────────────────────────
    # Pointe vers backend/models/{exp_id}/ qui contient :
    #   model.keras, scaler_x.pkl, scaler_y.pkl,
    #   comp_mapping.json, config.json, metrics.json, predictions.csv
    model_dir       = Column(String(500), nullable=True)

    notes           = Column(Text, nullable=True)
    error_message   = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "name":             self.name,
            "architecture":     self.architecture,
            "mode":             self.mode,
            "status":           self.status,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "completed_at":     self.completed_at.isoformat() if self.completed_at else None,
            "duration_sec":     self.duration_sec,
            "dataset_id":       self.dataset_id,
            "dataset_name":     self.dataset.name if self.dataset else None,
            "hyperparams":      self.hyperparams,
            "search_space":     self.search_space,
            "r2":               self.r2,
            "mae":              self.mae,
            "rmse":             self.rmse,
            "mape":             self.mape,
            "accuracy":         self.accuracy,
            "precision":        self.precision,
            "recall":           self.recall,
            "f1_score":         self.f1_score,
            "training_history": self.training_history,
            "total_trials":     self.total_trials,
            "best_trial_id":    self.best_trial_id,
            "model_dir":        self.model_dir,
            "notes":            self.notes,
            "error_message":    self.error_message,
        }


# ═══════════════════════════════════════════════════════════════
# Init / session helpers
# ═══════════════════════════════════════════════════════════════
def init_db():
    """Crée toutes les tables si absentes (Dataset + Experiment)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
