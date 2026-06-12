"""
base.py — Classe abstraite BasePipeline pour l'architecture en plugins.

Tous les pipelines de la plateforme (Cevital, Azure historique, futurs)
doivent hériter de BasePipeline et exposer le même contrat minimal :
identifiant, description, et hooks "ready" / "info".

Le but de cette abstraction : permettre à `main.py` et au registre de
référencer un pipeline par son ID (str) sans connaître ses spécificités
métier, et de pouvoir basculer / ajouter des pipelines facilement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePipeline(ABC):
    """Classe abstraite — contrat minimal d'un pipeline de la plateforme."""

    # ─── Identité (à surcharger par chaque sous-classe) ────────
    PIPELINE_ID: str = "base"
    PIPELINE_NAME: str = "Pipeline abstrait"
    PIPELINE_DESCRIPTION: str = "À surcharger dans la sous-classe."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}
        self.is_ready: bool = False
        self.phases_completed: list[str] = []

    # ─── Interface obligatoire ────────────────────────────────
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Retourne les métadonnées du pipeline (id, nom, description, état)."""
        raise NotImplementedError

    # ─── Helpers communs (non obligatoires à surcharger) ──────
    def _mark_phase(self, phase_name: str) -> None:
        """Marque une phase comme complétée dans l'ordre."""
        if phase_name not in self.phases_completed:
            self.phases_completed.append(phase_name)

    def reset(self) -> None:
        """Réinitialise l'état du pipeline (utile pour la fusion / réentraînement)."""
        self.is_ready = False
        self.phases_completed = []
