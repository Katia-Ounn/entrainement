"""
registry.py — Registre des pipelines de la plateforme.

Mono-pipeline en Partie 2 : seul Cevital y figure (Azure a été retiré).
Pour ajouter un futur pipeline, créer la classe (héritant de BasePipeline)
et l'enregistrer ici.
"""
from typing import Dict, Type

from .base import BasePipeline
from .cevital_pipeline import CevitalPipeline

PIPELINE_REGISTRY: Dict[str, Type[BasePipeline]] = {
    CevitalPipeline.PIPELINE_ID: CevitalPipeline,
}


def get_pipeline_class(pipeline_id: str) -> Type[BasePipeline]:
    """Retourne la classe pipeline correspondant à l'ID (lève KeyError sinon)."""
    if pipeline_id not in PIPELINE_REGISTRY:
        raise KeyError(
            f"Pipeline inconnu : '{pipeline_id}'. "
            f"Disponibles : {list(PIPELINE_REGISTRY.keys())}"
        )
    return PIPELINE_REGISTRY[pipeline_id]


def list_pipelines() -> list[dict]:
    """Retourne la liste des pipelines disponibles avec leur identité."""
    return [
        {
            "id":          cls.PIPELINE_ID,
            "name":        cls.PIPELINE_NAME,
            "description": cls.PIPELINE_DESCRIPTION,
        }
        for cls in PIPELINE_REGISTRY.values()
    ]
