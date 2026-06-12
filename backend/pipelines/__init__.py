"""
Backend pipelines package.

Contient l'architecture en plugins :
- base.py       : classe abstraite BasePipeline
- cevital_pipeline.py : pipeline GMAO Cevital (squelette en Phase 0, rempli en Phase 1)
- registry.py   : dictionnaire des pipelines disponibles
"""
from .base import BasePipeline
from .cevital_pipeline import CevitalPipeline
from .registry import PIPELINE_REGISTRY

__all__ = ["BasePipeline", "CevitalPipeline", "PIPELINE_REGISTRY"]
