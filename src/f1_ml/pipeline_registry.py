"""Project pipelines registry.

This module contains the pipeline registry for the F1 ML project.
It imports and registers all available pipelines so they can be executed by Kedro.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from f1_ml.pipelines import regresion_data
from f1_ml.pipelines import regresion_models
from f1_ml.pipelines import classification_data
from f1_ml.pipelines import classification_models
from f1_ml.pipelines import clustering_pilotos


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Pipelines de regresión
    regresion_data_pipeline = regresion_data.create_pipeline()
    regresion_models_pipeline = regresion_models.create_pipeline()

    # Pipelines de clasificación
    classification_data_pipeline = classification_data.create_pipeline()
    classification_models_pipeline = classification_models.create_pipeline()

    # Pipeline de clustering
    clustering_pilotos_pipeline = clustering_pilotos.create_pipeline()

    return {
        # Pipeline completo por defecto: ejecuta regresión + clasificación + clustering
        "__default__": (
            regresion_data_pipeline + regresion_models_pipeline +
            classification_data_pipeline + classification_models_pipeline +
            clustering_pilotos_pipeline
        ),

        # Pipeline completo de regresión
        "regresion": regresion_data_pipeline + regresion_models_pipeline,

        # Pipeline completo de clasificación
        "classification": classification_data_pipeline + classification_models_pipeline,
        
        # Pipeline de clustering
        "clustering_pilotos": clustering_pilotos_pipeline,

        # Pipelines individuales de regresión
        "regresion_data": regresion_data_pipeline,
        "regresion_models": regresion_models_pipeline,

        # Pipelines individuales de clasificación
        "classification_data": classification_data_pipeline,
        "classification_models": classification_models_pipeline,

        # Alias legacy
        "data": regresion_data_pipeline,
        "models": regresion_models_pipeline,
    }