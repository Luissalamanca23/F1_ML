"""
Pipeline de Entrenamiento de Modelos para Regresión F1
=======================================================

Este pipeline conecta los 6 nodes de entrenamiento y optimización de modelos
para predecir la posición final en F1.

Autor: Pipeline F1 ML
Fecha: 2025-10-25
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    train_all_models,
    evaluate_models,
    select_top5_models,
    optimize_with_gridsearch,
    evaluate_optimized_models,
    save_final_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de entrenamiento de modelos.

    Flujo:
    1. train_all_models - Entrenar 11 modelos base
    2. evaluate_models - Evaluar y comparar todos
    3. select_top5_models - Seleccionar TOP 5
    4. optimize_with_gridsearch - GridSearch en TOP 5
    5. evaluate_optimized_models - Evaluar modelos optimizados
    6. save_final_model - Guardar mejor modelo final

    Returns:
        Pipeline de Kedro
    """
    return Pipeline(
        [
            # Node 1: Entrenar 11 modelos base
            node(
                func=train_all_models,
                inputs=[
                    "X_train_scaled",
                    "y_train",
                    "X_test_scaled",
                    "y_test",
                    "params:regresion_models",
                ],
                outputs="resultados_modelos_base",
                name="train_all_models_node",
            ),
            # Node 2: Evaluar modelos base
            node(
                func=evaluate_models,
                inputs="resultados_modelos_base",
                outputs="comparacion_modelos_base",
                name="evaluate_models_node",
            ),
            # Node 3: Seleccionar TOP 5
            node(
                func=select_top5_models,
                inputs=["resultados_modelos_base", "comparacion_modelos_base"],
                outputs="top5_modelos",
                name="select_top5_models_node",
            ),
            # Node 4: GridSearch en TOP 5
            node(
                func=optimize_with_gridsearch,
                inputs=[
                    "top5_modelos",
                    "X_train_scaled",
                    "y_train",
                    "X_test_scaled",
                    "y_test",
                    "params:regresion_models",
                ],
                outputs="resultados_optimizados",
                name="optimize_with_gridsearch_node",
            ),
            # Node 5: Evaluar modelos optimizados
            node(
                func=evaluate_optimized_models,
                inputs="resultados_optimizados",
                outputs="comparacion_optimizada",
                name="evaluate_optimized_models_node",
            ),
            # Node 6: Guardar mejor modelo final
            node(
                func=save_final_model,
                inputs=["resultados_optimizados", "comparacion_optimizada"],
                outputs=["modelo_final_regresion", "nombre_modelo_final", "metricas_finales"],
                name="save_final_model_node",
            ),
        ]
    )
