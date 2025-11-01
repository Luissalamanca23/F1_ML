"""
Pipeline de modelos de clasificación (predicción de podio)
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    apply_smote_balancing,
    train_classification_models_base,
    evaluate_classification_models,
    select_top5_classification,
    optimize_classification_gridsearch,
    evaluate_optimized_classification,
    save_best_classification_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # NODE 1: Aplicar SMOTE
            node(
                func=apply_smote_balancing,
                inputs=[
                    "classification_X_train",
                    "classification_y_train",
                    "params:classification_models",
                ],
                outputs=[
                    "classification_X_train_smote",
                    "classification_y_train_smote",
                ],
                name="apply_smote_balancing",
            ),
            # NODE 2: Entrenar modelos base
            node(
                func=train_classification_models_base,
                inputs=[
                    "classification_X_train_smote",
                    "classification_y_train_smote",
                    "params:classification_models",
                ],
                outputs="classification_models_base",
                name="train_classification_models_base",
            ),
            # NODE 3: Evaluar modelos base
            node(
                func=evaluate_classification_models,
                inputs=[
                    "classification_models_base",
                    "classification_X_test",
                    "classification_y_test",
                ],
                outputs="classification_comparacion_modelos_base",
                name="evaluate_classification_models",
            ),
            # NODE 4: Seleccionar TOP 5
            node(
                func=select_top5_classification,
                inputs=[
                    "classification_comparacion_modelos_base",
                    "classification_models_base",
                ],
                outputs="classification_top5_dict",
                name="select_top5_classification",
            ),
            # NODE 5: Optimizar con GridSearch
            node(
                func=optimize_classification_gridsearch,
                inputs=[
                    "classification_top5_dict",
                    "classification_X_train_smote",
                    "classification_y_train_smote",
                    "params:classification_models",
                ],
                outputs="classification_optimized_dict",
                name="optimize_classification_gridsearch",
            ),
            # NODE 6: Evaluar modelos optimizados
            node(
                func=evaluate_optimized_classification,
                inputs=[
                    "classification_optimized_dict",
                    "classification_X_test",
                    "classification_y_test",
                ],
                outputs="classification_comparacion_optimizada",
                name="evaluate_optimized_classification",
            ),
            # NODE 7: Guardar mejor modelo
            node(
                func=save_best_classification_model,
                inputs=[
                    "classification_optimized_dict",
                    "classification_comparacion_optimizada",
                    "classification_X_test",
                    "classification_y_test",
                ],
                outputs=[
                    "classification_modelo_final",
                    "classification_nombre_modelo_final",
                    "classification_metricas_finales",
                ],
                name="save_best_classification_model",
            ),
        ]
    )
