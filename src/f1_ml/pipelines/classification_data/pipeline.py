"""
Pipeline de preparación de datos de clasificación (predicción de podio)
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    load_raw_data_classification,
    merge_tables_classification,
    filter_valid_finishes,
    create_podium_target,
    drop_leakage_features_classification,
    sort_by_date,
    create_driver_historical_features,
    create_constructor_historical_features,
    create_qualifying_temporal_features,
    impute_missing_classification,
    split_temporal_train_test,
    apply_dual_scaling,
    apply_encoding,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # NODE 1: Cargar datos crudos
            node(
                func=load_raw_data_classification,
                inputs=[
                    "classification_raw_race_results",
                    "classification_raw_qualifying_results",
                    "classification_raw_races",
                    "classification_raw_drivers",
                    "classification_raw_constructors",
                    "classification_raw_circuits",
                ],
                outputs="classification_raw_data_dict",
                name="load_raw_data_classification",
            ),
            # NODE 2: Merge de tablas
            node(
                func=merge_tables_classification,
                inputs="classification_raw_data_dict",
                outputs="classification_merged_data",
                name="merge_tables_classification",
            ),
            # NODE 3: Filtrar registros válidos (DNFs)
            node(
                func=filter_valid_finishes,
                inputs="classification_merged_data",
                outputs="classification_filtered_data",
                name="filter_valid_finishes",
            ),
            # NODE 4: Crear variable target
            node(
                func=create_podium_target,
                inputs="classification_filtered_data",
                outputs="classification_data_with_target",
                name="create_podium_target",
            ),
            # NODE 5: Eliminar data leakage
            node(
                func=drop_leakage_features_classification,
                inputs=["classification_data_with_target", "params:classification_data"],
                outputs="classification_no_leakage",
                name="drop_leakage_features_classification",
            ),
            # NODE 6: Ordenar por fecha (CRÍTICO)
            node(
                func=sort_by_date,
                inputs="classification_no_leakage",
                outputs="classification_sorted_data",
                name="sort_by_date",
            ),
            # NODE 7: Features históricas piloto
            node(
                func=create_driver_historical_features,
                inputs="classification_sorted_data",
                outputs="classification_driver_features",
                name="create_driver_historical_features",
            ),
            # NODE 8: Features históricas constructor
            node(
                func=create_constructor_historical_features,
                inputs="classification_driver_features",
                outputs="classification_constructor_features",
                name="create_constructor_historical_features",
            ),
            # NODE 9: Features qualifying y temporales
            node(
                func=create_qualifying_temporal_features,
                inputs="classification_constructor_features",
                outputs="classification_all_features",
                name="create_qualifying_temporal_features",
            ),
            # NODE 10: Imputar valores faltantes
            node(
                func=impute_missing_classification,
                inputs="classification_all_features",
                outputs="classification_imputed_data",
                name="impute_missing_classification",
            ),
            # NODE 11: Split temporal train/test
            node(
                func=split_temporal_train_test,
                inputs=["classification_imputed_data", "params:classification_data"],
                outputs=[
                    "classification_X_train_raw",
                    "classification_X_test_raw",
                    "classification_y_train",
                    "classification_y_test",
                ],
                name="split_temporal_train_test",
            ),
            # NODE 12: Aplicar escalado dual
            node(
                func=apply_dual_scaling,
                inputs=[
                    "classification_X_train_raw",
                    "classification_X_test_raw",
                    "params:classification_data",
                ],
                outputs=[
                    "classification_X_train_scaled",
                    "classification_X_test_scaled",
                    "classification_scalers",
                ],
                name="apply_dual_scaling",
            ),
            # NODE 13: Aplicar encoding
            node(
                func=apply_encoding,
                inputs=[
                    "classification_X_train_scaled",
                    "classification_X_test_scaled",
                    "classification_y_train",
                    "params:classification_data",
                ],
                outputs=[
                    "classification_X_train",
                    "classification_X_test",
                    "classification_encoders",
                ],
                name="apply_encoding",
            ),
        ]
    )
