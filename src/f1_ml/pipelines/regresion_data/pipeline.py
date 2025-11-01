"""
Pipeline de Preparación de Datos para Regresión F1
===================================================

Este pipeline conecta los 11 nodes de transformación de datos desde
los CSVs crudos hasta el dataset final escalado listo para modelado.

Autor: Pipeline F1 ML
Fecha: 2025-10-25
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    load_raw_data,
    merge_tables,
    filter_modern_era,
    remove_dnfs,
    drop_leakage_features,
    impute_missing_values,
    split_train_test,
    create_advanced_features,
    apply_target_encoding,
    apply_onehot_encoding,
    apply_scaling,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de preparación de datos para regresión.

    Flujo:
    1. load_raw_data - Cargar 6 CSVs desde 01_raw
    2. merge_tables - Merge de tablas relacionales
    3. filter_modern_era - Filtrar 2010-2024
    4. remove_dnfs - Eliminar DNFs
    5. drop_leakage_features - Eliminar 16 variables prohibidas
    6. impute_missing_values - Imputar NaNs
    7. split_train_test - Split 80/20
    8. create_advanced_features - Feature Engineering (9 features)
    9. apply_target_encoding - Target Encoding para IDs
    10. apply_onehot_encoding - One-Hot Encoding para texto
    11. apply_scaling - StandardScaler

    Returns:
        Pipeline de Kedro
    """
    return Pipeline(
        [
            # Node 1: Cargar datos crudos
            node(
                func=load_raw_data,
                inputs=[
                    "race_results",
                    "qualifying_results",
                    "driver_rankings",
                    "constructor_rankings",
                    "track_information",
                    "race_schedule",
                ],
                outputs="raw_tables_dict",
                name="load_raw_data_node",
            ),
            # Node 2: Merge de tablas
            node(
                func=merge_tables,
                inputs="raw_tables_dict",
                outputs="df_merged",
                name="merge_tables_node",
            ),
            # Node 3: Filtrar era moderna (2010-2024)
            node(
                func=filter_modern_era,
                inputs=["df_merged", "params:regresion_data"],
                outputs="df_modern_era",
                name="filter_modern_era_node",
            ),
            # Node 4: Eliminar DNFs
            node(
                func=remove_dnfs,
                inputs="df_modern_era",
                outputs="df_no_dnfs",
                name="remove_dnfs_node",
            ),
            # Node 5: Eliminar variables con data leakage
            node(
                func=drop_leakage_features,
                inputs=["df_no_dnfs", "params:regresion_data"],
                outputs="df_no_leakage",
                name="drop_leakage_features_node",
            ),
            # Node 6: Imputar valores faltantes
            node(
                func=impute_missing_values,
                inputs="df_no_leakage",
                outputs="df_imputed",
                name="impute_missing_values_node",
            ),
            # Node 7: Split train/test
            node(
                func=split_train_test,
                inputs=["df_imputed", "params:regresion_data"],
                outputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
                name="split_train_test_node",
            ),
            # Node 8: Feature Engineering (9 features avanzadas)
            node(
                func=create_advanced_features,
                inputs=["X_train_raw", "X_test_raw"],
                outputs=["X_train_fe", "X_test_fe"],
                name="create_advanced_features_node",
            ),
            # Node 9: Target Encoding para IDs
            node(
                func=apply_target_encoding,
                inputs=["X_train_fe", "X_test_fe", "y_train"],
                outputs=["X_train_target_enc", "X_test_target_enc", "target_encoder"],
                name="apply_target_encoding_node",
            ),
            # Node 10: One-Hot Encoding para texto
            node(
                func=apply_onehot_encoding,
                inputs=["X_train_target_enc", "X_test_target_enc"],
                outputs=["X_train_onehot", "X_test_onehot"],
                name="apply_onehot_encoding_node",
            ),
            # Node 11: StandardScaler
            node(
                func=apply_scaling,
                inputs=["X_train_onehot", "X_test_onehot"],
                outputs=["X_train_scaled", "X_test_scaled", "scaler"],
                name="apply_scaling_node",
            ),
        ]
    )
