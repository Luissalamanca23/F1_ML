"""Pipeline de preprocesamiento de datos para F1 Machine Learning.

Este pipeline implementa el preprocesamiento completo de los datos de Formula 1,
incluyendo limpieza, feature engineering, imputación y creación de datasets
listos para modelado de regresión y clasificación.
"""

from kedro.pipeline import Node, Pipeline, node
from .nodes import (
    limpiar_resultados_carrera,
    limpiar_resultados_clasificacion,
    limpiar_tiempos_vuelta,
    procesar_detalles_piloto,
    imputar_valores_faltantes,
    calcular_estadisticas_vuelta,
    calcular_rendimiento_historico,
    calcular_estadisticas_constructor,
    calcular_rendimiento_circuito,
    crear_dataset_unificado,
    limpiar_dataset_final,
    codificar_variables_categoricas,
    crear_datasets_modelado,
    tratar_outliers_avanzado,
    imputar_valores_faltantes_avanzado,
    detectar_multicolinealidad,
    aplicar_validacion_temporal
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline completo de preprocesamiento de datos.

    Este pipeline procesa los datos raw de Formula 1 y genera datasets
    listos para entrenar modelos de regresión y clasificación.

    Returns:
        Pipeline: Pipeline completo de preprocesamiento
    """

    # Pipeline de limpieza inicial de datos
    cleaning_pipeline = Pipeline([
        # Limpieza de resultados de carreras
        node(
            func=limpiar_resultados_carrera,
            inputs="race_results",
            outputs="race_results_clean",
            name="limpiar_resultados_carrera_node",
            tags=["cleaning", "race_data"]
        ),

        # Limpieza de resultados de clasificación
        node(
            func=limpiar_resultados_clasificacion,
            inputs="qualifying_results",
            outputs="qualifying_results_clean",
            name="limpiar_resultados_clasificacion_node",
            tags=["cleaning", "qualifying_data"]
        ),

        # Limpieza de tiempos de vuelta
        node(
            func=limpiar_tiempos_vuelta,
            inputs="lap_timings",
            outputs="lap_timings_clean",
            name="limpiar_tiempos_vuelta_node",
            tags=["cleaning", "lap_data"]
        ),

        # Procesamiento de detalles de piloto
        node(
            func=procesar_detalles_piloto,
            inputs="driver_details",
            outputs="driver_details_processed",
            name="procesar_detalles_piloto_node",
            tags=["cleaning", "driver_data"]
        )
    ])

    # Pipeline de imputación de valores faltantes
    imputation_pipeline = Pipeline([
        # Imputación en datos de clasificación
        node(
            func=imputar_valores_faltantes,
            inputs="qualifying_results_clean",
            outputs="qualifying_results_imputed",
            name="imputar_valores_faltantes_node",
            tags=["imputation", "qualifying_data"]
        )
    ])

    # Pipeline de feature engineering
    feature_engineering_pipeline = Pipeline([
        # Calcular estadísticas de vuelta
        node(
            func=calcular_estadisticas_vuelta,
            inputs="lap_timings_clean",
            outputs="lap_statistics",
            name="calcular_estadisticas_vuelta_node",
            tags=["feature_engineering", "lap_stats"]
        ),

        # Calcular rendimiento histórico de pilotos
        node(
            func=calcular_rendimiento_historico,
            inputs=["race_results_clean", "race_schedule"],
            outputs="race_results_enhanced",
            name="calcular_rendimiento_historico_node",
            tags=["feature_engineering", "historical_performance"]
        ),

        # Calcular estadísticas de constructores
        node(
            func=calcular_estadisticas_constructor,
            inputs=["constructor_rankings", "race_schedule"],
            outputs="constructor_yearly_stats",
            name="calcular_estadisticas_constructor_node",
            tags=["feature_engineering", "constructor_stats"]
        ),

        # Calcular rendimiento por circuito
        node(
            func=calcular_rendimiento_circuito,
            inputs=["race_results_enhanced", "race_schedule"],
            outputs="race_results_circuit",
            name="calcular_rendimiento_circuito_node",
            tags=["feature_engineering", "circuit_performance"]
        )
    ])

    # Pipeline de unificación y limpieza final
    unification_pipeline = Pipeline([
        # Crear dataset maestro unificado
        node(
            func=crear_dataset_unificado,
            inputs=[
                "race_results_circuit",
                "lap_statistics",
                "qualifying_results_imputed",
                "driver_details_processed",
                "team_details",
                "race_schedule",
                "track_information",
                "constructor_yearly_stats"
            ],
            outputs="master_dataset_raw",
            name="crear_dataset_unificado_node",
            tags=["unification", "master_dataset"]
        ),

        # Limpieza final del dataset
        node(
            func=limpiar_dataset_final,
            inputs="master_dataset_raw",
            outputs="master_dataset_clean",
            name="limpiar_dataset_final_node",
            tags=["cleaning", "final_cleaning"]
        )
    ])

    # Pipeline de codificación y preparación para modelado
    modeling_preparation_pipeline = Pipeline([
        # Codificar variables categóricas
        node(
            func=codificar_variables_categoricas,
            inputs="master_dataset_clean",
            outputs="master_dataset_encoded",
            name="codificar_variables_categoricas_node",
            tags=["encoding", "categorical_variables"]
        ),

        # Crear datasets finales para modelado
        node(
            func=crear_datasets_modelado,
            inputs="master_dataset_encoded",
            outputs=["regression_dataset", "classification_dataset"],
            name="crear_datasets_modelado_node",
            tags=["modeling_preparation", "final_datasets"]
        )
    ])

    # Combinar todos los pipelines
    return (
        cleaning_pipeline +
        imputation_pipeline +
        feature_engineering_pipeline +
        unification_pipeline +
        modeling_preparation_pipeline
    )