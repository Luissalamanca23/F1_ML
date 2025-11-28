"""
This is a boilerplate pipeline 'clustering_pilotos'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_driver_data, train_driver_clustering, evaluate_driver_clustering

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_driver_data,
            inputs=["race_results", "driver_details", "params:clustering_pilotos_options"],
            outputs="drivers_features_processed",
            name="preprocess_driver_data_node",
        ),
        node(
            func=train_driver_clustering,
            inputs=["drivers_features_processed", "params:clustering_pilotos_options"],
            outputs=["driver_clustering_model", "drivers_clustered"],
            name="train_driver_clustering_node",
        ),
        node(
            func=evaluate_driver_clustering,
            inputs=["drivers_clustered", "params:clustering_pilotos_options"],
            outputs="clustering_metrics",
            name="evaluate_driver_clustering_node",
        ),
    ])