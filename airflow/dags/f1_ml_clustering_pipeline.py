from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Definir rutas absolutas para asegurar la ejecución correcta en el entorno local
PROJECT_DIR = "/opt/airflow/project"
KEDRO_BIN = "kedro"

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'Luis Salamanca, Brahian Gonzalez',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

# ==============================================================================
# DEFINICIÓN DEL DAG
# ==============================================================================

with DAG(
    dag_id='f1_ml_clustering_pipeline',
    default_args=default_args,
    description='Pipeline de Clustering de Pilotos F1 (Mejor Modelo Validado)',
    schedule_interval='@weekly',  # Ejecutar semanalmente o bajo demanda
    catchup=False,
    tags=['f1', 'ml', 'clustering', 'produccion'],
) as dag:

    # --------------------------------------------------------------------------
    # Tarea 1: Preprocesamiento de Datos de Pilotos
    # --------------------------------------------------------------------------
    # Ejecuta el nodo que limpia, agrega y genera features (Ratios, Racecraft)
    preprocess_drivers = BashOperator(
        task_id='preprocess_drivers_data',
        bash_command=f"cd {PROJECT_DIR} && {KEDRO_BIN} run --pipeline=clustering_pilotos --nodes=preprocess_driver_data_node",
        doc_md="""
        ### Preprocesamiento de Pilotos
        - Carga `Race_Results.csv` y `Driver_Details.csv`.
        - Calcula métricas históricas: Win Rate, Podium Rate, DNF Rate.
        - Genera métricas de habilidad: Racecraft (posiciones ganadas).
        - Filtra pilotos inactivos o con pocas carreras.
        """
    )

    # --------------------------------------------------------------------------
    # Tarea 2: Entrenamiento del Modelo KMeans
    # --------------------------------------------------------------------------
    # Entrena el modelo usando RobustScaler y K=3 (Validado A/B)
    train_clustering = BashOperator(
        task_id='train_driver_clustering',
        bash_command=f"cd {PROJECT_DIR} && {KEDRO_BIN} run --pipeline=clustering_pilotos --nodes=train_driver_clustering_node",
        doc_md="""
        ### Entrenamiento de Clustering
        - Aplica `RobustScaler` para manejar outliers.
        - Entrena `KMeans` con K=3 (Élite, Midfield, Backmarkers).
        - Asigna etiquetas a cada piloto.
        - Guarda el modelo en `data/06_models`.
        """
    )

    # --------------------------------------------------------------------------
    # Tarea 3: Evaluación y Métricas
    # --------------------------------------------------------------------------
    # Genera reporte JSON con Silhouette Score e Indices de validación
    evaluate_clustering = BashOperator(
        task_id='evaluate_clustering_metrics',
        bash_command=f"cd {PROJECT_DIR} && {KEDRO_BIN} run --pipeline=clustering_pilotos --nodes=evaluate_driver_clustering_node",
        doc_md="""
        ### Evaluación del Modelo
        - Calcula Silhouette Score.
        - Calcula Calinski-Harabasz Index.
        - Genera perfiles promedio de cada cluster.
        - Exporta métricas a `data/08_reporting/driver_clustering_metrics.json`.
        """
    )

    # --------------------------------------------------------------------------
    # FLUJO DE EJECUCIÓN
    # --------------------------------------------------------------------------
    preprocess_drivers >> train_clustering >> evaluate_clustering

if __name__ == "__main__":
    dag.test()
