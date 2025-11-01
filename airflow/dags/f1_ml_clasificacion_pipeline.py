"""
DAG de Apache Airflow para Pipeline de Clasificación F1 ML
===========================================================

Este DAG ejecuta el pipeline de clasificación de Kedro para predicción de podios.

Estructura:
- 1 tarea de verificacion de entorno
- 1 pipeline classification_data (13 nodos internos)
- 1 pipeline classification_models (7 nodos internos)
- 1 tarea de reporte final
Total: 4 tareas secuenciales

Autor: Pipeline F1 ML con Apache Airflow
Fecha: 2025-11-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Configuracion por defecto del DAG
default_args = {
    'owner': 'f1_ml_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Definicion del DAG
dag = DAG(
    dag_id='f1_ml_clasificacion',
    default_args=default_args,
    description='Pipeline F1 ML - Clasificacion de Podios',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['f1', 'machine-learning', 'kedro', 'clasificacion'],
)

# ============================================================================
# TAREA 01: VERIFICAR ENTORNO Y DATOS
# ============================================================================

verificar_entorno = BashOperator(
    task_id='01_verificar_entorno',
    bash_command="""
        echo "=========================================="
        echo "VERIFICANDO ENTORNO Y DATOS - CLASIFICACION"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Verificando datos de entrada..."
        ls -lh data/01_raw/*.csv || echo "ADVERTENCIA: No se encontraron CSVs"

        echo "[INFO] Verificando directorios..."
        mkdir -p data/02_intermediate
        mkdir -p data/03_primary
        mkdir -p data/04_feature
        mkdir -p data/05_model_input
        mkdir -p data/06_models
        mkdir -p data/07_model_output

        echo "[INFO] Sistema listo para ejecutar pipeline de clasificacion"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# PIPELINE CLASSIFICATION_DATA - Procesamiento de Datos para Clasificación
# ============================================================================

ejecutar_pipeline_classification_data = BashOperator(
    task_id='02_pipeline_classification_data',
    bash_command="""
        set -e
        echo "=========================================="
        echo "EJECUTANDO: Pipeline Classification Data"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando pipeline de procesamiento de datos para clasificacion..."
        echo "[INFO] Este pipeline incluye 13 nodos:"
        echo "  1. Cargar datos crudos (6 CSVs)"
        echo "  2. Merge de tablas relacionales"
        echo "  3. Filtrar registros validos (DNFs)"
        echo "  4. Crear target is_podium"
        echo "  5. Eliminar features con data leakage"
        echo "  6. Ordenar por fecha (prevenir leakage temporal)"
        echo "  7. Features historicas piloto (rolling windows)"
        echo "  8. Features historicas constructor"
        echo "  9. Features qualifying y temporales"
        echo "  10. Imputar valores faltantes"
        echo "  11. Split temporal (train < 2023, test >= 2023)"
        echo "  12. Escalado dual (MinMax + Standard)"
        echo "  13. Encoding (Target + OneHot)"
        echo ""
        echo "[INFO] Ejecutando pipeline completo..."
        echo "------------------------------------------"

        kedro run --pipeline=classification_data

        echo "------------------------------------------"
        echo "[OK] Pipeline classification_data completado exitosamente"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# PIPELINE CLASSIFICATION_MODELS - Entrenamiento de Modelos de Clasificación
# ============================================================================

ejecutar_pipeline_classification_models = BashOperator(
    task_id='03_pipeline_classification_models',
    bash_command="""
        set -e
        echo "=========================================="
        echo "EJECUTANDO: Pipeline Classification Models"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando pipeline de entrenamiento de modelos de clasificacion..."
        echo "[INFO] Este pipeline incluye 7 nodos:"
        echo "  1. Aplicar SMOTE (balanceo de clases)"
        echo "  2. Entrenar 6 modelos base"
        echo "  3. Evaluar modelos base (F1, Precision, Recall, ROC-AUC)"
        echo "  4. Seleccionar TOP 5 modelos"
        echo "  5. Optimizar con GridSearch (esto tomara tiempo)"
        echo "  6. Evaluar modelos optimizados"
        echo "  7. Guardar modelo final"
        echo ""
        echo "[INFO] ADVERTENCIA: GridSearch puede tomar 30-60 minutos"
        echo "[INFO] Ejecutando pipeline completo..."
        echo "------------------------------------------"

        kedro run --pipeline=classification_models

        echo "------------------------------------------"
        echo "[OK] Pipeline classification_models completado exitosamente"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# TAREA FINAL: REPORTE DE EJECUCION
# ============================================================================

generar_reporte = BashOperator(
    task_id='04_generar_reporte_final',
    bash_command="""
        echo "=========================================="
        echo "REPORTE FINAL - CLASIFICACION"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Verificando archivos generados..."
        echo ""

        if [ -f "data/07_model_output/classification_comparacion_optimizada.csv" ]; then
            echo "[OK] Comparacion de modelos generada"
            cat data/07_model_output/classification_comparacion_optimizada.csv
        else
            echo "[ERROR] classification_comparacion_optimizada.csv NO encontrado"
        fi

        echo ""
        if [ -f "data/06_models/classification_modelo_final.pkl" ]; then
            echo "[OK] Modelo final guardado"
            ls -lh data/06_models/classification_modelo_final.pkl
        else
            echo "[ERROR] classification_modelo_final.pkl NO encontrado"
        fi

        echo ""
        if [ -f "data/07_model_output/classification_metricas_finales.json" ]; then
            echo "[OK] Metricas finales"
            cat data/07_model_output/classification_metricas_finales.json
        fi

        echo ""
        echo "=========================================="
        echo "[OK] PIPELINE CLASIFICACION COMPLETADO"
        echo "Fecha finalizacion: $(date)"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# DEFINICION DE DEPENDENCIAS (FLUJO SECUENCIAL)
# ============================================================================

# Flujo completo del DAG (4 tareas):
# 1. Verificar entorno -> 2. Pipeline Data -> 3. Pipeline Models -> 4. Reporte

verificar_entorno >> ejecutar_pipeline_classification_data >> ejecutar_pipeline_classification_models >> generar_reporte
