"""
DAG de Apache Airflow para Pipeline de Regresión F1 ML
=======================================================

Este DAG ejecuta el pipeline de regresión de Kedro para predicción de posiciones.

Estructura:
- 1 tarea de verificacion de entorno
- 1 pipeline regresion_data (11 nodos internos)
- 1 pipeline regresion_models (6 nodos internos)
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
    dag_id='f1_ml_regresion',
    default_args=default_args,
    description='Pipeline F1 ML - Regresion de Posiciones',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['f1', 'machine-learning', 'kedro', 'regresion'],
)

# ============================================================================
# TAREA 01: VERIFICAR ENTORNO Y DATOS
# ============================================================================

verificar_entorno = BashOperator(
    task_id='01_verificar_entorno',
    bash_command="""
        echo "=========================================="
        echo "VERIFICANDO ENTORNO Y DATOS - REGRESION"
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

        echo "[INFO] Sistema listo para ejecutar pipeline de regresion"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# PIPELINE REGRESION_DATA - Procesamiento de Datos
# ============================================================================

ejecutar_pipeline_data = BashOperator(
    task_id='02_pipeline_regresion_data',
    bash_command="""
        set -e
        echo "=========================================="
        echo "EJECUTANDO: Pipeline Regresion Data"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando pipeline de procesamiento de datos..."
        echo "[INFO] Este pipeline incluye 11 nodos:"
        echo "  1. Cargar datos crudos (6 CSVs)"
        echo "  2. Merge de tablas relacionales"
        echo "  3. Filtrar era moderna (2010-2024)"
        echo "  4. Eliminar DNFs"
        echo "  5. Eliminar features con data leakage"
        echo "  6. Imputar valores faltantes"
        echo "  7. Split train/test (80/20)"
        echo "  8. Feature Engineering"
        echo "  9. Target Encoding"
        echo "  10. One-Hot Encoding"
        echo "  11. Escalado con StandardScaler"
        echo ""
        echo "[INFO] Ejecutando pipeline completo..."
        echo "------------------------------------------"

        kedro run --pipeline=regresion_data

        echo "------------------------------------------"
        echo "[OK] Pipeline regresion_data completado exitosamente"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# PIPELINE REGRESION_MODELS - Entrenamiento de Modelos
# ============================================================================

ejecutar_pipeline_models = BashOperator(
    task_id='03_pipeline_regresion_models',
    bash_command="""
        set -e
        echo "=========================================="
        echo "EJECUTANDO: Pipeline Regresion Models"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando pipeline de entrenamiento de modelos..."
        echo "[INFO] Este pipeline incluye 6 nodos:"
        echo "  1. Entrenar 11 modelos base"
        echo "  2. Evaluar modelos base"
        echo "  3. Seleccionar TOP 5 modelos"
        echo "  4. Optimizar con GridSearch (esto tomara tiempo)"
        echo "  5. Evaluar modelos optimizados"
        echo "  6. Guardar modelo final"
        echo ""
        echo "[INFO] ADVERTENCIA: GridSearch puede tomar 30-60 minutos"
        echo "[INFO] Ejecutando pipeline completo..."
        echo "------------------------------------------"

        kedro run --pipeline=regresion_models

        echo "------------------------------------------"
        echo "[OK] Pipeline regresion_models completado exitosamente"
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
        echo "REPORTE FINAL - REGRESION"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Verificando archivos generados..."
        echo ""

        if [ -f "data/07_model_output/regresion_comparacion_optimizada.csv" ]; then
            echo "[OK] Comparacion de modelos generada"
            cat data/07_model_output/regresion_comparacion_optimizada.csv
        else
            echo "[ERROR] regresion_comparacion_optimizada.csv NO encontrado"
        fi

        echo ""
        if [ -f "data/06_models/regresion_modelo_final.pkl" ]; then
            echo "[OK] Modelo final guardado"
            ls -lh data/06_models/regresion_modelo_final.pkl
        else
            echo "[ERROR] regresion_modelo_final.pkl NO encontrado"
        fi

        echo ""
        if [ -f "data/07_model_output/regresion_metricas_finales.json" ]; then
            echo "[OK] Metricas finales"
            cat data/07_model_output/regresion_metricas_finales.json
        fi

        echo ""
        echo "=========================================="
        echo "[OK] PIPELINE REGRESION COMPLETADO"
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

verificar_entorno >> ejecutar_pipeline_data >> ejecutar_pipeline_models >> generar_reporte
