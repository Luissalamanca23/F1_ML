"""
DAG Unificado de Apache Airflow para Pipeline F1 ML Completo
==============================================================

Este DAG ejecuta ambos pipelines (Regresión y Clasificación) de forma
consolidada, con paralelización de tareas independientes.

Estructura del DAG:
┌─────────────────────────┐
│ 01. Verificar Entorno   │
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼──────────┐  ┌──▼──────────────┐
│ Regresion    │  │ Clasificacion   │
│ Data Prep    │  │ Data Prep       │
└───┬──────────┘  └──┬──────────────┘
    │                │
┌───▼──────────┐  ┌──▼──────────────┐
│ Regresion    │  │ Clasificacion   │
│ Models       │  │ Models          │
└───┬──────────┘  └──┬──────────────┘
    │                │
    └────────┬───────┘
             │
    ┌────────▼────────────┐
    │ Consolidar          │
    │ Resultados          │
    └─────────────────────┘

Total: 6 tareas (2 en paralelo en cada fase)

Autor: F1 ML Team
Fecha: 2025-11-13
Versión: 2.0 (Unified Pipeline)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import json
import os

# ============================================================================
# CONFIGURACION DEL DAG
# ============================================================================

default_args = {
    'owner': 'f1_ml_team',
    'depends_on_past': False,
    'email_on_failure': False,  # Cambiar a True y configurar email en producción
    'email_on_retry': False,
    'email': ['f1ml-team@example.com'],  # Configurar email real
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=3),  # Tiempo total para ambos pipelines
}

dag = DAG(
    dag_id='f1_ml_unified_pipeline',
    default_args=default_args,
    description='Pipeline F1 ML Unificado - Regresión + Clasificación con consolidación',
    schedule_interval=None,  # Cambiar a cron para ejecución programada
    start_date=days_ago(1),
    catchup=False,
    tags=['f1', 'machine-learning', 'kedro', 'unified', 'production'],
    max_active_runs=1,  # Solo una ejecución a la vez
)

# ============================================================================
# TAREA 01: VERIFICAR ENTORNO Y DATOS
# ============================================================================

verificar_entorno = BashOperator(
    task_id='01_verificar_entorno',
    bash_command="""
        set -e
        echo "=========================================="
        echo "VERIFICANDO ENTORNO Y DATOS - PIPELINE UNIFICADO"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Verificando datos de entrada..."
        if [ ! -d "data/01_raw" ]; then
            echo "[ERROR] Directorio data/01_raw no existe"
            exit 1
        fi

        CSV_COUNT=$(ls data/01_raw/*.csv 2>/dev/null | wc -l)
        echo "[INFO] Archivos CSV encontrados: $CSV_COUNT"

        if [ "$CSV_COUNT" -lt 6 ]; then
            echo "[ERROR] Se esperan al menos 6 archivos CSV"
            exit 1
        fi

        echo "[INFO] CSVs detectados:"
        ls -lh data/01_raw/*.csv

        echo "[INFO] Verificando y creando directorios necesarios..."
        mkdir -p data/02_intermediate
        mkdir -p data/03_primary
        mkdir -p data/04_feature
        mkdir -p data/05_model_input
        mkdir -p data/06_models
        mkdir -p data/07_model_output
        mkdir -p data/08_reporting

        echo "[INFO] Verificando espacio en disco..."
        df -h /opt/airflow/project

        echo "[SUCCESS] Sistema listo para ejecutar ambos pipelines"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# TAREA 02: PIPELINE REGRESION_DATA
# ============================================================================

ejecutar_regresion_data = BashOperator(
    task_id='02_regresion_data_preparation',
    bash_command="""
        set -e
        echo "=========================================="
        echo "PIPELINE: Regresion Data Preparation"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando pipeline de regresión (11 nodos)..."
        kedro run --pipeline=regresion_data --runner=SequentialRunner

        echo "[INFO] Verificando outputs generados..."
        if [ ! -f "data/04_feature/X_train_scaled_regresion.csv" ]; then
            echo "[ERROR] No se generaron features de regresión"
            exit 1
        fi

        echo "[SUCCESS] Pipeline regresion_data completado"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# TAREA 03: PIPELINE CLASSIFICATION_DATA (En paralelo con regresion_data)
# ============================================================================

ejecutar_classification_data = BashOperator(
    task_id='03_classification_data_preparation',
    bash_command="""
        set -e
        echo "=========================================="
        echo "PIPELINE: Classification Data Preparation"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando pipeline de clasificación (13 nodos)..."
        kedro run --pipeline=classification_data --runner=SequentialRunner

        echo "[INFO] Verificando outputs generados..."
        if [ ! -f "data/04_feature/X_train_scaled_classification.csv" ]; then
            echo "[ERROR] No se generaron features de clasificación"
            exit 1
        fi

        echo "[SUCCESS] Pipeline classification_data completado"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# TAREA 04: PIPELINE REGRESION_MODELS
# ============================================================================

ejecutar_regresion_models = BashOperator(
    task_id='04_regresion_models_training',
    bash_command="""
        set -e
        echo "=========================================="
        echo "PIPELINE: Regresion Models Training"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando entrenamiento de modelos de regresión (6 nodos)..."
        echo "[INFO] - Entrenar 11 modelos base"
        echo "[INFO] - Seleccionar TOP 5"
        echo "[INFO] - GridSearch con CV=5"
        echo "[INFO] - Evaluar y guardar mejor modelo"

        kedro run --pipeline=regresion_models --runner=SequentialRunner

        echo "[INFO] Verificando modelo generado..."
        if [ ! -f "data/06_models/regresion_best_model.pkl" ]; then
            echo "[ERROR] No se generó modelo de regresión"
            exit 1
        fi

        echo "[SUCCESS] Pipeline regresion_models completado"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# TAREA 05: PIPELINE CLASSIFICATION_MODELS (En paralelo con regresion_models)
# ============================================================================

ejecutar_classification_models = BashOperator(
    task_id='05_classification_models_training',
    bash_command="""
        set -e
        echo "=========================================="
        echo "PIPELINE: Classification Models Training"
        echo "=========================================="
        cd /opt/airflow/project

        echo "[INFO] Iniciando entrenamiento de modelos de clasificación (7 nodos)..."
        echo "[INFO] - Aplicar SMOTE"
        echo "[INFO] - Entrenar 6 modelos base"
        echo "[INFO] - Seleccionar TOP 5"
        echo "[INFO] - GridSearch con CV=5"
        echo "[INFO] - Evaluar y guardar mejor modelo"

        kedro run --pipeline=classification_models --runner=SequentialRunner

        echo "[INFO] Verificando modelo generado..."
        if [ ! -f "data/06_models/classification_best_model.pkl" ]; then
            echo "[ERROR] No se generó modelo de clasificación"
            exit 1
        fi

        echo "[SUCCESS] Pipeline classification_models completado"
        echo "=========================================="
    """,
    dag=dag,
)

# ============================================================================
# TAREA 06: CONSOLIDAR RESULTADOS
# ============================================================================

def consolidar_resultados(**context):
    """
    Consolida los resultados de ambos pipelines en un único reporte.

    Lee las métricas de regresión y clasificación, las combina en un
    reporte unificado y genera un resumen comparativo.
    """
    import json
    import os
    from datetime import datetime

    print("=" * 70)
    print("CONSOLIDANDO RESULTADOS DE AMBOS PIPELINES")
    print("=" * 70)

    base_path = "/opt/airflow/project/data/07_model_output"

    # Inicializar reporte consolidado
    consolidated_report = {
        "timestamp": datetime.now().isoformat(),
        "dag_run_id": context.get('dag_run').run_id,
        "execution_date": str(context.get('execution_date')),
        "pipelines": {}
    }

    # ========================================================================
    # LEER MÉTRICAS DE REGRESIÓN
    # ========================================================================

    regresion_metrics_file = os.path.join(base_path, "regresion_metrics.json")

    if os.path.exists(regresion_metrics_file):
        with open(regresion_metrics_file, 'r') as f:
            regresion_metrics = json.load(f)

        consolidated_report["pipelines"]["regresion"] = {
            "status": "SUCCESS",
            "metrics": regresion_metrics,
            "model_path": "data/06_models/regresion_best_model.pkl"
        }

        print("\n[REGRESIÓN] Métricas cargadas exitosamente:")
        print(json.dumps(regresion_metrics, indent=2))
    else:
        print("\n[WARNING] No se encontraron métricas de regresión")
        consolidated_report["pipelines"]["regresion"] = {
            "status": "FAILED",
            "error": "Metrics file not found"
        }

    # ========================================================================
    # LEER MÉTRICAS DE CLASIFICACIÓN
    # ========================================================================

    classification_metrics_file = os.path.join(base_path, "classification_metrics.json")

    if os.path.exists(classification_metrics_file):
        with open(classification_metrics_file, 'r') as f:
            classification_metrics = json.load(f)

        consolidated_report["pipelines"]["clasificacion"] = {
            "status": "SUCCESS",
            "metrics": classification_metrics,
            "model_path": "data/06_models/classification_best_model.pkl"
        }

        print("\n[CLASIFICACIÓN] Métricas cargadas exitosamente:")
        print(json.dumps(classification_metrics, indent=2))
    else:
        print("\n[WARNING] No se encontraron métricas de clasificación")
        consolidated_report["pipelines"]["clasificacion"] = {
            "status": "FAILED",
            "error": "Metrics file not found"
        }

    # ========================================================================
    # GENERAR RESUMEN CONSOLIDADO
    # ========================================================================

    print("\n" + "=" * 70)
    print("RESUMEN CONSOLIDADO")
    print("=" * 70)

    # Guardar reporte consolidado
    os.makedirs(os.path.join(base_path, ".."), exist_ok=True)
    consolidated_file = "/opt/airflow/project/data/08_reporting/consolidated_report.json"
    os.makedirs(os.path.dirname(consolidated_file), exist_ok=True)

    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_report, indent=2, fp=f)

    print(f"\n[SUCCESS] Reporte consolidado guardado en: {consolidated_file}")

    # Generar resumen en consola
    print("\n" + "=" * 70)
    print("PIPELINES EJECUTADOS:")
    print("=" * 70)

    for pipeline_name, pipeline_data in consolidated_report["pipelines"].items():
        print(f"\n{pipeline_name.upper()}:")
        print(f"  Status: {pipeline_data['status']}")
        if pipeline_data['status'] == 'SUCCESS':
            print(f"  Modelo: {pipeline_data['model_path']}")

    print("\n" + "=" * 70)
    print("CONSOLIDACIÓN COMPLETADA")
    print("=" * 70)

    return consolidated_report

consolidar_resultados_task = PythonOperator(
    task_id='06_consolidar_resultados',
    python_callable=consolidar_resultados,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# DEFINIR DEPENDENCIAS DEL DAG
# ============================================================================
#
# Flujo de ejecución:
# 1. Verificar entorno (secuencial)
# 2. Preparación de datos en PARALELO (regresion_data || classification_data)
# 3. Entrenamiento de modelos en PARALELO (regresion_models || classification_models)
# 4. Consolidar resultados (secuencial)
#
# ============================================================================

# Fase 1: Verificación (secuencial)
verificar_entorno >> [ejecutar_regresion_data, ejecutar_classification_data]

# Fase 2: Data Preparation → Models Training (secuencial por pipeline)
ejecutar_regresion_data >> ejecutar_regresion_models
ejecutar_classification_data >> ejecutar_classification_models

# Fase 3: Consolidación (secuencial después de ambos modelos)
[ejecutar_regresion_models, ejecutar_classification_models] >> consolidar_resultados_task

# ============================================================================
# NOTAS DE CONFIGURACIÓN
# ============================================================================
#
# Para configurar alertas por email:
# 1. Configurar SMTP en airflow.cfg
# 2. Cambiar email_on_failure=True en default_args
# 3. Configurar email real en default_args['email']
#
# Para ejecución programada:
# - Cambiar schedule_interval a cron: '0 2 * * 0' (domingos a las 2 AM)
#
# Para integrar con DVC:
# - Agregar task de dvc pull antes de verificar_entorno
# - Agregar task de dvc push después de consolidar_resultados
#
# ============================================================================
