# F1_ML - Proyecto de Machine Learning para Formula 1

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Tabla de Contenidos

- [Descripcion del Proyecto](#descripcion-del-proyecto)
- [Entendimiento del Negocio](#entendimiento-del-negocio)
- [Metodologia](#metodologia)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Gestion de Versiones de Datos con DVC](#gestion-de-versiones-de-datos-con-dvc)
- [Instalacion y Configuracion](#instalacion-y-configuracion)
- [Formas de Ejecucion](#formas-de-ejecucion)
  - [Opcion 1: Ejecucion Local en Python](#opcion-1-ejecucion-local-en-python)
  - [Opcion 2: Ejecucion con Docker](#opcion-2-ejecucion-con-docker)
- [Visualizaciones y Herramientas](#visualizaciones-y-herramientas)
- [Apache Airflow - Automatizacion](#apache-airflow---automatizacion)
- [Desarrolladores](#desarrolladores)

---

## Descripcion del Proyecto

Proyecto de analisis y machine learning para Formula 1 desarrollado con Kedro y PySpark. El proyecto implementa multiples modelos de machine learning tanto de regresion como de clasificacion para predecir diferentes aspectos de las carreras de Formula 1.

**Fuente de datos:** [Formula 1 Analysis Dataset](https://www.kaggle.com/datasets/danish1212/formula1-analysis)

---

## Entendimiento del Negocio

Este proyecto aborda tres problemas principales de prediccion en el contexto de Formula 1:

### 1. Regresion: Prediccion de Posicion Final (IMPLEMENTADO)

**Objetivo:** Determinar la posicion final que obtendra un piloto en la carrera basandose en su posicion de salida en la parrilla y otras caracteristicas pre-carrera.

**Problema de negocio:** Antes de que comience una carrera, equipos, patrocinadores y casas de apuestas necesitan estimar cual sera la posicion final de cada piloto. Este modelo permite hacer predicciones informadas sobre el desempeno esperado.

**Variables utilizadas:**
- Posicion de salida en la parrilla
- Caracteristicas del circuito (latitud, longitud, altitud)
- Historial del piloto y constructor
- Estadisticas de la temporada actual
- Condiciones de clasificacion (Q1, Q2, Q3)

**Modelos entrenados:** GradientBoosting, Ridge, LightGBM, CatBoost, RandomForest, entre otros.

**Pipeline:** `regresion_data` + `regresion_models`

### 2. Clasificacion: Prediccion del Podio (IMPLEMENTADO)

**Objetivo:** Predecir cuales seran los tres pilotos que terminaran en el podio (posiciones 1, 2 y 3) en una carrera.

**Problema de negocio:** Identificar los potenciales ganadores del podio es crucial para estrategias de marketing, transmisiones deportivas, y apuestas deportivas. Este modelo clasifica si un piloto terminara en el podio o no.

**Variables utilizadas:**
- Posicion de salida (grid)
- Historial de podios del piloto y constructor
- Progreso de la temporada
- Caracteristicas del circuito
- Edad del piloto
- Tiempos de clasificacion

**Modelos entrenados:** LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting

**Tecnicas aplicadas:** SMOTE para balanceo de clases, dado que solo 3 de aproximadamente 20 pilotos llegan al podio.

**Pipeline:** `classification_data` + `classification_models`

### 3. Regresion: Prediccion de Tiempo por Vuelta (EN DESARROLLO)

**Objetivo:** Predecir el tiempo que tardara un piloto en completar cada vuelta durante la carrera.

**Problema de negocio:** Conocer los tiempos estimados por vuelta permite a los equipos optimizar estrategias de pit stops, gestion de neumaticos y estrategias de adelantamiento.

**Estado actual:** Se estan evaluando diferentes enfoques en los notebooks de exploracion:
- `notebooks/01-DataProcessingRegressor.ipynb` - Procesamiento de datos de tiempos por vuelta
- `notebooks/02-ModelsRegression.ipynb` - Evaluacion de modelos de regresion

**Nota:** Este modelo aun no esta implementado en los pipelines de Kedro. Se encuentra en fase de investigacion y desarrollo.

---

## Metodologia

El proyecto sigue una metodologia estructurada basada en las mejores practicas de Data Science e Ingenieria de Machine Learning:

### 1. CRISP-DM (Cross-Industry Standard Process for Data Mining)

- **Comprension del negocio:** Identificacion de los problemas de prediccion en Formula 1
- **Comprension de los datos:** Analisis exploratorio de datos historicos de F1 desde 1950 hasta 2024
- **Preparacion de datos:** Limpieza, transformacion y feature engineering especifico para cada problema
- **Modelado:** Entrenamiento de multiples algoritmos con optimizacion de hiperparametros
- **Evaluacion:** Metricas especificas (MAE, RMSE para regresion; F1-Score, Precision, Recall para clasificacion)
- **Despliegue:** Automatizacion con Apache Airflow y containerizacion con Docker

### 2. Ingenieria de Caracteristicas (Feature Engineering)

**Para Regresion:**
- Filtracion de datos de la era moderna (2010-2024)
- Eliminacion de variables con data leakage (positionText, points, time, etc.)
- Codificacion de variables categoricas (circuitName, driverRef, constructorRef)
- Escalado con StandardScaler y MinMaxScaler
- Split temporal train/test (80/20)

**Para Clasificacion:**
- Creacion de variable objetivo binaria (podio: 1-3 vs no podio: 4+)
- Features de agregacion temporal (podios ultimas 5 carreras)
- Tasa de podios por temporada para piloto y constructor
- Split temporal con fecha de corte (01-01-2023)
- Balanceo de clases con SMOTE
- Escalado diferenciado (MinMaxScaler para features de conteo, StandardScaler para coordenadas)

### 3. Validacion y Optimizacion

- **GridSearchCV:** Busqueda exhaustiva de hiperparametros optimos
- **Validacion cruzada:** 3-5 folds segun disponibilidad de memoria
- **Seleccion de modelos:** Entrenamiento de 11+ modelos base, seleccion de TOP 5, optimizacion final
- **Metricas de evaluacion:** MAE, RMSE, R2 (regresion); F1-Score, Precision, Recall, ROC-AUC (clasificacion)

---

## Tecnologias Utilizadas

### Framework de Proyecto
- **Kedro** - Framework de desarrollo de pipelines de datos reproducibles y mantenibles
- **Python 3.9+** - Lenguaje de programacion principal

### Procesamiento de Datos
- **PySpark** - Procesamiento distribuido de grandes volumenes de datos
- **Pandas** - Manipulacion y analisis de datos
- **NumPy** - Operaciones numericas

### Machine Learning
- **Scikit-learn** - Algoritmos de ML, preprocesamiento, metricas
- **XGBoost** - Gradient Boosting optimizado
- **LightGBM** - Gradient Boosting ligero y rapido
- **CatBoost** - Gradient Boosting con manejo nativo de categoricas
- **Imbalanced-learn** - Tecnicas de balanceo de clases (SMOTE)

### Visualizacion
- **Kedro-Viz** - Visualizacion interactiva de pipelines y linaje de datos
- **Matplotlib** - Graficos estaticos
- **Seaborn** - Graficos estadisticos

### Orquestacion y Automatizacion
- **Apache Airflow** - Orquestacion y programacion de pipelines
- **Docker & Docker Compose** - Containerizacion y despliegue

### Gestion de Datos
- **DVC (Data Version Control)** - Control de versiones de datos y modelos

### Notebooks
- **Jupyter Lab** - Entorno de desarrollo interactivo
- **IPython** - Shell interactiva de Python

---

## Gestion de Versiones de Datos con DVC

Este proyecto utiliza **DVC (Data Version Control)** para gestionar el versionado de datos y modelos de machine learning.

### Que es DVC

DVC es un sistema de control de versiones para datos y modelos de ML que funciona sobre Git. Permite:
- Rastrear cambios en archivos grandes (datasets, modelos)
- Compartir datos entre el equipo sin subirlos a Git
- Reproducir experimentos y resultados
- Gestionar multiples versiones de datasets y modelos

### Como Funciona en Este Proyecto

Los archivos de datos se encuentran en las carpetas:
- `data/01_raw/` - Datos originales sin procesar
- `data/02_intermediate/` - Datos intermedios en proceso
- `data/03_primary/` - Datos limpios listos para modelado
- `data/04_feature/` - Datos con features engineerizadas
- `data/05_model_input/` - Datos de entrada para modelos
- `data/06_models/` - Modelos entrenados guardados
- `data/07_model_output/` - Predicciones y resultados
- `data/08_reporting/` - Reportes y metricas

### Archivos .dvc

DVC crea archivos `.dvc` que actuan como punteros a los datos reales:
- Los archivos `.dvc` se versionan en Git (son pequenos, solo metadatos)
- Los datos reales se almacenan en un storage remoto (configurado en `.dvc/config`)
- Al hacer `git clone`, solo se descargan los `.dvc`, no los datos completos
- Para descargar los datos reales: `dvc pull`
- Para subir nuevas versiones: `dvc add <archivo>` + `git add <archivo>.dvc` + `dvc push`

### Pipeline de DVC (dvc.yaml)

El proyecto utiliza `dvc.yaml` para definir un pipeline reproducible de ML que incluye:

**Stages del Pipeline:**
1. **regresion_data_preparation**: Prepara datos para regresión (11 nodos de Kedro)
2. **regresion_models_training**: Entrena y optimiza modelos de regresión (6 nodos)
3. **classification_data_preparation**: Prepara datos para clasificación (13 nodos)
4. **classification_models_training**: Entrena y optimiza modelos de clasificación (7 nodos)

**Ventajas del pipeline DVC:**
- Reproducibilidad automática de experimentos
- Tracking de dependencias entre stages
- Cacheo inteligente (solo re-ejecuta stages con cambios)
- Versionado automático de datos, features y modelos
- Métricas y plots versionados con Git

### Comandos DVC Básicos

```bash
# Descargar datos trackeados por DVC
dvc pull

# Agregar nuevos datos al tracking
dvc add data/01_raw/nuevo_dataset.csv
git add data/01_raw/nuevo_dataset.csv.dvc .gitignore
git commit -m "Agregar nuevo dataset"

# Subir datos al storage remoto
dvc push

# Ver el estado de archivos DVC
dvc status

# Reproducir pipeline completo (ejecuta todos los stages)
dvc repro

# Reproducir solo un stage específico
dvc repro regresion_models_training

# Ver el DAG del pipeline
dvc dag

# Ver métricas de experimentos
dvc metrics show
dvc metrics diff  # Comparar con versión anterior

# Ver plots de experimentos
dvc plots show
```

### Scripts de Automatización DVC

El proyecto incluye scripts bash para facilitar operaciones comunes de DVC:

#### dvc_push.sh - Subir datos y modelos al remote

```bash
# Push completo de todos los artefactos
./scripts/dvc_push.sh

# Push solo datos
./scripts/dvc_push.sh --data

# Push solo modelos
./scripts/dvc_push.sh --models

# Push específico
./scripts/dvc_push.sh data/06_models.dvc
```

**Características:**
- Validación de entorno y remote configurado
- Confirmación interactiva antes de push
- Auto-commit de archivos .dvc y dvc.lock en Git
- Logging detallado de operaciones
- Manejo de errores robusto

#### dvc_pull.sh - Descargar datos y modelos desde remote

```bash
# Pull completo de todos los artefactos
./scripts/dvc_pull.sh

# Pull solo datos raw
./scripts/dvc_pull.sh --data-raw

# Pull solo features procesadas
./scripts/dvc_pull.sh --features

# Pull solo modelos
./scripts/dvc_pull.sh --models

# Pull con verificación de checksums
./scripts/dvc_pull.sh --all --verify
```

**Características:**
- Validación de integridad de datos descargados
- Contador de archivos por categoría
- Verificación opcional de checksums MD5
- Sugerencias de próximos pasos después del pull

### Configuracion del Storage Remoto

#### Opción 1: Local (Por defecto - Desarrollo)

El proyecto viene configurado con storage local en `dvcstore/`:

```bash
# Ya está configurado, no requiere acción adicional
dvc push  # Guarda en ./dvcstore/
dvc pull  # Descarga de ./dvcstore/
```

#### Opción 2: Amazon S3 (Producción recomendada)

**Configuración:**

1. Crear bucket en AWS S3:
```bash
aws s3 mb s3://f1-ml-dvc-storage --region us-east-1
```

2. Descomentar y configurar en `.dvc/config`:
```ini
['remote "s3remote"']
    url = s3://f1-ml-dvc-storage/dvc-storage
    region = us-east-1
```

3. Cambiar remote por defecto:
```bash
dvc remote default s3remote
```

4. Configurar credenciales (opción 1 - AWS CLI):
```bash
aws configure
# Ingresa AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY
```

**O usar variables de entorno:**
```bash
export AWS_ACCESS_KEY_ID="tu-access-key"
export AWS_SECRET_ACCESS_KEY="tu-secret-key"
```

#### Opción 3: Google Cloud Storage (GCS)

**Configuración:**

1. Crear bucket en GCS:
```bash
gsutil mb -p tu-proyecto gs://f1-ml-dvc-storage
```

2. Descomentar y configurar en `.dvc/config`:
```ini
['remote "gcsremote"']
    url = gs://f1-ml-dvc-storage/dvc-storage
```

3. Cambiar remote por defecto:
```bash
dvc remote default gcsremote
```

4. Autenticar con gcloud:
```bash
gcloud auth application-default login
```

**O usar service account:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Opción 4: Azure Blob Storage

**Configuración:**

1. Crear container en Azure:
```bash
az storage container create --name f1-ml-dvc --account-name tuaccount
```

2. Descomentar y configurar en `.dvc/config`:
```ini
['remote "azureremote"']
    url = azure://f1-ml-dvc/dvc-storage
```

3. Cambiar remote por defecto:
```bash
dvc remote default azureremote
```

4. Configurar connection string:
```bash
dvc remote modify azureremote --local connection_string "tu-connection-string"
```

### Workflow Completo con DVC

**Caso de uso: Experimentar con nuevas features**

```bash
# 1. Pull de datos actuales
dvc pull

# 2. Modificar código (ej: agregar nueva feature en regresion_data/nodes.py)
vim src/f1_ml/pipelines/regresion_data/nodes.py

# 3. Modificar parámetros si es necesario
vim conf/base/parameters.yml

# 4. Reproducir pipeline (DVC detecta cambios y re-ejecuta stages afectados)
dvc repro

# 5. Revisar métricas y comparar con versión anterior
dvc metrics show
dvc metrics diff

# 6. Si los resultados son buenos, commitear cambios
git add dvc.lock src/ conf/
git commit -m "feat: Agregar nueva feature que mejora R2 en 0.05"

# 7. Push de datos y modelos al remote
dvc push

# 8. Push de código al repositorio
git push
```

### Integración DVC + Airflow

El DAG unificado de Airflow (`f1_ml_unified_pipeline.py`) puede integrarse con DVC:

**Agregar tasks de DVC al DAG:**

```python
# Task de dvc pull antes de ejecutar pipelines
dvc_pull_task = BashOperator(
    task_id='00_dvc_pull',
    bash_command='dvc pull',
    dag=dag,
)

# Task de dvc push después de consolidar resultados
dvc_push_task = BashOperator(
    task_id='07_dvc_push',
    bash_command='dvc push',
    dag=dag,
)

# Flujo: dvc_pull → pipelines → consolidar → dvc_push
dvc_pull_task >> verificar_entorno >> ... >> consolidar_resultados >> dvc_push_task
```

### Reproducibilidad Completa

Con DVC, Git y Docker, el proyecto es 100% reproducible:

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/f1-ml.git
cd f1-ml

# 2. Configurar remote de DVC (si usas S3/GCS/Azure)
vim .dvc/config  # Descomentar remote apropiado

# 3. Pull de datos y modelos
dvc pull

# 4. Levantar entorno con Docker
docker-compose up -d

# 5. Ejecutar pipeline de Kedro (opcional, ya tienes los modelos)
kedro run

# 6. O reproducir desde cero con DVC
dvc repro
```

### Troubleshooting DVC

**Problema: `dvc push` falla con error de autenticación**

Solución AWS S3:
```bash
aws configure list  # Verificar configuración
aws s3 ls s3://tu-bucket  # Probar acceso
dvc remote modify s3remote --local access_key_id TU_KEY
dvc remote modify s3remote --local secret_access_key TU_SECRET
```

**Problema: `dvc pull` descarga archivos corruptos**

Solución:
```bash
dvc status  # Ver qué archivos están desactualizados
dvc fetch --all  # Descargar todo sin modificar working directory
dvc checkout  # Actualizar working directory con archivos del cache
```

**Problema: Archivos grandes causan OOM al hacer `dvc add`**

Solución:
```bash
# Dividir archivo grande en chunks o usar streaming
split -b 100M archivo_grande.csv archivo_chunk_
dvc add archivo_chunk_*
```

**Problema: `.dvc/cache` ocupa mucho espacio en disco**

Solución:
```bash
# Limpiar cache de versiones antiguas no referenciadas
dvc gc --workspace  # Mantener solo versiones actuales
dvc gc --cloud      # Limpiar también en remote
```

### Mejores Prácticas DVC

1. **Commitear siempre dvc.lock**: Este archivo garantiza reproducibilidad exacta
2. **No editar manualmente archivos .dvc**: Usar comandos `dvc add`, `dvc run`, etc.
3. **Usar dvc.yaml para pipelines ML**: Mejor que scripts bash ad-hoc
4. **Versionar parámetros en params.yaml**: DVC trackea cambios automáticamente
5. **Configurar remote en producción**: No usar storage local para proyectos serios
6. **Usar `.dvcignore` para excluir archivos**: Similar a `.gitignore` para DVC
7. **Hacer `dvc push` regularmente**: No perder trabajo por fallos de disco local

---

## Instalacion y Configuracion

### Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- Docker y Docker Compose (opcional, para ejecucion con contenedores)
- Git
- DVC (opcional, para gestion de datos versionados)

### Instalacion de Dependencias

1. Clonar el repositorio:
```bash
git clone https://github.com/Luissalamanca23/F1_ML.git
cd F1_ML
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instalar dependencias del proyecto:
```bash
pip install -r requirements.txt
```

4. (Opcional) Si deseas usar Airflow localmente sin Docker:
```bash
pip install -r requirements-airflow.txt
```

5. (Opcional) Descargar datos versionados con DVC:
```bash
dvc pull
```

---

## Formas de Ejecucion

Existen dos formas principales de ejecutar los pipelines del proyecto:

---

## Opcion 1: Ejecucion Local en Python

Esta es la forma recomendada para computadoras con recursos limitados o para desarrollo iterativo.

### A. Instalacion de Dependencias

Como se menciono anteriormente, primero instala las dependencias:

```bash
pip install -r requirements.txt
```

### B. Ejecucion Completa de Todos los Pipelines

Para ejecutar el pipeline completo (regresion + clasificacion) de principio a fin:

```bash
kedro run
```

Este comando ejecutara automaticamente todos los pipelines en el orden correcto:
1. `regresion_data` - Limpieza y preparacion de datos para regresion
2. `regresion_models` - Entrenamiento y evaluacion de modelos de regresion
3. `classification_data` - Limpieza y preparacion de datos para clasificacion
4. `classification_models` - Entrenamiento y evaluacion de modelos de clasificacion

**Nota:** La ejecucion completa puede tomar entre 30 minutos y 2 horas dependiendo de tu hardware.

### C. Ejecucion Individual de Pipelines

Para tener mayor control y ejecutar pipelines especificos, utiliza los siguientes comandos:

#### 1. Pipeline de Limpieza de Datos de Regresion
```bash
kedro run --pipeline=regresion_data
```

**Que hace:**
- Carga datos raw de Formula 1
- Filtra por era moderna (2010-2024)
- Limpia valores nulos y duplicados
- Realiza merge de multiples tablas (results, races, drivers, constructors, etc.)
- Elimina variables con data leakage
- Aplica feature engineering
- Codifica variables categoricas
- Escala features numericas
- Divide en train/test (80/20)

**Salida:** Datasets listos para modelado en `data/05_model_input/`

#### 2. Pipeline de Modelos de Regresion
```bash
kedro run --pipeline=regresion_models
```

**Que hace:**
- Entrena 11 modelos base (LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, SVR)
- Evalua con metricas MAE, RMSE, R2
- Selecciona los TOP 5 mejores modelos
- Optimiza hiperparametros con GridSearchCV
- Guarda el mejor modelo final

**Salida:**
- Modelos entrenados en `data/06_models/`
- Metricas y reportes en `data/08_reporting/`

**Requisitos previos:** Debe haberse ejecutado `regresion_data` primero.

#### 3. Pipeline de Limpieza de Datos de Clasificacion
```bash
kedro run --pipeline=classification_data
```

**Que hace:**
- Carga datos raw de Formula 1
- Crea variable objetivo binaria (podio: 1 si posicion <= 3, 0 si no)
- Calcula features de agregacion temporal (podios ultimas 5 carreras)
- Calcula tasas de podio por temporada
- Elimina variables con data leakage
- Codifica variables categoricas
- Aplica escalado diferenciado (MinMaxScaler y StandardScaler)
- Split temporal con fecha de corte (01-01-2023)

**Salida:** Datasets listos para modelado en `data/05_model_input/`

#### 4. Pipeline de Modelos de Clasificacion
```bash
kedro run --pipeline=classification_models
```

**Que hace:**
- Aplica SMOTE para balanceo de clases (sampling_strategy=0.5)
- Entrena 11 modelos base (LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting, etc.)
- Evalua con metricas F1-Score, Precision, Recall, ROC-AUC
- Selecciona los TOP 5 mejores modelos
- Optimiza hiperparametros con GridSearchCV
- Guarda el mejor modelo final

**Salida:**
- Modelos entrenados en `data/06_models/`
- Metricas y reportes en `data/08_reporting/`

**Requisitos previos:** Debe haberse ejecutado `classification_data` primero.

### D. Orden Recomendado de Ejecucion Manual

Si deseas ejecutar los pipelines uno por uno, sigue este orden:

```bash
# 1. Preparar datos de regresion
kedro run --pipeline=regresion_data

# 2. Entrenar modelos de regresion
kedro run --pipeline=regresion_models

# 3. Preparar datos de clasificacion
kedro run --pipeline=classification_data

# 4. Entrenar modelos de clasificacion
kedro run --pipeline=classification_models
```

### E. Visualizacion de Pipelines con Kedro Viz

Para visualizar la estructura de los pipelines, sus nodos y el linaje de datos:

```bash
kedro viz
```

Esto abrira una interfaz web interactiva en `http://localhost:4141` donde podras:
- Ver graficamente todos los pipelines
- Explorar las dependencias entre nodos
- Visualizar el flujo de datos
- Inspeccionar parametros y datasets

---

## Opcion 2: Ejecucion con Docker

Docker permite ejecutar todo el proyecto en contenedores aislados, garantizando reproducibilidad y evitando conflictos de dependencias.

### Servicios Disponibles

El proyecto incluye los siguientes servicios Docker:

1. **Jupyter Notebook** - Para desarrollo y exploracion interactiva
2. **Kedro Viz** - Para visualizacion de pipelines
3. **Apache Airflow** - Para orquestacion automatizada de pipelines

### A. Inicializacion Inicial de Docker

Solo es necesario la primera vez:

```bash
docker compose up airflow-init
```

Este comando inicializa la base de datos de Airflow y crea el usuario administrador.

### B. Iniciar Todos los Servicios

Para levantar todos los servicios (Kedro + Airflow + Jupyter + Kedro Viz):

```bash
docker compose up -d
```

### C. Acceder a las Interfaces Web

Una vez que los servicios esten corriendo, puedes acceder a:

- **Airflow Web UI:** http://localhost:8080
  - Usuario: `airflow`
  - Password: `airflow`

- **Kedro Viz:** http://localhost:4141
  - Visualizacion interactiva de pipelines

- **Jupyter Lab:** http://localhost:8888
  - Notebooks y desarrollo interactivo

### D. Ejecutar Pipelines Manualmente en Docker

Para acceder a una terminal dentro del contenedor y ejecutar comandos Kedro:

```bash
docker compose run --rm --profile manual kedro-shell
```

Dentro del contenedor, ejecuta cualquiera de los comandos de Kedro:

```bash
# Pipeline completo
kedro run

# Pipelines individuales
kedro run --pipeline=regresion_data
kedro run --pipeline=regresion_models
kedro run --pipeline=classification_data
kedro run --pipeline=classification_models

# Visualizar pipelines
kedro viz
```

Para salir del contenedor: `exit`

### E. Iniciar Solo Jupyter y Kedro Viz

Si solo necesitas estos servicios sin Airflow:

```bash
docker compose up jupyter kedro-viz -d
```

### F. Detener Todos los Servicios

```bash
docker compose down
```

Para detener y eliminar volumenes (CUIDADO: borrara datos de Airflow):

```bash
docker compose down -v
```

---

## Apache Airflow - Automatizacion

Apache Airflow permite orquestar y automatizar la ejecucion de los pipelines de Kedro.

### Que es Apache Airflow

Apache Airflow es una plataforma open-source para:
- Programar ejecuciones automaticas de pipelines (diarias, semanales, etc.)
- Monitorear el estado de las tareas en tiempo real
- Gestionar dependencias entre tareas
- Reintentar tareas fallidas automaticamente
- Visualizar el flujo de trabajo en una interfaz web intuitiva

### DAG Implementado: f1_ml_pipeline_granular

El proyecto incluye un DAG (Directed Acyclic Graph) que ejecuta cada nodo de Kedro como una tarea individual.

**Estructura del DAG:**

1. **Pipeline regresion_data (11 nodos):**
   - Carga de datos
   - Merge de tablas
   - Filtrado temporal
   - Limpieza de nulos
   - Eliminacion de data leakage
   - Split train/test
   - Feature engineering
   - Encoding de categoricas
   - Escalado de features

2. **Pipeline regresion_models (6 nodos):**
   - Entrenamiento de 11 modelos base
   - Evaluacion de modelos
   - Seleccion de TOP 5
   - Optimizacion con GridSearchCV
   - Seleccion del mejor modelo
   - Guardado del modelo final

3. **Pipeline classification_data (nodos):**
   - Preparacion de datos de clasificacion
   - Creacion de variable objetivo
   - Feature engineering temporal
   - Encoding y escalado

4. **Pipeline classification_models (nodos):**
   - Balanceo con SMOTE
   - Entrenamiento de modelos
   - Optimizacion de hiperparametros
   - Guardado del mejor modelo

**Total:** Tareas secuenciales con reintentos automaticos

**Configuracion de tareas:**
- Reintentos: 2 intentos por tarea
- Delay entre reintentos: 3 minutos
- Timeout por tarea: 30 minutos
- Logs especificos para debugging

### Como Usar Airflow

#### 1. Iniciar Airflow

```bash
# Inicializar (solo primera vez)
docker compose up airflow-init

# Levantar servicios
docker compose up -d
```

#### 2. Acceder a la Interfaz Web

Abre tu navegador en: http://localhost:8080

- Usuario: `airflow`
- Password: `airflow`

#### 3. Ejecutar el DAG desde la Interfaz

1. Busca el DAG `f1_ml_pipeline_granular` en la lista
2. Activa el toggle a la izquierda para habilitarlo
3. Click en el boton "Play" (triangulo) a la derecha para ejecutar manualmente
4. Monitorea el progreso en tiempo real desde la vista de Grid o Graph

#### 4. Ejecutar el DAG desde CLI

```bash
docker compose run --rm --profile debug airflow-cli dags trigger f1_ml_pipeline_granular
```

### ADVERTENCIA IMPORTANTE: Limitaciones de RAM en Docker

**PROBLEMA:** Los pipelines de machine learning requieren cantidades significativas de memoria RAM, especialmente durante el entrenamiento de multiples modelos con GridSearchCV. Al ejecutar los DAGs en Apache Airflow dentro de Docker, el sistema puede experimentar:

- Saturacion de RAM
- Lentitud extrema en la ejecucion
- Tareas que no terminan correctamente
- Crashes de contenedores
- Errores de "OOM Killer" (Out Of Memory)

**RECOMENDACIONES:**

1. **Si tu computadora tiene recursos limitados (< 16 GB RAM):**
   - NO ejecutes los DAGs de Airflow
   - Ejecuta los pipelines manualmente de forma local con `kedro run --pipeline <nombre>`
   - Ejecuta los pipelines uno por uno para distribuir la carga de memoria

2. **Si tu computadora es potente (>= 16 GB RAM, CPU multi-core):**
   - Puedes intentar ejecutar los DAGs en Airflow
   - Monitorea el uso de recursos con `docker stats`
   - Si ves problemas de memoria, detén la ejecucion y ejecuta manualmente

3. **Configuracion de recursos de Docker:**
   - Asegurate de asignar suficiente RAM a Docker Desktop (minimo 8 GB, recomendado 12+ GB)
   - En Docker Desktop: Settings > Resources > Memory

**Alternativa mas segura:**

```bash
# En lugar de usar Airflow, ejecuta manualmente:
kedro run --pipeline=regresion_data
kedro run --pipeline=regresion_models
kedro run --pipeline=classification_data
kedro run --pipeline=classification_models
```


---

## Visualizaciones y Herramientas

### Kedro Viz

Kedro Viz proporciona una visualizacion interactiva de:
- Arquitectura completa de pipelines
- Dependencias entre nodos
- Linaje de datos (data lineage)
- Parametros de configuracion
- Metricas de ejecucion

**Ejecucion local:**
```bash
kedro viz
```

**Ejecucion en Docker:**
```bash
docker compose up kedro-viz -d
```

Accede en: http://localhost:4141

### Jupyter Lab

Para exploracion de datos y desarrollo de notebooks:

**Ejecucion local:**
```bash
kedro jupyter lab
```

**Ejecucion en Docker:**
```bash
docker compose up jupyter -d
```

Accede en: http://localhost:8888

**Variables disponibles en notebooks:**
- `catalog` - Acceso al catalogo de datos
- `context` - Contexto del proyecto Kedro
- `pipelines` - Diccionario de pipelines disponibles
- `session` - Sesion activa de Kedro

### Reportes de Modelos

Los reportes de metricas y evaluacion se generan automaticamente en:
- `data/08_reporting/regresion/metrics_*.csv` - Metricas de modelos de regresion
- `data/08_reporting/classification/metrics_*.csv` - Metricas de modelos de clasificacion

---

## Reglas y Pautas de Desarrollo

Para mantener el proyecto reproducible y mantenible:

- No remover lineas del archivo `.gitignore` proporcionado
- Seguir las [convenciones de ingenieria de datos de Kedro](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
- NO hacer commit de datos al repositorio (usar DVC en su lugar)
- NO hacer commit de credenciales o configuracion local al repositorio
- Mantener todas las credenciales y configuracion local en `conf/local/`
- Documentar nuevos pipelines y nodos en el codigo
- Escribir tests para funciones criticas

---

## Testing

### Ejecutar Tests

```bash
pytest
```

### Configurar Cobertura

El umbral de cobertura se configura en `pyproject.toml` bajo la seccion `[tool.coverage.report]`.

### Escribir Tests

Los tests se encuentran en:
- `tests/test_run.py` - Tests generales
- `tests/pipelines/` - Tests de pipelines especificos

---

## Roadmap Futuro

### Modelo de Regresion: Tiempo por Vuelta (En Desarrollo)

**Objetivo:** Predecir el tiempo que tardara un piloto en completar cada vuelta durante la carrera.

**Estado actual:**
- Notebooks de exploracion en `notebooks/01-DataProcessingRegressor.ipynb` y `notebooks/02-ModelsRegression.ipynb`
- Analisis de viabilidad en progreso
- Aun no implementado en pipelines de Kedro

**Proximos pasos:**
1. Finalizar exploracion de features relevantes
2. Definir estrategia de modelado (serie temporal vs regresion tradicional)
3. Implementar pipelines `laptime_data` y `laptime_models`
4. Integrar en el DAG de Airflow
5. Evaluar metricas de rendimiento

**Contribuciones:** Se aceptan sugerencias y contribuciones para este modelo.

---

## Desarrolladores

- **Braihan Gonzales**
- **Luis Salamanca**

---

## Recursos Adicionales

- [Documentacion de Kedro](https://docs.kedro.org/)
- [Documentacion de Apache Airflow](https://airflow.apache.org/docs/)
- [Documentacion de DVC](https://dvc.org/doc)
- [Dataset de Formula 1 en Kaggle](https://www.kaggle.com/datasets/danish1212/formula1-analysis)


---

## Contacto

Para preguntas, sugerencias o reporte de bugs, contacta a los desarrolladores o abre un issue en el repositorio.
