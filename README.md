# F1_ML

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Descripcion del Proyecto

Proyecto de analisis y machine learning de Formula 1 desarrollado con Kedro y PySpark.

**Fuente de datos:** [Formula 1 Analysis Dataset](https://www.kaggle.com/datasets/danish1212/formula1-analysis)

**Desarrolladores:**
- Braihan Gonzales
- Luis Salamanca

## Reglas y pautas

Para obtener el mejor rendimiento del proyecto:

* No remover lineas del archivo `.gitignore` proporcionado
* Asegurar que los resultados sean reproducibles siguiendo las [convenciones de ingenieria de datos](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* No hacer commit de datos al repositorio
* No hacer commit de credenciales o configuracion local al repositorio. Mantener todas las credenciales y configuracion local en `conf/local/`

## Como instalar dependencias

Declarar cualquier dependencia en `requirements.txt` para instalacion con `pip`.

Para instalarlas, ejecutar:

```
pip install -r requirements.txt
```

## Como ejecutar los pipelines de Kedro

Puedes ejecutar el proyecto Kedro con:

```
kedro run
```

Para ejecutar un pipeline especifico:

```
kedro run --pipeline <nombre_pipeline>
```

## Como visualizar el pipeline con Kedro Viz

Para visualizar los pipelines y el flujo de datos:

```
kedro viz
```

## Como probar el proyecto Kedro

Revisa los archivos `tests/test_run.py` y `tests/pipelines/data_science/test_pipeline.py` para instrucciones sobre como escribir pruebas. Ejecuta las pruebas de la siguiente manera:

```
pytest
```

Puedes configurar el umbral de cobertura en el archivo `pyproject.toml` del proyecto bajo la seccion `[tool.coverage.report]`.

## Dependencias del proyecto

Para ver y actualizar los requisitos de dependencias del proyecto usa `requirements.txt`. Instala los requisitos del proyecto con `pip install -r requirements.txt`.

[Mas informacion sobre dependencias del proyecto](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## Como trabajar con Kedro y notebooks

> Nota: Usar `kedro jupyter` o `kedro ipython` para ejecutar tu notebook proporciona estas variables en el ambito: `catalog`, `context`, `pipelines` y `session`.
>
> Jupyter, JupyterLab, e IPython ya estan incluidos en los requisitos del proyecto por defecto, asi que una vez que hayas ejecutado `pip install -r requirements.txt` no necesitaras pasos adicionales antes de usarlos.

### Jupyter
Para usar notebooks de Jupyter en tu proyecto Kedro, necesitas instalar Jupyter:

```
pip install jupyter
```

Despues de instalar Jupyter, puedes iniciar un servidor local de notebook:

```
kedro jupyter notebook
```

### JupyterLab
Para usar JupyterLab, necesitas instalarlo:

```
pip install jupyterlab
```

Tambien puedes iniciar JupyterLab:

```
kedro jupyter lab
```

### IPython
Y si quieres ejecutar una sesion de IPython:

```
kedro ipython
```

### Como ignorar las celdas de salida del notebook en `git`
Para eliminar automaticamente todo el contenido de las celdas de salida antes de hacer commit a `git`, puedes usar herramientas como [`nbstripout`](https://github.com/kynan/nbstripout). Por ejemplo, puedes agregar un hook en `.git/config` con `nbstripout --install`. Esto ejecutara `nbstripout` antes de que cualquier cosa se haga commit a `git`.

> *Nota:* Tus celdas de salida se mantendran localmente.

## Empaquetar tu proyecto Kedro

[Mas informacion sobre construccion de documentacion del proyecto y empaquetado de tu proyecto](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)


## Docker

Este proyecto incluye configuracion de Docker para facilitar el desarrollo y despliegue.

### Servicios disponibles

#### Jupyter Notebook
Para ejecutar Jupyter con Kedro:

```bash
docker compose up jupyter -d
```

Accede a Jupyter en: http://localhost:8888

#### Kedro Viz
Para visualizar los pipelines:

```bash
docker compose up kedro-viz -d
```

Accede a Kedro Viz en: http://localhost:4141

#### Ejecutar pipelines manualmente
Para acceder a la terminal y ejecutar pipelines:

```bash
docker compose run --rm --profile manual kedro-shell
```

Dentro del contenedor, ejecuta:
```bash
# Ejecutar pipeline especifico de regresion
kedro run --pipeline regresion_data
kedro run --pipeline regresion_models

# Ejecutar todos los pipelines
kedro run
```

### Ejecutar servicios Kedro
Para iniciar Jupyter y Kedro Viz simultaneamente:

```bash
docker compose up jupyter kedro-viz -d
```

## Apache Airflow - Orquestacion de Pipelines

Este proyecto integra Apache Airflow para orquestar los pipelines de Kedro de forma automatizada y escalable.

### Que es Airflow

Apache Airflow es una plataforma de orquestacion que permite:
- Programar ejecuciones automaticas de pipelines
- Monitorear el estado de las tareas en tiempo real
- Gestionar dependencias entre tareas
- Reintentar tareas fallidas automaticamente
- Visualizar el flujo de trabajo en una interfaz web

### DAG Implementado

El DAG `f1_ml_pipeline_granular` ejecuta cada nodo de Kedro como una tarea individual:

**Pipeline regresion_data (11 nodos):**
- Carga de datos, merge, filtrado, limpieza
- Split train/test, feature engineering
- Encoding y escalado

**Pipeline regresion_models (6 nodos):**
- Entrenamiento de 11 modelos base
- Seleccion y optimizacion de TOP 5
- Guardado del modelo final

**Total: 18 tareas secuenciales con reintentos automaticos**

### Inicio Rapido con Airflow

```bash
# 1. Inicializar Airflow (solo la primera vez)
docker compose up airflow-init

# 2. Iniciar TODOS los servicios (Kedro + Airflow)
docker compose up -d

# 3. Acceder a las interfaces web
# Airflow: http://localhost:8080 (usuario: airflow / password: airflow)
# Kedro Viz: http://localhost:4141
# Jupyter: http://localhost:8888

# 4. Detener todos los servicios
docker compose down
```

### Ejecutar el Pipeline desde Airflow

1. Accede a http://localhost:8080
2. Busca el DAG `f1_ml_pipeline_granular`
3. Activa el toggle para habilitarlo
4. Click en el boton "Play" para ejecutar

**Ventajas del DAG granular:**
- Visibilidad de cada nodo individual
- Reintentos automaticos por nodo (2 intentos, delay 3 min)
- Timeout de 30 minutos por nodo
- Logs especificos para debugging

O desde la linea de comandos:

```bash
docker compose run --rm --profile debug airflow-cli dags trigger f1_ml_pipeline_granular
```

### Documentacion Completa de Airflow

Para instrucciones detalladas, troubleshooting y mejores practicas, consulta:

- [docs/AIRFLOW_GUIA.md](docs/AIRFLOW_GUIA.md) - Guia completa de uso de Airflow

### Servicios de Airflow Incluidos

El stack completo de Airflow incluye:
- **PostgreSQL** - Base de datos de metadatos
- **Redis** - Message broker para Celery
- **Webserver** - Interfaz web en puerto 8080
- **Scheduler** - Programador de tareas
- **Worker** - Ejecutor de tareas con Celery
- **Triggerer** - Gestor de triggers asincronos

## Empaquetar tu proyecto Kedro

[Mas informacion sobre construccion de documentacion del proyecto y empaquetado de tu proyecto](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
