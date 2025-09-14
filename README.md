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
