"""
Nodes del Pipeline de Entrenamiento de Modelos para Regresión F1
=================================================================

Este módulo contiene las 6 funciones (nodes) que entrenan, evalúan y optimizan
modelos de regresión para predecir la posición final en F1.

Flujo:
1. train_all_models - Entrenar 11 modelos base
2. evaluate_models - Evaluar todos los modelos
3. select_top5_models - Seleccionar TOP 5 por R2
4. optimize_with_gridsearch - GridSearch en TOP 5
5. evaluate_optimized_models - Evaluar modelos optimizados
6. save_final_model - Guardar mejor modelo y métricas

Autor: Pipeline F1 ML
Fecha: 2025-10-25
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import time

# Sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Modelos avanzados
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# NODE 1: ENTRENAR TODOS LOS MODELOS BASE (11 MODELOS)
# =============================================================================

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Entrena 11 modelos base sin optimización de hiperparámetros.

    Modelos:
    1. RandomForest
    2. GradientBoosting
    3. ExtraTrees
    4. Ridge
    5. Lasso
    6. ElasticNet
    7. DecisionTree
    8. KNN
    9. XGBoost (si está instalado)
    10. LightGBM (si está instalado)
    11. CatBoost (si está instalado)

    Args:
        X_train: Features de entrenamiento escaladas
        y_train: Target de entrenamiento
        X_test: Features de test escaladas
        y_test: Target de test
        params: Diccionario con random_state

    Returns:
        Dict con resultados de cada modelo
    """
    logger.info("[TRAIN] Entrenando 11 modelos base...")

    random_state = params.get("random_state", 42)
    resultados = {}

    # Definir modelos
    modelos = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=1.0, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=1.0, random_state=random_state),
        "DecisionTree": DecisionTreeRegressor(
            random_state=random_state, max_depth=10
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }

    # Agregar modelos avanzados si están disponibles
    if XGBOOST_AVAILABLE:
        modelos["XGBoost"] = XGBRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0
        )

    if LIGHTGBM_AVAILABLE:
        modelos["LightGBM"] = LGBMRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1
        )

    if CATBOOST_AVAILABLE:
        modelos["CatBoost"] = CatBoostRegressor(
            iterations=100, random_state=random_state, verbose=0
        )

    logger.info(f"  Total de modelos a entrenar: {len(modelos)}")

    # Entrenar cada modelo
    for nombre, modelo in modelos.items():
        logger.info(f"  Entrenando {nombre}...")
        inicio = time.time()

        try:
            # Entrenar
            modelo.fit(X_train, y_train)

            # Predicciones
            y_train_pred = modelo.predict(X_train)
            y_test_pred = modelo.predict(X_test)

            # Métricas
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            overfitting = r2_train - r2_test

            tiempo = time.time() - inicio

            resultados[nombre] = {
                "modelo": modelo,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "mae_test": mae_test,
                "rmse_test": rmse_test,
                "overfitting": overfitting,
                "y_test_pred": y_test_pred,
                "tiempo_entrenamiento": tiempo,
            }

            logger.info(
                f"    [OK] {nombre} - R2 test: {r2_test:.4f} (tiempo: {tiempo:.1f}s)"
            )

        except Exception as e:
            logger.error(f"    [ERROR] {nombre} falló: {str(e)}")
            continue

    logger.info(f"[OK] Modelos entrenados exitosamente: {len(resultados)}")

    return resultados


# =============================================================================
# NODE 2: EVALUAR MODELOS
# =============================================================================

def evaluate_models(
    resultados_modelos: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Genera tabla comparativa con métricas de todos los modelos.

    Args:
        resultados_modelos: Dict con resultados de train_all_models

    Returns:
        DataFrame con comparación de modelos ordenado por R2 test
    """
    logger.info("[EVAL] Evaluando y comparando modelos...")

    # Crear DataFrame
    datos_comparacion = []
    for nombre, resultado in resultados_modelos.items():
        datos_comparacion.append(
            {
                "Modelo": nombre,
                "R2_Train": resultado["r2_train"],
                "R2_Test": resultado["r2_test"],
                "MAE_Test": resultado["mae_test"],
                "RMSE_Test": resultado["rmse_test"],
                "Overfitting": resultado["overfitting"],
                "Tiempo_seg": resultado["tiempo_entrenamiento"],
            }
        )

    df_comparacion = pd.DataFrame(datos_comparacion)

    # Ordenar por R2 test descendente
    df_comparacion = df_comparacion.sort_values(
        "R2_Test", ascending=False
    ).reset_index(drop=True)

    logger.info(f"[OK] Comparación generada: {len(df_comparacion)} modelos")
    logger.info("\nRanking de modelos (TOP 5):")
    for idx, row in df_comparacion.head(5).iterrows():
        logger.info(
            f"  {idx+1}. {row['Modelo']}: R2={row['R2_Test']:.4f}, MAE={row['MAE_Test']:.2f}"
        )

    return df_comparacion


# =============================================================================
# NODE 3: SELECCIONAR TOP 5 MODELOS
# =============================================================================

def select_top5_models(
    resultados_modelos: Dict[str, Dict[str, Any]],
    df_comparacion: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Selecciona los TOP 5 modelos por R2 test para optimización.

    Args:
        resultados_modelos: Dict con todos los modelos entrenados
        df_comparacion: DataFrame con comparación de modelos

    Returns:
        Dict con solo los TOP 5 modelos
    """
    logger.info("[SELECT] Seleccionando TOP 5 modelos...")

    # Obtener nombres de TOP 5
    top5_nombres = df_comparacion.head(5)["Modelo"].tolist()

    # Filtrar solo TOP 5 del dict original
    top5_modelos = {
        nombre: resultados_modelos[nombre] for nombre in top5_nombres
    }

    logger.info(f"[OK] TOP 5 seleccionados: {', '.join(top5_nombres)}")

    return top5_modelos


# =============================================================================
# NODE 4: OPTIMIZAR CON GRIDSEARCH (TOP 5)
# =============================================================================

def optimize_with_gridsearch(
    top5_modelos: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Aplica GridSearchCV a los TOP 5 modelos con validación cruzada.

    Args:
        top5_modelos: Dict con TOP 5 modelos
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de test
        y_test: Target de test
        params: Dict con 'random_state', 'gridsearch_grids' y 'cv_folds'

    Returns:
        Dict con modelos optimizados
    """
    random_state = params.get("random_state", 42)
    grids = params.get("gridsearch_grids", {})
    cv_folds = params.get("cv_folds", 5)

    logger.info("[GRIDSEARCH] Iniciando optimización de TOP 5 modelos...")
    logger.info(f"  Método: GridSearchCV con {cv_folds}-fold CV")

    resultados_optimizados = {}

    for nombre in top5_modelos.keys():
        logger.info(f"\n  [{nombre}] Iniciando GridSearch...")

        # Obtener grid de parámetros
        param_grid = grids.get(nombre, {})

        if not param_grid:
            logger.warning(f"    [SKIP] No hay grid para {nombre}, se omite")
            continue

        # Calcular combinaciones
        from itertools import product
        n_combinaciones = len(list(product(*param_grid.values())))
        logger.info(f"    Combinaciones a probar: {n_combinaciones}")

        # Crear modelo base
        modelo_base = _crear_modelo_base(nombre, random_state)

        if modelo_base is None:
            logger.warning(f"    [SKIP] No se pudo crear modelo base para {nombre}")
            continue

        # GridSearch con validación cruzada
        inicio = time.time()
        grid_search = GridSearchCV(
            estimator=modelo_base,
            param_grid=param_grid,
            scoring="r2",
            cv=cv_folds,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)
        tiempo = time.time() - inicio

        logger.info(f"    [OK] Completado en {tiempo:.1f}s")
        logger.info(f"    Mejor R2 (CV): {grid_search.best_score_:.4f}")
        logger.info(f"    Mejores params: {grid_search.best_params_}")

        # Evaluar mejor modelo
        mejor_modelo = grid_search.best_estimator_
        y_train_pred = mejor_modelo.predict(X_train)
        y_test_pred = mejor_modelo.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        resultados_optimizados[nombre] = {
            "modelo": mejor_modelo,
            "r2_cv": grid_search.best_score_,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_test": mae_test,
            "rmse_test": rmse_test,
            "overfitting": r2_train - r2_test,
            "best_params": grid_search.best_params_,
            "y_test_pred": y_test_pred,
            "tiempo_gridsearch": tiempo,
        }

    logger.info(f"\n[OK] GridSearch completado: {len(resultados_optimizados)} modelos optimizados")

    return resultados_optimizados


def _crear_modelo_base(nombre: str, random_state: int):
    """Crea instancia de modelo base para GridSearch."""
    if nombre == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif nombre == "GradientBoosting":
        return GradientBoostingRegressor(random_state=random_state)
    elif nombre == "ExtraTrees":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    elif nombre == "Ridge":
        return Ridge(random_state=random_state)
    elif nombre == "Lasso":
        return Lasso(random_state=random_state)
    elif nombre == "ElasticNet":
        return ElasticNet(random_state=random_state)
    elif nombre == "XGBoost" and XGBOOST_AVAILABLE:
        return XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0)
    elif nombre == "LightGBM" and LIGHTGBM_AVAILABLE:
        return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
    elif nombre == "CatBoost" and CATBOOST_AVAILABLE:
        return CatBoostRegressor(random_state=random_state, verbose=0)
    else:
        return None


# =============================================================================
# NODE 5: EVALUAR MODELOS OPTIMIZADOS
# =============================================================================

def evaluate_optimized_models(
    resultados_optimizados: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Genera tabla comparativa de modelos después de GridSearch.

    Args:
        resultados_optimizados: Dict con modelos optimizados

    Returns:
        DataFrame con comparación ordenado por R2 test
    """
    logger.info("[EVAL] Evaluando modelos optimizados...")

    # Crear DataFrame
    datos_comparacion = []
    for nombre, resultado in resultados_optimizados.items():
        datos_comparacion.append(
            {
                "Modelo": nombre,
                "R2_CV": resultado["r2_cv"],
                "R2_Train": resultado["r2_train"],
                "R2_Test": resultado["r2_test"],
                "MAE_Test": resultado["mae_test"],
                "RMSE_Test": resultado["rmse_test"],
                "Overfitting": resultado["overfitting"],
            }
        )

    df_comparacion = pd.DataFrame(datos_comparacion)

    # Ordenar por R2 test descendente
    df_comparacion = df_comparacion.sort_values(
        "R2_Test", ascending=False
    ).reset_index(drop=True)

    logger.info(f"[OK] Comparación generada: {len(df_comparacion)} modelos")
    logger.info("\nRanking final (post-GridSearch):")
    for idx, row in df_comparacion.iterrows():
        objetivo = "[OBJETIVO]" if row["R2_Test"] >= 0.80 else ""
        logger.info(
            f"  {idx+1}. {row['Modelo']}: R2={row['R2_Test']:.4f}, "
            f"MAE={row['MAE_Test']:.2f} {objetivo}"
        )

    return df_comparacion


# =============================================================================
# NODE 6: GUARDAR MEJOR MODELO FINAL
# =============================================================================

def save_final_model(
    resultados_optimizados: Dict[str, Dict[str, Any]],
    df_comparacion_optimizada: pd.DataFrame,
) -> Tuple[Any, str, Dict[str, float]]:
    """
    Selecciona y retorna el mejor modelo final con sus métricas.

    Args:
        resultados_optimizados: Dict con modelos optimizados
        df_comparacion_optimizada: DataFrame con comparación

    Returns:
        Tupla (mejor_modelo, nombre_mejor_modelo, metricas_finales)
    """
    logger.info("[SAVE] Seleccionando mejor modelo final...")

    # Mejor modelo por R2 test
    mejor_nombre = df_comparacion_optimizada.iloc[0]["Modelo"]
    mejor_resultado = resultados_optimizados[mejor_nombre]

    logger.info(f"\n[OK] Mejor modelo: {mejor_nombre}")
    logger.info(f"  R2 Test: {mejor_resultado['r2_test']:.4f}")
    logger.info(f"  MAE Test: {mejor_resultado['mae_test']:.2f} posiciones")
    logger.info(f"  RMSE Test: {mejor_resultado['rmse_test']:.2f} posiciones")
    logger.info(f"  Overfitting: {mejor_resultado['overfitting']:.4f}")

    # Verificar objetivo
    if mejor_resultado["r2_test"] >= 0.80:
        logger.info("\n[OBJETIVO ALCANZADO] R2 >= 0.80")
    else:
        brecha = 0.80 - mejor_resultado["r2_test"]
        logger.info(f"\n[OBJETIVO] Falta {brecha:.4f} puntos para R2 = 0.80")

    # Preparar salidas
    modelo_final = mejor_resultado["modelo"]
    metricas_finales = {
        "r2_cv": mejor_resultado["r2_cv"],
        "r2_train": mejor_resultado["r2_train"],
        "r2_test": mejor_resultado["r2_test"],
        "mae_test": mejor_resultado["mae_test"],
        "rmse_test": mejor_resultado["rmse_test"],
        "overfitting": mejor_resultado["overfitting"],
    }

    return modelo_final, mejor_nombre, metricas_finales
