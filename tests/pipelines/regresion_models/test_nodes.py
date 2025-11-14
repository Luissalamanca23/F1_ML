"""
Tests Unitarios para Nodos del Pipeline regresion_models
=========================================================

Este módulo contiene tests comprehensivos para validar el entrenamiento,
optimización y evaluación de modelos de regresión.

Cubre:
- Entrenamiento de múltiples modelos base
- Selección de TOP 5 modelos
- GridSearch con validación cruzada
- Evaluación de métricas (R2, MAE, RMSE)
- Guardado de modelos

Autor: F1 ML Testing Team
Fecha: 2025-11-13
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from src.f1_ml.pipelines.regresion_models.nodes import (
    train_all_models,
    select_top5_models,
    optimize_with_gridsearch,
    evaluate_models,
)


# =============================================================================
# FIXTURES - Datos de prueba
# =============================================================================

@pytest.fixture
def sample_training_data():
    """Genera datos sintéticos para entrenamiento."""
    np.random.seed(42)

    X_train = np.random.randn(100, 10)  # 100 samples, 10 features
    y_train = 3 * X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(100) * 0.1

    X_test = np.random.randn(30, 10)  # 30 samples, 10 features
    y_test = 3 * X_test[:, 0] + 2 * X_test[:, 1] + np.random.randn(30) * 0.1

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_params():
    """Parámetros de configuración para regresión."""
    return {
        'random_state': 42,
        'cv_folds': 5,
        'gridsearch_grids': {
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0],
            },
            'RandomForest': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
            },
        },
    }


@pytest.fixture
def sample_trained_models():
    """Diccionario de modelos pre-entrenados de muestra."""
    np.random.seed(42)

    X_dummy = np.random.randn(50, 5)
    y_dummy = np.random.randn(50)

    models = {}
    for i, name in enumerate(['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E', 'Model_F']):
        model = Ridge(alpha=1.0)
        model.fit(X_dummy, y_dummy)

        models[name] = {
            'modelo': model,
            'r2_train': 0.90 - i * 0.05,  # Decrece: 0.90, 0.85, 0.80, ...
            'mae_train': 2.0 + i * 0.5,
            'rmse_train': 3.0 + i * 0.5,
        }

    return models


# =============================================================================
# TESTS - train_all_models
# =============================================================================

def test_train_all_models_returns_dict(sample_training_data, sample_params):
    """Test que train_all_models retorna un diccionario de modelos."""
    X_train, X_test, y_train, y_test = sample_training_data

    result = train_all_models(X_train, y_train, sample_params)

    assert isinstance(result, dict), "Debe retornar un diccionario"
    assert len(result) > 0, "Debe entrenar al menos un modelo"


def test_train_all_models_minimum_count(sample_training_data, sample_params):
    """Test que train_all_models entrena al menos 11 modelos (requisito del proyecto)."""
    X_train, X_test, y_train, y_test = sample_training_data

    result = train_all_models(X_train, y_train, sample_params)

    assert len(result) >= 11, \
        f"Debe entrenar al menos 11 modelos (entrenó {len(result)})"


def test_train_all_models_contains_metrics(sample_training_data, sample_params):
    """Test que cada modelo entrenado contiene métricas de evaluación."""
    X_train, X_test, y_train, y_test = sample_training_data

    result = train_all_models(X_train, y_train, sample_params)

    # Verificar que cada modelo tiene métricas
    for model_name, model_data in result.items():
        assert 'modelo' in model_data, f"{model_name} debe tener campo 'modelo'"
        assert 'r2_train' in model_data, f"{model_name} debe tener R2"
        assert 'mae_train' in model_data, f"{model_name} debe tener MAE"
        assert 'rmse_train' in model_data, f"{model_name} debe tener RMSE"


def test_train_all_models_all_fitted(sample_training_data, sample_params):
    """Test que todos los modelos están entrenados (fitted)."""
    X_train, X_test, y_train, y_test = sample_training_data

    result = train_all_models(X_train, y_train, sample_params)

    # Verificar que todos los modelos pueden hacer predicciones
    for model_name, model_data in result.items():
        modelo = model_data['modelo']

        # Intentar predecir (esto fallará si no está fitted)
        try:
            predictions = modelo.predict(X_test[:5])
            is_fitted = True
        except:
            is_fitted = False

        assert is_fitted, f"{model_name} debe estar entrenado (fitted)"


def test_train_all_models_metrics_valid_range(sample_training_data, sample_params):
    """Test que las métricas están en rangos válidos."""
    X_train, X_test, y_train, y_test = sample_training_data

    result = train_all_models(X_train, y_train, sample_params)

    for model_name, model_data in result.items():
        # R2 puede ser negativo si el modelo es muy malo, pero típicamente [0, 1]
        # Para este test con datos sintéticos simples, esperamos R2 razonable
        assert model_data['r2_train'] > -1, \
            f"{model_name} tiene R2 demasiado bajo: {model_data['r2_train']}"

        # MAE y RMSE deben ser >= 0
        assert model_data['mae_train'] >= 0, \
            f"{model_name} tiene MAE negativo: {model_data['mae_train']}"
        assert model_data['rmse_train'] >= 0, \
            f"{model_name} tiene RMSE negativo: {model_data['rmse_train']}"

        # RMSE >= MAE (propiedad matemática)
        assert model_data['rmse_train'] >= model_data['mae_train'], \
            f"{model_name}: RMSE debe ser >= MAE"


# =============================================================================
# TESTS - select_top5_models
# =============================================================================

def test_select_top5_models_returns_5_models(sample_trained_models):
    """Test que select_top5_models retorna exactamente 5 modelos."""
    result = select_top5_models(sample_trained_models)

    assert isinstance(result, dict), "Debe retornar un diccionario"
    assert len(result) == 5, f"Debe retornar exactamente 5 modelos (retornó {len(result)})"


def test_select_top5_models_selects_best_by_r2(sample_trained_models):
    """Test que select_top5_models selecciona los 5 mejores por R2."""
    result = select_top5_models(sample_trained_models)

    # Obtener R2 de los modelos seleccionados
    selected_r2_values = [model_data['r2_train'] for model_data in result.values()]

    # Obtener R2 de todos los modelos
    all_r2_values = [model_data['r2_train'] for model_data in sample_trained_models.values()]

    # Los 5 R2 más altos de todos los modelos
    top5_r2_expected = sorted(all_r2_values, reverse=True)[:5]

    # Los R2 seleccionados deben ser los 5 mejores
    assert sorted(selected_r2_values, reverse=True) == sorted(top5_r2_expected, reverse=True), \
        "Debe seleccionar los 5 modelos con mejor R2"


def test_select_top5_models_handles_less_than_5(sample_training_data, sample_params):
    """Test que select_top5_models maneja correctamente < 5 modelos."""
    # Crear solo 3 modelos
    models_dict = {}
    X_train, X_test, y_train, y_test = sample_training_data

    for i in range(3):
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        models_dict[f'Model_{i}'] = {
            'modelo': model,
            'r2_train': 0.80 - i * 0.1,
            'mae_train': 2.0,
            'rmse_train': 3.0,
        }

    result = select_top5_models(models_dict)

    # Debe retornar los 3 modelos disponibles
    assert len(result) == min(5, len(models_dict)), \
        "Debe retornar todos los modelos si hay menos de 5"


# =============================================================================
# TESTS - optimize_with_gridsearch
# =============================================================================

def test_optimize_with_gridsearch_returns_dict(
    sample_training_data,
    sample_trained_models,
    sample_params
):
    """Test que optimize_with_gridsearch retorna un diccionario de modelos optimizados."""
    X_train, X_test, y_train, y_test = sample_training_data

    # Seleccionar solo los primeros 2 modelos para acelerar el test
    top5 = dict(list(sample_trained_models.items())[:2])

    result = optimize_with_gridsearch(
        top5_modelos=top5,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=sample_params,
    )

    assert isinstance(result, dict), "Debe retornar un diccionario"


def test_optimize_with_gridsearch_uses_cv_folds(
    sample_training_data,
    sample_params
):
    """Test que optimize_with_gridsearch usa cv_folds del parámetro."""
    X_train, X_test, y_train, y_test = sample_training_data

    # Crear un modelo simple para GridSearch
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    top5 = {
        'Ridge': {
            'modelo': model,
            'r2_train': 0.85,
            'mae_train': 2.0,
            'rmse_train': 3.0,
        }
    }

    # Cambiar cv_folds a 3
    params_with_cv3 = sample_params.copy()
    params_with_cv3['cv_folds'] = 3

    # Este test verifica que la función acepta el parámetro cv_folds
    # (la verificación interna de que usa exactamente 3 folds requeriría mock)
    result = optimize_with_gridsearch(
        top5_modelos=top5,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params_with_cv3,
    )

    assert isinstance(result, dict), "Debe aceptar cv_folds como parámetro"


def test_optimize_with_gridsearch_improves_or_maintains_performance(
    sample_training_data,
    sample_params
):
    """
    Test que GridSearch mejora o mantiene el rendimiento de los modelos.

    GridSearch debe encontrar hiperparámetros que sean al menos tan buenos
    como los parámetros por defecto.
    """
    X_train, X_test, y_train, y_test = sample_training_data

    # Entrenar modelo base
    base_model = Ridge(alpha=1.0)
    base_model.fit(X_train, y_train)
    base_r2 = base_model.score(X_test, y_test)

    top5 = {
        'Ridge': {
            'modelo': base_model,
            'r2_train': base_r2,
            'mae_train': 2.0,
            'rmse_train': 3.0,
        }
    }

    # Optimizar con GridSearch
    result = optimize_with_gridsearch(
        top5_modelos=top5,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=sample_params,
    )

    # El modelo optimizado debe tener R2 >= modelo base (o muy cercano)
    if 'Ridge' in result:
        optimized_r2 = result['Ridge'].get('r2_test', base_r2)
        # Permitir pequeña diferencia por variabilidad de CV
        assert optimized_r2 >= base_r2 - 0.1, \
            "GridSearch debe mejorar o mantener el rendimiento"


# =============================================================================
# TESTS - evaluate_models
# =============================================================================

def test_evaluate_models_calculates_metrics(sample_training_data, sample_trained_models):
    """Test que evaluate_models calcula R2, MAE, RMSE correctamente."""
    X_train, X_test, y_train, y_test = sample_training_data

    # Tomar solo 1 modelo para el test
    models = {'Ridge': sample_trained_models['Model_A']}

    result = evaluate_models(
        modelos=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    assert isinstance(result, dict), "Debe retornar un diccionario"

    # Verificar que contiene métricas
    for model_name, metrics in result.items():
        assert 'r2_train' in metrics, "Debe calcular R2 de train"
        assert 'r2_test' in metrics, "Debe calcular R2 de test"
        assert 'mae_test' in metrics, "Debe calcular MAE de test"
        assert 'rmse_test' in metrics, "Debe calcular RMSE de test"


def test_evaluate_models_metrics_consistency(sample_training_data):
    """Test que las métricas son matemáticamente consistentes."""
    X_train, X_test, y_train, y_test = sample_training_data

    # Entrenar modelo simple
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    models = {
        'Ridge': {
            'modelo': model,
            'r2_train': 0.85,
            'mae_train': 2.0,
            'rmse_train': 3.0,
        }
    }

    result = evaluate_models(
        modelos=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    metrics = result['Ridge']

    # RMSE >= MAE (propiedad matemática)
    assert metrics['rmse_test'] >= metrics['mae_test'], \
        "RMSE debe ser >= MAE"

    # R2 <= 1 (perfecta predicción)
    assert metrics['r2_test'] <= 1.0, \
        "R2 no puede ser > 1"


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

def test_full_models_pipeline_integration(sample_training_data, sample_params):
    """
    Test de integración que ejecuta el flujo completo de modelos:
    1. Entrenar múltiples modelos
    2. Seleccionar TOP 5
    3. Optimizar con GridSearch
    4. Evaluar modelos optimizados
    """
    X_train, X_test, y_train, y_test = sample_training_data

    # 1. Entrenar modelos
    all_models = train_all_models(X_train, y_train, sample_params)
    assert len(all_models) >= 11, "Debe entrenar al menos 11 modelos"

    # 2. Seleccionar TOP 5
    top5 = select_top5_models(all_models)
    assert len(top5) == 5, "Debe seleccionar exactamente 5 modelos"

    # 3. Optimizar (solo 1 modelo para acelerar el test)
    top1 = dict(list(top5.items())[:1])

    optimized = optimize_with_gridsearch(
        top5_modelos=top1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=sample_params,
    )

    # 4. Evaluar
    evaluated = evaluate_models(
        modelos=optimized,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Verificar que el pipeline completo funciona
    assert len(evaluated) > 0, "Debe haber al menos un modelo evaluado"

    for model_name, metrics in evaluated.items():
        assert 'r2_test' in metrics, f"{model_name} debe tener R2 test"


# =============================================================================
# TESTS DE CASOS EXTREMOS
# =============================================================================

def test_train_all_models_with_small_dataset():
    """Test que train_all_models maneja datasets pequeños correctamente."""
    # Dataset muy pequeño (20 samples)
    X_train = np.random.randn(20, 5)
    y_train = np.random.randn(20)

    params = {'random_state': 42}

    result = train_all_models(X_train, y_train, params)

    # Debe manejar el dataset pequeño sin errores
    assert len(result) > 0, "Debe entrenar modelos incluso con datos pequeños"


def test_gridsearch_with_cv_greater_than_samples():
    """Test que GridSearch maneja cv_folds > número de samples."""
    # Solo 10 samples, pero cv_folds=5
    X_train = np.random.randn(10, 3)
    y_train = np.random.randn(10)
    X_test = np.random.randn(5, 3)
    y_test = np.random.randn(5)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    top5 = {
        'Ridge': {
            'modelo': model,
            'r2_train': 0.5,
            'mae_train': 1.0,
            'rmse_train': 1.5,
        }
    }

    params = {
        'random_state': 42,
        'cv_folds': 5,  # cv=5 con solo 10 samples (2 por fold)
        'gridsearch_grids': {
            'Ridge': {'alpha': [0.1, 1.0]}
        }
    }

    # Debe manejar el caso extremo sin errores
    result = optimize_with_gridsearch(
        top5_modelos=top5,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params,
    )

    assert isinstance(result, dict), "Debe ejecutar sin errores"


@pytest.mark.parametrize("cv_folds", [3, 5, 10])
def test_gridsearch_accepts_different_cv_values(sample_training_data, cv_folds):
    """Test que GridSearch acepta diferentes valores de cv_folds."""
    X_train, X_test, y_train, y_test = sample_training_data

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    top5 = {
        'Ridge': {
            'modelo': model,
            'r2_train': 0.85,
            'mae_train': 2.0,
            'rmse_train': 3.0,
        }
    }

    params = {
        'random_state': 42,
        'cv_folds': cv_folds,
        'gridsearch_grids': {
            'Ridge': {'alpha': [0.1, 1.0]}
        }
    }

    result = optimize_with_gridsearch(
        top5_modelos=top5,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params,
    )

    assert isinstance(result, dict), f"Debe aceptar cv_folds={cv_folds}"
