"""
Tests Unitarios para Nodos del Pipeline regresion_data
=======================================================

Este módulo contiene tests comprehensivos para validar la funcionalidad
de cada nodo del pipeline de preparación de datos para regresión.

Cubre:
- Carga de datos
- Merge de tablas
- Filtrado temporal
- Eliminación de data leakage
- Split train/test
- Feature engineering
- Imputación de valores faltantes
- Encoding y escalado

Autor: F1 ML Testing Team
Fecha: 2025-11-13
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.f1_ml.pipelines.regresion_data.nodes import (
    load_raw_data,
    merge_tables,
    filter_modern_era,
    remove_dnfs,
    drop_leakage_features,
    split_train_test,
    impute_missing_values,
)


# =============================================================================
# FIXTURES - Datos de prueba
# =============================================================================

@pytest.fixture
def sample_race_results():
    """DataFrame de muestra para resultados de carreras."""
    return pd.DataFrame({
        'resultId': [1, 2, 3, 4, 5],
        'raceId': [100, 100, 100, 101, 101],
        'driverId': [1, 2, 3, 1, 2],
        'constructorId': [10, 10, 20, 10, 20],
        'position': [1.0, 2.0, 3.0, 1.0, 2.0],
        'positionText': ['1', '2', '3', '1', '2'],
        'positionOrder': [1, 2, 3, 1, 2],
        'points': [25.0, 18.0, 15.0, 25.0, 18.0],
        'grid': [1, 3, 5, 2, 4],
        'laps': [58, 58, 58, 70, 70],
        'statusId': [1, 1, 1, 1, 1],
    })


@pytest.fixture
def sample_qualifying_results():
    """DataFrame de muestra para resultados de qualifying."""
    return pd.DataFrame({
        'raceId': [100, 100, 100, 101, 101],
        'driverId': [1, 2, 3, 1, 2],
        'position': [1, 3, 5, 2, 4],
    })


@pytest.fixture
def sample_driver_rankings():
    """DataFrame de muestra para rankings de pilotos."""
    return pd.DataFrame({
        'raceId': [100, 100, 100, 101, 101],
        'driverId': [1, 2, 3, 1, 2],
        'points': [100.0, 80.0, 60.0, 125.0, 98.0],
        'position': [1, 2, 3, 1, 2],
        'wins': [4, 2, 1, 5, 2],
    })


@pytest.fixture
def sample_constructor_rankings():
    """DataFrame de muestra para rankings de constructores."""
    return pd.DataFrame({
        'raceId': [100, 100, 100, 101, 101],
        'constructorId': [10, 10, 20, 10, 20],
        'points': [200.0, 200.0, 150.0, 250.0, 170.0],
        'position': [1, 1, 2, 1, 2],
        'wins': [6, 6, 3, 7, 3],
    })


@pytest.fixture
def sample_track_information():
    """DataFrame de muestra para información de circuitos."""
    return pd.DataFrame({
        'circuitId': [1, 2],
        'name': ['Monza', 'Monaco'],
        'location': ['Monza', 'Monte Carlo'],
        'country': ['Italy', 'Monaco'],
        'lat': [45.6156, 43.7347],
        'lng': [9.2811, 7.4206],
        'alt': [146.0, 7.0],
    })


@pytest.fixture
def sample_race_schedule():
    """DataFrame de muestra para calendario de carreras."""
    return pd.DataFrame({
        'raceId': [100, 101],
        'year': [2020, 2021],
        'round': [1, 1],
        'circuitId': [1, 2],
        'name': ['Italian GP', 'Monaco GP'],
        'date': ['2020-09-06', '2021-05-23'],
    })


@pytest.fixture
def sample_merged_data():
    """DataFrame de muestra con datos ya mergeados."""
    return pd.DataFrame({
        'resultId': [1, 2, 3, 4, 5],
        'raceId': [100, 100, 100, 101, 101],
        'driverId': [1, 2, 3, 1, 2],
        'year': [2009, 2009, 2009, 2015, 2015],
        'position': [1.0, 2.0, 3.0, 1.0, 2.0],
        'positionText': ['1', '2', '3', '1', '2'],
        'positionOrder': [1, 2, 3, 1, 2],
        'points': [25.0, 18.0, 15.0, 25.0, 18.0],
        'grid': [1, 3, 5, 2, 4],
        'statusId': [1, 1, 11, 1, 1],  # statusId 11 = DNF
        'laps': [58, 58, 45, 70, 70],  # El 3er corredor no completó
    })


@pytest.fixture
def sample_parameters():
    """Parámetros de configuración de muestra."""
    return {
        'modern_era_start': 2010,
        'modern_era_end': 2024,
        'test_size': 0.20,
        'random_state': 42,
        'leakage_features': [
            'positionText',
            'positionOrder',
            'points',
            'time',
            'milliseconds',
            'rank',
            'fastestLap',
            'fastestLapTime',
            'fastestLapSpeed',
            'statusId',
            'laps',
            'resultId',
            'number',
            'q1',
            'q2',
            'q3',
        ]
    }


# =============================================================================
# TESTS - Node 1: load_raw_data
# =============================================================================

def test_load_raw_data_returns_dict(
    sample_race_results,
    sample_qualifying_results,
    sample_driver_rankings,
    sample_constructor_rankings,
    sample_track_information,
    sample_race_schedule
):
    """Test que load_raw_data retorna un diccionario con las 6 tablas."""
    result = load_raw_data(
        race_results=sample_race_results,
        qualifying_results=sample_qualifying_results,
        driver_rankings=sample_driver_rankings,
        constructor_rankings=sample_constructor_rankings,
        track_information=sample_track_information,
        race_schedule=sample_race_schedule,
    )

    assert isinstance(result, dict), "Debe retornar un diccionario"
    assert len(result) == 6, "Debe contener 6 tablas"
    assert all(key in result for key in [
        'race_results', 'qualifying_results', 'driver_rankings',
        'constructor_rankings', 'track_information', 'race_schedule'
    ]), "Debe contener todas las claves esperadas"


def test_load_raw_data_preserves_data(sample_race_results):
    """Test que load_raw_data preserva los datos sin modificarlos."""
    original_shape = sample_race_results.shape

    result = load_raw_data(
        race_results=sample_race_results,
        qualifying_results=pd.DataFrame(),
        driver_rankings=pd.DataFrame(),
        constructor_rankings=pd.DataFrame(),
        track_information=pd.DataFrame(),
        race_schedule=pd.DataFrame(),
    )

    assert result['race_results'].shape == original_shape, \
        "No debe modificar el tamaño de los DataFrames"


# =============================================================================
# TESTS - Node 2: merge_tables
# =============================================================================

def test_merge_tables_combines_all_tables(
    sample_race_results,
    sample_qualifying_results,
    sample_driver_rankings,
    sample_constructor_rankings,
    sample_track_information,
    sample_race_schedule
):
    """Test que merge_tables combina todas las tablas correctamente."""
    tables = {
        'race_results': sample_race_results,
        'qualifying_results': sample_qualifying_results,
        'driver_rankings': sample_driver_rankings,
        'constructor_rankings': sample_constructor_rankings,
        'track_information': sample_track_information,
        'race_schedule': sample_race_schedule,
    }

    result = merge_tables(tables)

    assert isinstance(result, pd.DataFrame), "Debe retornar un DataFrame"
    assert len(result) > 0, "El DataFrame no debe estar vacío"

    # Verificar que se agregaron columnas de diferentes tablas
    expected_columns = ['year', 'round', 'position_quali', 'puntos_campeonato_piloto']
    for col in expected_columns:
        assert col in result.columns, f"Debe contener la columna {col} después del merge"


def test_merge_tables_preserves_race_results_rows(
    sample_race_results,
    sample_qualifying_results,
    sample_driver_rankings,
    sample_constructor_rankings,
    sample_track_information,
    sample_race_schedule
):
    """Test que merge_tables preserva el número de filas de race_results (left join)."""
    tables = {
        'race_results': sample_race_results,
        'qualifying_results': sample_qualifying_results,
        'driver_rankings': sample_driver_rankings,
        'constructor_rankings': sample_constructor_rankings,
        'track_information': sample_track_information,
        'race_schedule': sample_race_schedule,
    }

    result = merge_tables(tables)

    # En left join, debe mantener todas las filas de race_results
    assert len(result) == len(sample_race_results), \
        "El merge debe preservar todas las filas de race_results"


# =============================================================================
# TESTS - Node 3: filter_modern_era
# =============================================================================

def test_filter_modern_era_filters_by_year(sample_merged_data, sample_parameters):
    """Test que filter_modern_era filtra correctamente por año."""
    result = filter_modern_era(sample_merged_data, sample_parameters)

    # Verificar que solo quedan años >= 2010
    assert all(result['year'] >= sample_parameters['modern_era_start']), \
        "Todos los años deben ser >= modern_era_start"
    assert all(result['year'] <= sample_parameters['modern_era_end']), \
        "Todos los años deben ser <= modern_era_end"


def test_filter_modern_era_reduces_data(sample_merged_data, sample_parameters):
    """Test que filter_modern_era reduce el dataset al filtrar años antiguos."""
    original_count = len(sample_merged_data)
    result = filter_modern_era(sample_merged_data, sample_parameters)

    # Como los datos de muestra tienen años 2009, 2015, debería filtrar 2009
    assert len(result) < original_count, \
        "Debe filtrar filas con años fuera del rango moderno"


# =============================================================================
# TESTS - Node 4: remove_dnfs
# =============================================================================

def test_remove_dnfs_filters_status(sample_merged_data):
    """Test que remove_dnfs elimina correctamente los DNFs."""
    # En los datos de muestra, hay un registro con statusId=11 (DNF)
    result = remove_dnfs(sample_merged_data)

    # Verificar que el statusId=11 fue eliminado
    assert 11 not in result['statusId'].values, \
        "No deben quedar registros con statusId diferente de 1"

    # Solo debe quedar statusId=1 (finalizaron la carrera)
    assert all(result['statusId'] == 1), \
        "Todos los registros deben tener statusId=1"


def test_remove_dnfs_reduces_dataset(sample_merged_data):
    """Test que remove_dnfs reduce el tamaño del dataset."""
    original_count = len(sample_merged_data)
    result = remove_dnfs(sample_merged_data)

    assert len(result) < original_count, \
        "Debe eliminar al menos un DNF del dataset"


# =============================================================================
# TESTS - Node 5: drop_leakage_features
# =============================================================================

def test_drop_leakage_features_removes_columns(sample_merged_data, sample_parameters):
    """Test que drop_leakage_features elimina las columnas prohibidas."""
    leakage_features = sample_parameters['leakage_features']

    result = drop_leakage_features(sample_merged_data, sample_parameters)

    # Verificar que las columnas prohibidas fueron eliminadas
    for feature in leakage_features:
        assert feature not in result.columns, \
            f"La columna {feature} debe ser eliminada (data leakage)"


def test_drop_leakage_features_preserves_other_columns(sample_merged_data, sample_parameters):
    """Test que drop_leakage_features preserva las columnas permitidas."""
    allowed_columns = ['year', 'grid', 'driverId', 'raceId']

    result = drop_leakage_features(sample_merged_data, sample_parameters)

    # Verificar que las columnas permitidas siguen presentes
    for col in allowed_columns:
        if col in sample_merged_data.columns:
            assert col in result.columns, \
                f"La columna {col} debe ser preservada"


# =============================================================================
# TESTS - Node 6: split_train_test
# =============================================================================

def test_split_train_test_returns_four_arrays(sample_merged_data, sample_parameters):
    """Test que split_train_test retorna 4 arrays (X_train, X_test, y_train, y_test)."""
    # Preparar datos de muestra
    df_clean = sample_merged_data.copy()
    df_clean['position_final'] = df_clean['position']  # Crear target

    result = split_train_test(df_clean, sample_parameters)

    assert isinstance(result, tuple), "Debe retornar una tupla"
    assert len(result) == 4, "Debe retornar 4 elementos"

    X_train, X_test, y_train, y_test = result

    assert isinstance(X_train, np.ndarray), "X_train debe ser numpy array"
    assert isinstance(X_test, np.ndarray), "X_test debe ser numpy array"
    assert isinstance(y_train, np.ndarray), "y_train debe ser numpy array"
    assert isinstance(y_test, np.ndarray), "y_test debe ser numpy array"


def test_split_train_test_respects_test_size(sample_merged_data, sample_parameters):
    """Test que split_train_test respeta el test_size configurado."""
    df_clean = sample_merged_data.copy()
    df_clean['position_final'] = df_clean['position']

    X_train, X_test, y_train, y_test = split_train_test(df_clean, sample_parameters)

    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples

    # Verificar que el test_size sea aproximadamente 0.20 (con margen de error)
    assert abs(test_ratio - sample_parameters['test_size']) < 0.05, \
        f"El test_size debe ser aproximadamente {sample_parameters['test_size']}"


def test_split_train_test_no_data_leakage(sample_merged_data, sample_parameters):
    """Test que split_train_test no tiene overlap entre train y test."""
    df_clean = sample_merged_data.copy()
    df_clean['position_final'] = df_clean['position']

    X_train, X_test, y_train, y_test = split_train_test(df_clean, sample_parameters)

    # Verificar que no hay filas duplicadas entre train y test
    # (esto es garantizado por train_test_split, pero lo verificamos)
    assert len(X_train) + len(X_test) == len(df_clean.drop(columns=['position_final'])), \
        "El split debe preservar todas las filas sin duplicados"


# =============================================================================
# TESTS - Node 7: impute_missing_values
# =============================================================================

def test_impute_missing_values_fills_nans():
    """Test que impute_missing_values rellena valores faltantes."""
    # Crear datos con NaNs
    X_train = np.array([
        [1.0, 2.0, np.nan],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ])

    X_test = np.array([
        [1.0, np.nan, 3.0],
    ])

    X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test)

    # Verificar que no quedan NaNs
    assert not np.isnan(X_train_imputed).any(), \
        "X_train no debe contener NaNs después de imputación"
    assert not np.isnan(X_test_imputed).any(), \
        "X_test no debe contener NaNs después de imputación"


def test_impute_missing_values_preserves_shape():
    """Test que impute_missing_values preserva el shape de los arrays."""
    X_train = np.array([[1.0, np.nan], [3.0, 4.0]])
    X_test = np.array([[5.0, np.nan]])

    original_train_shape = X_train.shape
    original_test_shape = X_test.shape

    X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test)

    assert X_train_imputed.shape == original_train_shape, \
        "X_train debe mantener su shape original"
    assert X_test_imputed.shape == original_test_shape, \
        "X_test debe mantener su shape original"


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

def test_full_pipeline_integration(
    sample_race_results,
    sample_qualifying_results,
    sample_driver_rankings,
    sample_constructor_rankings,
    sample_track_information,
    sample_race_schedule,
    sample_parameters
):
    """Test de integración que ejecuta múltiples nodos secuencialmente."""
    # 1. Load raw data
    tables = load_raw_data(
        race_results=sample_race_results,
        qualifying_results=sample_qualifying_results,
        driver_rankings=sample_driver_rankings,
        constructor_rankings=sample_constructor_rankings,
        track_information=sample_track_information,
        race_schedule=sample_race_schedule,
    )

    # 2. Merge tables
    merged = merge_tables(tables)

    # Verificar que el merge fue exitoso
    assert len(merged) > 0, "El merge debe producir datos"
    assert 'year' in merged.columns, "El merge debe agregar columna 'year'"

    # 3. Drop leakage features
    cleaned = drop_leakage_features(merged, sample_parameters)

    # Verificar que se eliminaron columnas prohibidas
    assert 'points' not in cleaned.columns, \
        "La columna 'points' debe ser eliminada (data leakage)"


# =============================================================================
# TESTS DE VALIDACIÓN DE DATOS
# =============================================================================

@pytest.mark.parametrize("invalid_year", [-1, 0, 3000])
def test_filter_modern_era_handles_invalid_years(sample_merged_data, sample_parameters, invalid_year):
    """Test que filter_modern_era maneja años inválidos correctamente."""
    df = sample_merged_data.copy()
    df.loc[0, 'year'] = invalid_year

    result = filter_modern_era(df, sample_parameters)

    # Los años inválidos deben ser filtrados
    assert invalid_year not in result['year'].values, \
        f"El año inválido {invalid_year} debe ser filtrado"


def test_merge_tables_handles_empty_qualifying(
    sample_race_results,
    sample_driver_rankings,
    sample_constructor_rankings,
    sample_track_information,
    sample_race_schedule
):
    """Test que merge_tables maneja qualifying vacío sin errores."""
    tables = {
        'race_results': sample_race_results,
        'qualifying_results': pd.DataFrame(columns=['raceId', 'driverId', 'position']),
        'driver_rankings': sample_driver_rankings,
        'constructor_rankings': sample_constructor_rankings,
        'track_information': sample_track_information,
        'race_schedule': sample_race_schedule,
    }

    result = merge_tables(tables)

    # Debe ejecutar sin errores
    assert isinstance(result, pd.DataFrame), "Debe retornar un DataFrame"
    assert len(result) > 0, "Debe retornar datos aunque qualifying esté vacío"
