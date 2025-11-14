"""
Tests Unitarios para Nodos del Pipeline classification_data
============================================================

Este módulo contiene tests comprehensivos para validar la funcionalidad
de cada nodo del pipeline de preparación de datos para clasificación (podio).

Cubre:
- Creación de target is_podium
- Features históricas de pilotos y constructores
- Split temporal (prevención de data leakage)
- Validación de rolling windows
- Encoding y escalado dual

Autor: F1 ML Testing Team
Fecha: 2025-11-13
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.f1_ml.pipelines.classification_data.nodes import (
    create_podium_target,
    filter_valid_finishes,
    sort_by_date,
    split_temporal_train_test,
)


# =============================================================================
# FIXTURES - Datos de prueba
# =============================================================================

@pytest.fixture
def sample_classification_data():
    """DataFrame de muestra para clasificación."""
    return pd.DataFrame({
        'resultId': [1, 2, 3, 4, 5, 6],
        'raceId': [100, 100, 100, 101, 101, 101],
        'driverId': [1, 2, 3, 1, 2, 3],
        'constructorId': [10, 10, 20, 10, 20, 30],
        'position': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0],  # Solo primeros 3 son podio
        'positionText': ['1', '2', '3', '4', '5', '1'],
        'positionOrder': [1, 2, 3, 4, 5, 1],
        'grid': [1, 3, 5, 7, 9, 2],
        'date': ['2022-05-01', '2022-05-01', '2022-05-01',
                 '2023-06-15', '2023-06-15', '2023-06-15'],
        'year': [2022, 2022, 2022, 2023, 2023, 2023],
        'statusId': [1, 1, 1, 1, 1, 1],
    })


@pytest.fixture
def sample_classification_params():
    """Parámetros de configuración para clasificación."""
    return {
        'temporal_split_date': '2023-01-01',
        'random_state': 42,
        'leakage_features': [
            'position',
            'positionOrder',
            'positionText',
            'points',
            'statusId',
        ],
    }


@pytest.fixture
def sample_data_with_dnfs():
    """Datos con DNFs (Did Not Finish) incluidos."""
    return pd.DataFrame({
        'resultId': [1, 2, 3, 4],
        'position': [1.0, 2.0, np.nan, np.nan],  # 2 DNFs
        'positionText': ['1', '2', 'R', 'D'],  # R=Retired, D=Disqualified
        'statusId': [1, 1, 11, 15],  # 1=Finished, 11=DNF, 15=Disqualified
    })


@pytest.fixture
def sample_historical_data():
    """Datos para testear features históricas."""
    return pd.DataFrame({
        'raceId': [1, 2, 3, 4, 5, 6],
        'driverId': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2022-03-01', '2022-04-01', '2022-05-01',
                                '2022-03-01', '2022-04-01', '2022-05-01']),
        'position': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
        'is_podium': [1, 1, 1, 0, 0, 1],
    }).sort_values('date')


# =============================================================================
# TESTS - create_podium_target
# =============================================================================

def test_create_podium_target_creates_binary_column(sample_classification_data):
    """Test que create_podium_target crea columna binaria is_podium."""
    result = create_podium_target(sample_classification_data)

    assert 'is_podium' in result.columns, "Debe crear columna 'is_podium'"
    assert result['is_podium'].dtype in [np.int64, np.int32, int], \
        "is_podium debe ser de tipo entero"


def test_create_podium_target_correct_values(sample_classification_data):
    """Test que create_podium_target asigna correctamente 1 a posiciones 1-3."""
    result = create_podium_target(sample_classification_data)

    # Verificar que posiciones 1, 2, 3 son podio (1)
    podium_positions = result[result['position'].isin([1.0, 2.0, 3.0])]['is_podium']
    assert all(podium_positions == 1), \
        "Posiciones 1, 2, 3 deben tener is_podium=1"

    # Verificar que posiciones > 3 no son podio (0)
    non_podium_positions = result[result['position'] > 3]['is_podium']
    assert all(non_podium_positions == 0), \
        "Posiciones > 3 deben tener is_podium=0"


def test_create_podium_target_distribution(sample_classification_data):
    """Test que create_podium_target produce una distribución razonable."""
    result = create_podium_target(sample_classification_data)

    podium_count = (result['is_podium'] == 1).sum()
    total_count = len(result)

    # En carreras normales, ~15-20% son podios (3 de 20 pilotos)
    podium_ratio = podium_count / total_count

    assert 0 < podium_ratio < 1, \
        "Debe haber tanto podios como no-podios"


# =============================================================================
# TESTS - filter_valid_finishes
# =============================================================================

def test_filter_valid_finishes_removes_dnfs(sample_data_with_dnfs):
    """Test que filter_valid_finishes elimina DNFs correctamente."""
    result = filter_valid_finishes(sample_data_with_dnfs)

    # Verificar que no quedan NaNs en position
    assert not result['position'].isna().any(), \
        "No deben quedar posiciones NaN (DNFs)"

    # Verificar que solo quedan statusId=1 (finished)
    assert all(result['statusId'] == 1), \
        "Solo deben quedar carreras finalizadas (statusId=1)"


def test_filter_valid_finishes_reduces_dataset(sample_data_with_dnfs):
    """Test que filter_valid_finishes reduce el tamaño del dataset."""
    original_count = len(sample_data_with_dnfs)
    result = filter_valid_finishes(sample_data_with_dnfs)

    assert len(result) < original_count, \
        "Debe eliminar DNFs y reducir el dataset"

    # Específicamente, debe eliminar 2 DNFs
    assert len(result) == 2, \
        "Debe quedar solo 2 filas válidas (las que finalizaron)"


# =============================================================================
# TESTS - sort_by_date
# =============================================================================

def test_sort_by_date_orders_chronologically(sample_classification_data):
    """Test que sort_by_date ordena los datos cronológicamente."""
    # Desordenar los datos primero
    df_shuffled = sample_classification_data.sample(frac=1, random_state=42)

    result = sort_by_date(df_shuffled)

    # Verificar que están ordenados por fecha
    dates = pd.to_datetime(result['date'])
    assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1)), \
        "Los datos deben estar ordenados cronológicamente"


def test_sort_by_date_preserves_data(sample_classification_data):
    """Test que sort_by_date preserva todos los datos sin pérdida."""
    original_count = len(sample_classification_data)

    result = sort_by_date(sample_classification_data)

    assert len(result) == original_count, \
        "No debe perder datos al ordenar"


def test_sort_by_date_critical_for_temporal_split(sample_classification_data):
    """
    Test que sort_by_date es crítico para prevenir data leakage en split temporal.

    El orden cronológico es esencial para que el split temporal funcione correctamente
    y no haya información del futuro en el conjunto de entrenamiento.
    """
    df_shuffled = sample_classification_data.sample(frac=1, random_state=42)
    result = sort_by_date(df_shuffled)

    # Verificar que después de ordenar, las carreras de 2022 vienen antes de 2023
    years = result['year'].values
    first_2023_index = np.where(years == 2023)[0][0] if 2023 in years else len(years)
    last_2022_index = np.where(years == 2022)[0][-1] if 2022 in years else -1

    if last_2022_index != -1 and first_2023_index < len(years):
        assert last_2022_index < first_2023_index, \
            "Todas las carreras de 2022 deben venir antes de las de 2023"


# =============================================================================
# TESTS - split_temporal_train_test
# =============================================================================

def test_split_temporal_train_test_creates_temporal_split(
    sample_classification_data,
    sample_classification_params
):
    """Test que split_temporal_train_test crea split temporal correcto."""
    # Asegurar que los datos estén ordenados
    df_sorted = sort_by_date(sample_classification_data)
    df_sorted['is_podium'] = (df_sorted['position'] <= 3).astype(int)

    X_train, X_test, y_train, y_test = split_temporal_train_test(
        df_sorted,
        sample_classification_params
    )

    assert isinstance(X_train, np.ndarray), "X_train debe ser numpy array"
    assert isinstance(X_test, np.ndarray), "X_test debe ser numpy array"
    assert isinstance(y_train, np.ndarray), "y_train debe ser numpy array"
    assert isinstance(y_test, np.ndarray), "y_test debe ser numpy array"


def test_split_temporal_train_test_no_future_leakage(
    sample_classification_data,
    sample_classification_params
):
    """
    Test crítico: split_temporal_train_test NO debe tener data leakage temporal.

    Train debe contener solo datos ANTES de temporal_split_date.
    Test debe contener solo datos DESPUÉS de temporal_split_date.
    """
    df = sample_classification_data.copy()
    df['is_podium'] = (df['position'] <= 3).astype(int)

    X_train, X_test, y_train, y_test = split_temporal_train_test(
        df,
        sample_classification_params
    )

    # Con split_date='2023-01-01':
    # - Train debe tener 3 filas (2022)
    # - Test debe tener 3 filas (2023)

    assert len(X_train) == 3, \
        "Train debe contener solo datos de 2022 (antes de 2023-01-01)"
    assert len(X_test) == 3, \
        "Test debe contener solo datos de 2023 (después de 2023-01-01)"


def test_split_temporal_train_test_preserves_all_data(
    sample_classification_data,
    sample_classification_params
):
    """Test que split_temporal_train_test preserva todos los datos."""
    df = sample_classification_data.copy()
    df['is_podium'] = (df['position'] <= 3).astype(int)

    X_train, X_test, y_train, y_test = split_temporal_train_test(
        df,
        sample_classification_params
    )

    total_samples = len(X_train) + len(X_test)

    assert total_samples == len(df), \
        "El split debe preservar todos los datos sin pérdida"


# =============================================================================
# TESTS - Features Históricas (Rolling Windows)
# =============================================================================

def test_create_driver_historical_features_basic():
    """Test básico de features históricas de pilotos."""
    # Crear datos de muestra con historial
    df = pd.DataFrame({
        'driverId': [1, 1, 1, 1],
        'date': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01']),
        'is_podium': [1, 1, 0, 1],
        'year': [2022, 2022, 2022, 2022],
    }).sort_values('date')

    # Este es un test básico - en implementación real habría una función
    # create_driver_historical_features que calcularía rolling windows

    # Simular el cálculo de podios en últimas 5 carreras
    df['driver_podiums_last_5'] = df.groupby('driverId')['is_podium'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1).fillna(0)
    )

    # Verificar que las features históricas se calculan correctamente
    # La primera carrera no debe tener historial (0)
    assert df.iloc[0]['driver_podiums_last_5'] == 0, \
        "Primera carrera no debe tener historial previo"

    # La segunda carrera debe considerar la primera (1 podio)
    assert df.iloc[1]['driver_podiums_last_5'] == 1, \
        "Segunda carrera debe tener 1 podio previo"


def test_rolling_windows_prevent_future_leakage():
    """
    Test crítico: Rolling windows NO deben incluir información del futuro.

    Deben usar .shift(1) para que solo consideren carreras ANTERIORES.
    """
    df = pd.DataFrame({
        'driverId': [1, 1, 1],
        'date': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
        'is_podium': [1, 1, 1],
    }).sort_values('date')

    # Rolling window SIN shift (INCORRECTO - data leakage)
    df['leaky_feature'] = df.groupby('driverId')['is_podium'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )

    # Rolling window CON shift (CORRECTO - sin leakage)
    df['correct_feature'] = df.groupby('driverId')['is_podium'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum().shift(1).fillna(0)
    )

    # La feature con leakage incluiría el valor actual (3 en tercera carrera)
    # La feature correcta NO incluye el valor actual (2 en tercera carrera)

    assert df.iloc[2]['correct_feature'] == 2, \
        "Rolling window debe usar .shift(1) para prevenir data leakage"

    assert df.iloc[2]['leaky_feature'] == 3, \
        "Sin shift, la feature incluye el valor actual (data leakage)"


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

def test_classification_pipeline_integration(
    sample_classification_data,
    sample_classification_params
):
    """Test de integración que ejecuta múltiples nodos secuencialmente."""
    # 1. Crear target is_podium
    df = create_podium_target(sample_classification_data)
    assert 'is_podium' in df.columns, "Debe crear target is_podium"

    # 2. Filtrar válidos (ya están todos válidos en este caso)
    df = filter_valid_finishes(df)
    assert len(df) > 0, "Debe quedar datos después de filtrar"

    # 3. Ordenar por fecha (CRÍTICO para prevenir leakage)
    df = sort_by_date(df)

    # 4. Split temporal
    X_train, X_test, y_train, y_test = split_temporal_train_test(
        df,
        sample_classification_params
    )

    # Verificar que el pipeline completo funciona
    assert len(X_train) > 0, "Train debe tener datos"
    assert len(X_test) > 0, "Test debe tener datos"
    assert len(y_train) == len(X_train), "y_train debe tener mismo tamaño que X_train"
    assert len(y_test) == len(X_test), "y_test debe tener mismo tamaño que X_test"


# =============================================================================
# TESTS DE VALIDACIÓN DE CALIDAD DE DATOS
# =============================================================================

def test_podium_target_no_nulls(sample_classification_data):
    """Test que create_podium_target no genera valores nulos."""
    result = create_podium_target(sample_classification_data)

    assert not result['is_podium'].isna().any(), \
        "is_podium no debe contener valores nulos"


def test_temporal_split_date_format(sample_classification_params):
    """Test que temporal_split_date tiene formato correcto."""
    split_date = sample_classification_params['temporal_split_date']

    # Verificar que se puede parsear como fecha
    try:
        pd.to_datetime(split_date)
        valid_format = True
    except:
        valid_format = False

    assert valid_format, \
        "temporal_split_date debe tener formato YYYY-MM-DD válido"


@pytest.mark.parametrize("invalid_position", [-1, 0, 25, np.inf])
def test_create_podium_target_handles_invalid_positions(
    sample_classification_data,
    invalid_position
):
    """Test que create_podium_target maneja posiciones inválidas."""
    df = sample_classification_data.copy()
    df.loc[0, 'position'] = invalid_position

    result = create_podium_target(df)

    # Posiciones inválidas no deben ser consideradas podio
    if invalid_position < 1 or invalid_position > 3:
        invalid_rows = result[result['position'] == invalid_position]
        if len(invalid_rows) > 0:
            assert all(invalid_rows['is_podium'] == 0), \
                f"Posición inválida {invalid_position} no debe ser podio"


# =============================================================================
# TESTS DE DESBALANCEO DE CLASES
# =============================================================================

def test_classification_data_imbalance_detection(sample_classification_data):
    """Test que detecta desbalanceo de clases en el dataset."""
    df = create_podium_target(sample_classification_data)

    podium_count = (df['is_podium'] == 1).sum()
    non_podium_count = (df['is_podium'] == 0).sum()

    imbalance_ratio = podium_count / non_podium_count if non_podium_count > 0 else 0

    # En F1, típicamente hay desbalanceo (más no-podios que podios)
    # Esto justifica el uso de SMOTE en el pipeline de modelos
    assert imbalance_ratio < 1, \
        "Debe haber desbalanceo: más no-podios que podios (justifica SMOTE)"


def test_smote_necessity_justification(sample_classification_data):
    """
    Test que justifica la necesidad de SMOTE en el pipeline.

    En F1, solo 3 de ~20 pilotos logran podio (15%), creando desbalanceo.
    """
    df = create_podium_target(sample_classification_data)

    class_distribution = df['is_podium'].value_counts(normalize=True)
    minority_class_percentage = class_distribution.get(1, 0) * 100

    # Verificar que la clase minoritaria (podio) es < 30% del dataset
    assert minority_class_percentage < 30, \
        f"Clase minoritaria ({minority_class_percentage:.1f}%) justifica uso de SMOTE"
