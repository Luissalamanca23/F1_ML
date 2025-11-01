"""
Nodos para el pipeline de preparación de datos de clasificación (predicción de podio)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from category_encoders import TargetEncoder


def load_raw_data_classification(
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame,
    races: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    circuits: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Cargar datos crudos de las 6 tablas"""
    return {
        'race_results': race_results,
        'qualifying': qualifying_results,
        'races': races,
        'drivers': drivers,
        'constructors': constructors,
        'circuits': circuits
    }


def merge_tables_classification(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge secuencial de las 6 tablas"""
    race_results = data_dict['race_results'].copy()
    qualifying = data_dict['qualifying'].copy()
    races = data_dict['races'].copy()
    circuits = data_dict['circuits'].copy()
    drivers = data_dict['drivers'].copy()
    constructors = data_dict['constructors'].copy()

    # Merge secuencial
    df = race_results.merge(races, on='raceId', how='left')
    df = df.merge(qualifying, on=['raceId', 'driverId'], how='left', suffixes=('', '_quali'))
    df = df.merge(circuits, on='circuitId', how='left', suffixes=('', '_circuit'))
    df = df.merge(drivers, on='driverId', how='left', suffixes=('', '_driver'))
    df = df.merge(constructors, on='constructorId', how='left', suffixes=('', '_constructor'))

    return df


def filter_valid_finishes(df: pd.DataFrame) -> pd.DataFrame:
    """Filtrar solo pilotos que terminaron la carrera (position notna)"""
    df_filtered = df[df['position'].notna()].copy()
    return df_filtered


def create_podium_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crear variable target is_podium (1 si posición <= 3, 0 si > 3)"""
    df = df.copy()
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df = df[df['position'].notna()].copy()
    df['position_original'] = df['position']
    df['is_podium'] = (df['position'] <= 3).astype(int)
    return df


def drop_leakage_features_classification(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Eliminar variables con data leakage"""
    df = df.copy()
    leakage_features = params['leakage_features']

    # Eliminar solo las que existen
    cols_to_drop = [col for col in leakage_features if col in df.columns]

    # Eliminar variantes creadas por merges (ej. time_x, time_y)
    for feature in leakage_features:
        for suffix in ['_x', '_y']:
            candidate = f"{feature}{suffix}"
            if candidate in df.columns:
                cols_to_drop.append(candidate)

    # Quitar columnas de texto que no se usarán (URLs, nombres, etc.), dejando solo las requeridas para encoding
    allowed_object_columns = {
        'location',
        'country',
        'nationality',
        'nationality_constructor',
        'date',
        'dob',
        'q1',
        'q2',
        'q3',
    }
    object_columns = df.select_dtypes(include='object').columns
    cols_to_drop.extend(col for col in object_columns if col not in allowed_object_columns)

    # Evitar duplicados preservando el orden
    cols_to_drop = list(dict.fromkeys(cols_to_drop))

    df = df.drop(columns=cols_to_drop)

    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ordenar por fecha (CRÍTICO para prevenir leakage temporal)"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'raceId']).reset_index(drop=True)
    return df


def create_driver_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear features históricas del piloto (rolling/expanding con shift)"""
    df = df.copy()

    # 1. Podios últimas 5 carreras (con shift para evitar leakage)
    df['driver_podiums_last_5'] = df.groupby('driverId')['is_podium'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
    ).fillna(0)

    # 2. Tasa de podio en temporada actual (expanding mean con shift)
    df['driver_podium_rate_season'] = df.groupby(['driverId', 'year'])['is_podium'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)

    # 3. Victorias en la temporada (position_original == 1)
    df['driver_wins_season'] = df.groupby(['driverId', 'year'])['position_original'].transform(
        lambda x: (x == 1).astype(float).cumsum().shift(1)
    ).fillna(0)

    return df


def create_constructor_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear features históricas del constructor"""
    df = df.copy()

    # 1. Podios escudería últimas 5 carreras
    df['constructor_podiums_last_5'] = df.groupby('constructorId')['is_podium'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
    ).fillna(0)

    # 2. Tasa de podio constructor en temporada
    df['constructor_podium_rate_season'] = df.groupby(['constructorId', 'year'])['is_podium'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)

    return df


def create_qualifying_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear features de qualifying y temporales"""
    df = df.copy()

    # Features binarias de qualifying
    df['quali_position_top3'] = (df['position_quali'] <= 3).astype(int).fillna(0)
    df['quali_position_top5'] = (df['position_quali'] <= 5).astype(int).fillna(0)

    # Edad del piloto
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['driver_age'] = df['year'] - df['dob'].dt.year

    # Progreso de temporada
    max_round_per_season = df.groupby('year')['round'].transform('max')
    df['season_progress'] = df['round'] / max_round_per_season
    df['is_season_start'] = (df['round'] <= 3).astype(int)
    df['is_season_end'] = (df['round'] >= max_round_per_season - 2).astype(int)

    # Convertir tiempos de qualifying a milisegundos
    def time_to_milliseconds(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return (minutes * 60 + seconds) * 1000
            else:
                return float(time_str) * 1000
        except:
            return np.nan

    df['q1_ms'] = df['q1'].apply(time_to_milliseconds) if 'q1' in df.columns else np.nan
    df['q2_ms'] = df['q2'].apply(time_to_milliseconds) if 'q2' in df.columns else np.nan
    df['q3_ms'] = df['q3'].apply(time_to_milliseconds) if 'q3' in df.columns else np.nan

    # Flags de participación en qualifying
    df['has_q1'] = df['q1'].notna().astype(int) if 'q1' in df.columns else 0
    df['has_q2'] = df['q2'].notna().astype(int) if 'q2' in df.columns else 0
    df['has_q3'] = df['q3'].notna().astype(int) if 'q3' in df.columns else 0

    return df


def impute_missing_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Imputar valores faltantes"""
    df = df.copy()

    # position_quali: max + 1 por carrera (partió último)
    if 'position_quali' in df.columns:
        max_quali_per_race = df.groupby('raceId')['position_quali'].transform('max')
        df['position_quali'] = df['position_quali'].fillna(max_quali_per_race + 1)
        # Fallback: si toda la carrera no tiene datos, usar valor máximo global + 1
        if df['position_quali'].isna().any():
            fallback_value = df['position_quali'].max() + 1
            df['position_quali'] = df['position_quali'].fillna(fallback_value)

    # IDs de qualifying sin información -> marcador -1
    for col in ['qualifyId', 'constructorId_quali', 'number_quali']:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # dob: mediana
    if 'dob' in df.columns:
        median_dob = df['dob'].median()
        df['dob'] = df['dob'].fillna(median_dob)
        df['driver_age'] = df['driver_age'].fillna(df['driver_age'].median())

    # Tiempos qualifying: llenar con mediana
    for col in ['q1_ms', 'q2_ms', 'q3_ms']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Coordenadas: llenar con mediana
    for col in ['lat', 'lng', 'alt']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def split_temporal_train_test(
    df: pd.DataFrame,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split temporal train/test (train: < 2023, test: >= 2023)"""
    split_date = params['temporal_split_date']

    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    # Separar features y target
    # Eliminar columnas antes de separar X/y
    columns_to_drop = [
        'raceId', 'date', 'name', 'dob', 'q1', 'q2', 'q3',
        'resultId', 'number', 'positionText',
        'position_original', 'is_podium'
    ]

    # Eliminar solo las que existen
    cols_drop_train = [col for col in columns_to_drop if col in train_df.columns]
    cols_drop_test = [col for col in columns_to_drop if col in test_df.columns]

    y_train = train_df['is_podium'].copy()
    y_test = test_df['is_podium'].copy()

    X_train = train_df.drop(columns=cols_drop_train)
    X_test = test_df.drop(columns=cols_drop_test)

    return X_train, X_test, y_train, y_test


def apply_dual_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Aplicar MinMaxScaler y StandardScaler"""
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # MinMaxScaler para features acotadas
    minmax_features = [f for f in params['minmax_features'] if f in X_train.columns]
    if minmax_features:
        minmax_scaler = MinMaxScaler()
        X_train_scaled[minmax_features] = minmax_scaler.fit_transform(X_train[minmax_features])
        X_test_scaled[minmax_features] = minmax_scaler.transform(X_test[minmax_features])
    else:
        minmax_scaler = None

    # StandardScaler para features gaussianas
    standard_features = [f for f in params['standard_features'] if f in X_train.columns]
    if standard_features:
        standard_scaler = StandardScaler()
        X_train_scaled[standard_features] = standard_scaler.fit_transform(X_train[standard_features])
        X_test_scaled[standard_features] = standard_scaler.transform(X_test[standard_features])
    else:
        standard_scaler = None

    scalers = {
        'minmax_scaler': minmax_scaler,
        'standard_scaler': standard_scaler
    }

    return X_train_scaled, X_test_scaled, scalers


def apply_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Aplicar Target Encoding y One-Hot Encoding"""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    encoders = {}

    # Target Encoding para IDs de alta cardinalidad
    target_encoding_features = ['driverId', 'constructorId', 'circuitId']
    target_features = [f for f in target_encoding_features if f in X_train.columns]

    if target_features:
        target_encoder = TargetEncoder(cols=target_features, smoothing=1.0)
        X_train_encoded[target_features] = target_encoder.fit_transform(
            X_train[target_features], y_train
        )
        X_test_encoded[target_features] = target_encoder.transform(X_test[target_features])
        encoders['target_encoder'] = target_encoder

    # One-Hot Encoding para categóricas de baja cardinalidad
    onehot_features = ['location', 'country', 'nationality', 'nationality_constructor']
    onehot_features = [f for f in onehot_features if f in X_train.columns]

    if onehot_features:
        X_train_encoded = pd.get_dummies(
            X_train_encoded,
            columns=onehot_features,
            drop_first=True
        )
        X_test_encoded = pd.get_dummies(
            X_test_encoded,
            columns=onehot_features,
            drop_first=True
        )

        # Alinear columnas entre train y test
        missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_cols:
            X_test_encoded[col] = 0

        extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
        for col in extra_cols:
            X_test_encoded = X_test_encoded.drop(columns=[col])

        X_test_encoded = X_test_encoded[X_train_encoded.columns]

    # Eliminar cualquier columna no numérica remanente antes de SMOTE/modelado
    non_numeric_cols = X_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X_train_encoded = X_train_encoded.drop(columns=non_numeric_cols)
        existing_in_test = [col for col in non_numeric_cols if col in X_test_encoded.columns]
        X_test_encoded = X_test_encoded.drop(columns=existing_in_test)

    return X_train_encoded, X_test_encoded, encoders
