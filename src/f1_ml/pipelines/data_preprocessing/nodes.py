"""Nodos del pipeline de preprocesamiento de datos para F1 Machine Learning.

Este módulo contiene las funciones para el preprocesamiento completo de los datos
de Formula 1, incluyendo limpieza, feature engineering, imputación y creación
de datasets listos para modelado.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, List
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

warnings.filterwarnings('ignore')


def limpiar_resultados_carrera(race_results: pd.DataFrame) -> pd.DataFrame:
    """Limpia y procesa los resultados de carrera.

    Args:
        race_results: DataFrame con resultados de carreras

    Returns:
        DataFrame con resultados limpiados
    """
    df_race = race_results.copy()

    def clean_position(pos):
        """Limpia y convierte posiciones a numérico"""
        if pd.isna(pos):
            return np.nan
        pos_str = str(pos).strip()
        if pos_str.isdigit():
            return int(pos_str)
        elif pos_str in ['R', 'D', 'E', 'W', 'F', 'N']:
            return np.nan  # DNF, DSQ, etc.
        else:
            try:
                return int(pos_str)
            except:
                return np.nan

    def clean_milliseconds(ms):
        """Limpia columna de milisegundos"""
        if pd.isna(ms) or ms == '\\N' or ms == '':
            return np.nan
        try:
            return int(ms)
        except:
            return np.nan

    # Aplicar limpieza
    df_race['position_clean'] = df_race['position'].apply(clean_position)
    df_race['milliseconds_clean'] = df_race['milliseconds'].apply(clean_milliseconds)

    # Crear variable objetivo de clasificación (podio)
    df_race['podium'] = (df_race['position_clean'] <= 3).astype(int)
    df_race.loc[df_race['position_clean'].isna(), 'podium'] = 0

    return df_race


def limpiar_resultados_clasificacion(qualifying_results: pd.DataFrame) -> pd.DataFrame:
    """Limpia y procesa los resultados de clasificación.

    Args:
        qualifying_results: DataFrame con resultados de clasificación

    Returns:
        DataFrame con resultados de clasificación limpiados
    """
    df_qual = qualifying_results.copy()

    def time_to_seconds(time_str):
        """Convierte formato MM:SS.sss a segundos totales"""
        if pd.isna(time_str) or time_str == '\\N' or time_str == '':
            return np.nan
        try:
            time_str = str(time_str).strip()
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
            return float(time_str)
        except:
            return np.nan

    # Convertir tiempos de clasificación
    for session in ['q1', 'q2', 'q3']:
        if session in df_qual.columns:
            df_qual[f'{session}_seconds'] = df_qual[session].apply(time_to_seconds)

    # Calcular tiempos promedio y mejor tiempo
    time_cols = [col for col in ['q1_seconds', 'q2_seconds', 'q3_seconds'] if col in df_qual.columns]
    if time_cols:
        df_qual['avg_qualifying_time'] = df_qual[time_cols].mean(axis=1, skipna=True)
        df_qual['best_qualifying_time'] = df_qual[time_cols].min(axis=1, skipna=True)

    return df_qual


def limpiar_tiempos_vuelta(lap_timings: pd.DataFrame) -> pd.DataFrame:
    """Limpia y filtra los tiempos de vuelta.

    Args:
        lap_timings: DataFrame con tiempos de vuelta

    Returns:
        DataFrame con tiempos de vuelta limpiados
    """
    df_laps = lap_timings.copy()

    # Convertir milisegundos a segundos
    df_laps['seconds'] = df_laps['milliseconds'] / 1000

    # Filtrar valores extremos
    df_laps_clean = df_laps[
        (df_laps['seconds'] >= 50) &
        (df_laps['seconds'] <= 300)
    ].copy()

    return df_laps_clean


def procesar_detalles_piloto(driver_details: pd.DataFrame) -> pd.DataFrame:
    """Procesa los detalles de los pilotos.

    Args:
        driver_details: DataFrame con detalles de pilotos

    Returns:
        DataFrame con detalles de pilotos procesados
    """
    df_drivers = driver_details.copy()

    # Convertir fecha de nacimiento
    df_drivers['dob'] = pd.to_datetime(df_drivers['dob'], errors='coerce')

    # Calcular edad actual
    current_date = datetime.now()
    df_drivers['age_current'] = (current_date - df_drivers['dob']).dt.days / 365.25

    # Limpiar nacionalidades
    nationality_mapping = {
        'British': 'United Kingdom',
        'American': 'USA',
        'German': 'Germany',
        'Brazilian': 'Brazil',
        'Italian': 'Italy',
        'French': 'France',
        'Spanish': 'Spain',
        'Dutch': 'Netherlands',
        'Finnish': 'Finland',
        'Austrian': 'Austria'
    }

    df_drivers['nationality_clean'] = df_drivers['nationality'].replace(nationality_mapping)

    return df_drivers


def imputar_valores_faltantes(qualifying_results: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores faltantes en los resultados de clasificación.

    Args:
        qualifying_results: DataFrame con resultados de clasificación

    Returns:
        DataFrame con valores imputados
    """
    df_qual = qualifying_results.copy()

    # Columnas numéricas para KNN
    numeric_cols = ['position', 'q1_seconds', 'q2_seconds', 'q3_seconds']
    available_numeric = [col for col in numeric_cols if col in df_qual.columns]

    if len(available_numeric) > 1:
        # Aplicar KNN Imputer
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

        subset_data = df_qual[available_numeric].copy()

        # Imputar solo columnas con faltantes razonables (< 50%)
        for col in available_numeric:
            missing_pct = df_qual[col].isnull().sum() / len(df_qual) * 100
            if 0 < missing_pct < 50 and 'seconds' in col:
                imputed_values = knn_imputer.fit_transform(subset_data)
                imputed_df = pd.DataFrame(imputed_values, columns=available_numeric, index=subset_data.index)
                df_qual[col] = imputed_df[col]

    return df_qual


def calcular_estadisticas_vuelta(lap_timings: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas de tiempo de vuelta por piloto/carrera.

    Args:
        lap_timings: DataFrame con tiempos de vuelta limpios

    Returns:
        DataFrame con estadísticas calculadas
    """
    # Calcular estadísticas por carrera-piloto
    lap_stats = lap_timings.groupby(['raceId', 'driverId']).agg({
        'seconds': ['mean', 'min', 'max', 'std', 'count'],
        'position': ['mean', 'min', 'max']
    }).round(3)

    # Aplanar columnas multinivel
    lap_stats.columns = ['_'.join(col).strip() for col in lap_stats.columns]
    lap_stats = lap_stats.reset_index()

    # Renombrar columnas
    rename_mapping = {
        'seconds_mean': 'avg_lap_time',
        'seconds_min': 'fastest_lap_time',
        'seconds_max': 'slowest_lap_time',
        'seconds_std': 'lap_time_consistency',
        'seconds_count': 'total_laps',
        'position_mean': 'avg_position_in_race',
        'position_min': 'best_position_in_race',
        'position_max': 'worst_position_in_race'
    }

    lap_stats = lap_stats.rename(columns=rename_mapping)

    return lap_stats


def calcular_rendimiento_historico(race_results: pd.DataFrame,
                                  race_schedule: pd.DataFrame) -> pd.DataFrame:
    """Calcula el rendimiento histórico reciente por piloto.

    Args:
        race_results: DataFrame con resultados de carreras limpios
        race_schedule: DataFrame con calendario de carreras

    Returns:
        DataFrame con rendimiento histórico calculado
    """
    # Combinar resultados con fechas
    df_race = race_results.merge(
        race_schedule[['raceId', 'year', 'round', 'date']],
        on='raceId', how='left'
    )

    df_race['date'] = pd.to_datetime(df_race['date'], errors='coerce')
    df_race = df_race.sort_values(['driverId', 'date'])

    def calculate_rolling_stats(group, window=3):
        """Calcula estadísticas móviles para un piloto"""
        group = group.sort_values('date')

        # Métricas móviles
        group[f'avg_position_last_{window}'] = group['position_clean'].rolling(
            window=window, min_periods=1
        ).mean().shift(1)

        group[f'avg_points_last_{window}'] = group['points'].rolling(
            window=window, min_periods=1
        ).mean().shift(1)

        group[f'podiums_last_{window}'] = group['podium'].rolling(
            window=window, min_periods=1
        ).sum().shift(1)

        group[f'dnfs_last_{window}'] = group['position_clean'].isna().rolling(
            window=window, min_periods=1
        ).sum().shift(1)

        return group

    # Aplicar cálculos por piloto
    df_race_enhanced = df_race.groupby('driverId').apply(
        lambda x: calculate_rolling_stats(x, window=3)
    ).reset_index(drop=True)

    # Calcular experiencia del piloto
    df_race_enhanced['driver_race_count'] = df_race_enhanced.groupby('driverId').cumcount()

    return df_race_enhanced


def calcular_estadisticas_constructor(constructor_rankings: pd.DataFrame,
                                    race_schedule: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas anuales de constructores.

    Args:
        constructor_rankings: DataFrame con rankings de constructores
        race_schedule: DataFrame con calendario de carreras

    Returns:
        DataFrame con estadísticas anuales de constructores
    """
    df_const_enhanced = constructor_rankings.merge(
        race_schedule[['raceId', 'year']],
        on='raceId', how='left'
    )

    # Calcular estadísticas por constructor y año
    constructor_yearly_stats = df_const_enhanced.groupby(['constructorId', 'year']).agg({
        'points': ['sum', 'mean'],
        'position': ['mean', 'min'],
        'wins': 'sum'
    }).round(3)

    constructor_yearly_stats.columns = ['_'.join(col).strip() for col in constructor_yearly_stats.columns]
    constructor_yearly_stats = constructor_yearly_stats.reset_index()

    rename_const = {
        'points_sum': 'constructor_total_points',
        'points_mean': 'constructor_avg_points',
        'position_mean': 'constructor_avg_position',
        'position_min': 'constructor_best_position',
        'wins_sum': 'constructor_total_wins'
    }

    constructor_yearly_stats = constructor_yearly_stats.rename(columns=rename_const)

    return constructor_yearly_stats


def calcular_rendimiento_circuito(race_results_enhanced: pd.DataFrame,
                                 race_schedule: pd.DataFrame) -> pd.DataFrame:
    """Calcula rendimiento histórico por circuito.

    Args:
        race_results_enhanced: DataFrame con resultados mejorados
        race_schedule: DataFrame con calendario de carreras

    Returns:
        DataFrame con rendimiento por circuito
    """
    df_race_circ = race_results_enhanced.merge(
        race_schedule[['raceId', 'circuitId']],
        on='raceId', how='left'
    )

    def calculate_circuit_stats(group):
        """Calcula estadísticas por circuito"""
        group = group.sort_values('date')

        # Carreras acumuladas en ese circuito
        group['races_at_circuit'] = group.groupby(['driverId', 'circuitId']).cumcount()

        # Promedio de posición en el circuito
        group['avg_position_at_circuit'] = (
            group.groupby(['driverId', 'circuitId'])['position_clean']
                 .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Podios acumulados
        group['podiums_at_circuit'] = (
            group.groupby(['driverId', 'circuitId'])['podium']
                 .transform(lambda x: x.expanding().sum().shift(1))
        )

        # Victorias acumuladas
        group['wins_at_circuit'] = (
            group.groupby(['driverId', 'circuitId'])['position_clean']
                 .transform(lambda x: (x == 1).expanding().sum().shift(1))
        )

        return group

    df_circuit_enhanced = calculate_circuit_stats(df_race_circ)

    # Rellenar NaN para primeras carreras
    circuit_vars = ['races_at_circuit', 'avg_position_at_circuit', 'podiums_at_circuit', 'wins_at_circuit']
    for var in circuit_vars:
        df_circuit_enhanced[var] = df_circuit_enhanced[var].fillna(0)

    return df_circuit_enhanced


def crear_dataset_unificado(race_results_circuit: pd.DataFrame,
                            lap_statistics: pd.DataFrame,
                            qualifying_results: pd.DataFrame,
                            driver_details: pd.DataFrame,
                            team_details: pd.DataFrame,
                            race_schedule: pd.DataFrame,
                            track_information: pd.DataFrame,
                            constructor_yearly_stats: pd.DataFrame) -> pd.DataFrame:
    """Crea el dataset maestro unificado.

    Args:
        race_results_circuit: Resultados con rendimiento por circuito
        lap_statistics: Estadísticas de vuelta
        qualifying_results: Resultados de clasificación
        driver_details: Detalles de pilotos
        team_details: Detalles de equipos
        race_schedule: Calendario de carreras
        track_information: Información de circuitos
        constructor_yearly_stats: Estadísticas anuales de constructores

    Returns:
        DataFrame maestro unificado
    """
    df_master = race_results_circuit.copy()

    # Unir estadísticas de vuelta
    df_master = df_master.merge(
        lap_statistics,
        on=['raceId', 'driverId'],
        how='left'
    )

    # Unir resultados de clasificación
    qualifying_cols = ['raceId', 'driverId', 'position', 'q1_seconds', 'q2_seconds', 'q3_seconds',
                      'avg_qualifying_time', 'best_qualifying_time']
    available_qual_cols = ['raceId', 'driverId'] + [col for col in qualifying_cols[2:]
                                                    if col in qualifying_results.columns]

    df_master = df_master.merge(
        qualifying_results[available_qual_cols].rename(columns={'position': 'qualifying_position'}),
        on=['raceId', 'driverId'],
        how='left'
    )

    # Unir detalles del piloto
    driver_cols = ['driverId', 'nationality_clean', 'age_current']
    available_driver_cols = [col for col in driver_cols if col in driver_details.columns]

    df_master = df_master.merge(
        driver_details[available_driver_cols],
        on='driverId',
        how='left'
    )

    # Unir detalles del constructor
    team_cols = ['constructorId', 'name', 'nationality']
    available_team_cols = [col for col in team_cols if col in team_details.columns]

    df_master = df_master.merge(
        team_details[available_team_cols].rename(columns={
            'name': 'constructor_name',
            'nationality': 'constructor_nationality'
        }),
        on='constructorId',
        how='left'
    )

    # Unir información del circuito
    if 'circuitId' not in df_master.columns:
        df_master = df_master.merge(
            race_schedule[['raceId', 'circuitId']],
            on='raceId',
            how='left'
        )

    circuit_cols = ['circuitId', 'name', 'location', 'country', 'lat', 'lng', 'alt']
    available_circuit_cols = [col for col in circuit_cols if col in track_information.columns]

    df_master = df_master.merge(
        track_information[available_circuit_cols].rename(columns={
            'name': 'circuit_name',
            'location': 'circuit_location',
            'country': 'circuit_country'
        }),
        on='circuitId',
        how='left'
    )

    # Unir estadísticas anuales del constructor
    df_master = df_master.merge(
        constructor_yearly_stats,
        on=['constructorId', 'year'],
        how='left'
    )

    return df_master


def limpiar_dataset_final(master_dataset: pd.DataFrame) -> pd.DataFrame:
    """Limpia y filtra el dataset maestro final.

    Args:
        master_dataset: Dataset maestro sin limpiar

    Returns:
        Dataset maestro limpio
    """
    df_master_clean = master_dataset.copy()

    # Filtrar datos esenciales
    essential_columns = ['raceId', 'driverId', 'constructorId', 'year']
    df_master_clean = df_master_clean.dropna(subset=essential_columns)

    # Filtrar por rango de años
    min_year = 1990
    df_master_clean = df_master_clean[df_master_clean['year'] >= min_year]

    # Eliminar duplicados finales
    df_master_clean = df_master_clean.drop_duplicates(subset=['raceId', 'driverId'])

    return df_master_clean


def codificar_variables_categoricas(master_dataset: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas en el dataset.

    Args:
        master_dataset: Dataset maestro limpio

    Returns:
        Dataset con variables categóricas codificadas
    """
    df_model_ready = master_dataset.copy()

    # Codificación de posiciones
    def encode_position(pos):
        if pd.isna(pos):
            return 999  # DNF
        pos_str = str(pos).strip()
        if pos_str.isdigit():
            return int(pos_str)
        elif pos_str in ['R', 'D', 'E', 'W', 'F', 'N']:
            return 999  # DNF codes
        else:
            try:
                return int(pos_str)
            except:
                return 999

    # Aplicar label encoding a posiciones
    position_cols = ['position', 'positionText', 'qualifying_position']
    for col in position_cols:
        if col in df_model_ready.columns:
            if 'position' in col.lower():
                df_model_ready[f'{col}_encoded'] = df_model_ready[col].apply(encode_position)

    # One-hot encoding para nacionalidades de constructor
    if 'constructor_nationality' in df_model_ready.columns:
        # Limitar a top categorías
        top_categories = df_model_ready['constructor_nationality'].value_counts().head(15).index.tolist()

        df_model_ready['constructor_nationality_clean'] = df_model_ready['constructor_nationality'].apply(
            lambda x: x if x in top_categories else 'Other'
        )

        # One-hot encoding
        dummies = pd.get_dummies(df_model_ready['constructor_nationality_clean'],
                               prefix='constructor_nationality', dummy_na=False)
        df_model_ready = pd.concat([df_model_ready, dummies], axis=1)

    return df_model_ready


def crear_datasets_modelado(model_ready_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Crea datasets finales para regresión y clasificación.

    Args:
        model_ready_data: Dataset con variables codificadas

    Returns:
        Tupla con datasets de regresión y clasificación
    """
    # Variables para regresión
    regression_features = [
        'raceId', 'driverId', 'constructorId', 'grid', 'qualifying_position',
        'q1_seconds', 'q2_seconds', 'q3_seconds', 'avg_qualifying_time', 'best_qualifying_time',
        'avg_position_last_3', 'avg_points_last_3', 'podiums_last_3', 'dnfs_last_3',
        'driver_race_count', 'constructor_avg_points', 'constructor_avg_position',
        'races_at_circuit', 'avg_position_at_circuit', 'lat', 'lng', 'alt'
    ]

    # Variables para clasificación
    classification_features = [
        'raceId', 'driverId', 'constructorId', 'grid', 'qualifying_position',
        'avg_position_last_3', 'avg_points_last_3', 'podiums_last_3',
        'driver_race_count', 'constructor_avg_points', 'constructor_best_position',
        'races_at_circuit', 'podiums_at_circuit', 'wins_at_circuit', 'age_current'
    ]

    # Targets
    regression_target = 'avg_lap_time'
    classification_target = 'podium'

    # Añadir variables one-hot
    onehot_cols = [col for col in model_ready_data.columns if 'constructor_nationality_' in col]
    important_onehot = onehot_cols[:20]  # Máximo 20

    # Dataset de regresión
    if regression_target in model_ready_data.columns:
        regression_data = model_ready_data[model_ready_data[regression_target].notna()].copy()
        regression_features_final = [col for col in regression_features if col in regression_data.columns]
        regression_features_final.extend(important_onehot)
        regression_features_final.append(regression_target)

        df_regression = regression_data[regression_features_final].copy()
        threshold = len(regression_features_final) * 0.7
        df_regression = df_regression.dropna(thresh=threshold)
    else:
        df_regression = pd.DataFrame()

    # Dataset de clasificación
    classification_features_final = [col for col in classification_features if col in model_ready_data.columns]
    classification_features_final.extend(important_onehot)
    classification_features_final.append(classification_target)

    df_classification = model_ready_data[classification_features_final].copy()
    threshold = len(classification_features_final) * 0.7
    df_classification = df_classification.dropna(thresh=threshold)

    return df_regression, df_classification