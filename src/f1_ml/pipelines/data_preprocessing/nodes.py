"""Nodos del pipeline de preprocesamiento de datos para F1 Machine Learning.

Este módulo contiene las funciones para el preprocesamiento completo de los datos
de Formula 1, incluyendo limpieza, feature engineering, imputación y creación
de datasets listos para modelado.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, List
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from scipy import stats
from scipy.stats import zscore

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


def tratar_outliers_avanzado(df: pd.DataFrame, columns: List[str],
                           method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """Detecta y trata outliers usando métodos avanzados.

    Args:
        df: DataFrame a procesar
        columns: Lista de columnas a procesar
        method: Método ('iqr', 'zscore', 'winsorize')
        factor: Factor multiplicativo para límites

    Returns:
        DataFrame con outliers tratados
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or df_clean[col].dtype not in ['int64', 'float64']:
            continue

        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            # Winsorización en lugar de eliminación
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(zscore(df_clean[col].dropna()))
            threshold = factor  # Usar factor como threshold z-score
            outlier_mask = z_scores > threshold

            # Reemplazar outliers con percentiles
            if outlier_mask.any():
                p_lower = df_clean[col].quantile(0.05)
                p_upper = df_clean[col].quantile(0.95)
                df_clean.loc[df_clean[col] < p_lower, col] = p_lower
                df_clean.loc[df_clean[col] > p_upper, col] = p_upper

        elif method == 'winsorize':
            # Winsorización por percentiles
            lower_percentile = (100 - 95) / 2  # 2.5%
            upper_percentile = 100 - lower_percentile  # 97.5%

            lower_bound = df_clean[col].quantile(lower_percentile / 100)
            upper_bound = df_clean[col].quantile(upper_percentile / 100)

            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    return df_clean


def limpiar_tiempos_vuelta(lap_timings: pd.DataFrame) -> pd.DataFrame:
    """Limpia y filtra los tiempos de vuelta con tratamiento avanzado de outliers.

    Args:
        lap_timings: DataFrame con tiempos de vuelta

    Returns:
        DataFrame con tiempos de vuelta limpiados
    """
    df_laps = lap_timings.copy()

    # Convertir milisegundos a segundos
    df_laps['seconds'] = df_laps['milliseconds'] / 1000

    # Filtro inicial básico para valores imposibles
    df_laps = df_laps[
        (df_laps['seconds'] >= 60) &  # Tiempo mínimo más realista
        (df_laps['seconds'] <= 200)   # Tiempo máximo más estricto
    ].copy()

    # Tratamiento avanzado de outliers por grupo (raceId)
    time_columns = ['seconds']
    df_laps_clean = df_laps.groupby('raceId').apply(
        lambda x: tratar_outliers_avanzado(x, time_columns, method='iqr', factor=2.0)
    ).reset_index(drop=True)

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


def imputar_valores_faltantes_avanzado(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores faltantes usando estrategias avanzadas específicas por variable.

    Args:
        df: DataFrame con valores faltantes

    Returns:
        DataFrame con valores imputados
    """
    df_imputed = df.copy()

    # 1. TRATAMIENTO ESPECÍFICO PARA Q3
    # Q3 tiene 68% faltantes porque solo top 10 clasifican
    if 'q3_seconds' in df_imputed.columns and 'position' in df_imputed.columns:
        # Crear indicador de si clasificó para Q3
        df_imputed['qualified_for_q3'] = (~df_imputed['q3_seconds'].isna()).astype(int)

        # Para los que no clasificaron, usar la peor Q2 + penalización
        if 'q2_seconds' in df_imputed.columns:
            worst_q2_by_race = df_imputed.groupby('raceId')['q2_seconds'].transform('max')
            penalty_factor = 1.02  # 2% de penalización

            mask_no_q3 = df_imputed['q3_seconds'].isna()
            df_imputed.loc[mask_no_q3, 'q3_seconds'] = worst_q2_by_race[mask_no_q3] * penalty_factor

    # 2. IMPUTACIÓN CONTEXTUAL POR GRUPO
    # Agrupar por raceId y constructor para imputación más precisa
    grouping_cols = []
    if 'raceId' in df_imputed.columns:
        grouping_cols.append('raceId')
    if 'constructorId' in df_imputed.columns:
        grouping_cols.append('constructorId')

    if grouping_cols:
        numeric_cols_to_impute = [
            'q1_seconds', 'q2_seconds', 'avg_qualifying_time', 'best_qualifying_time'
        ]
        available_cols = [col for col in numeric_cols_to_impute if col in df_imputed.columns]

        for col in available_cols:
            missing_pct = df_imputed[col].isnull().sum() / len(df_imputed) * 100

            if 0 < missing_pct < 30:  # Solo si faltan menos del 30%
                # Imputación por mediana del grupo
                df_imputed[col] = df_imputed.groupby(grouping_cols)[col].transform(
                    lambda x: x.fillna(x.median()) if not x.isna().all() else x
                )

                # Fallback a mediana global si aún faltan valores
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

    # 3. IMPUTACIÓN ITERATIVA PARA VARIABLES RELACIONADAS
    # Solo para variables con correlación alta
    correlated_vars = ['q1_seconds', 'q2_seconds', 'avg_qualifying_time', 'best_qualifying_time']
    available_corr_vars = [col for col in correlated_vars if col in df_imputed.columns]

    if len(available_corr_vars) >= 3:
        # Usar IterativeImputer solo si tenemos suficientes variables correlacionadas
        iterative_imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy='median'
        )

        # Aplicar solo a casos con pocos faltantes por fila
        subset_cols = available_corr_vars
        subset_data = df_imputed[subset_cols].copy()

        # Solo impute filas que no tengan más del 50% de valores faltantes
        missing_ratio = subset_data.isnull().sum(axis=1) / len(subset_cols)
        mask_imputable = missing_ratio <= 0.5

        if mask_imputable.sum() > 0:
            imputed_values = iterative_imputer.fit_transform(subset_data[mask_imputable])
            df_imputed.loc[mask_imputable, subset_cols] = imputed_values

    # 4. IMPUTACIÓN SIMPLE PARA EL RESTO
    simple_imputer = SimpleImputer(strategy='median')

    remaining_numeric = df_imputed.select_dtypes(include=[np.number]).columns
    remaining_with_na = [col for col in remaining_numeric if df_imputed[col].isnull().any()]

    for col in remaining_with_na:
        df_imputed[[col]] = simple_imputer.fit_transform(df_imputed[[col]])

    return df_imputed


def imputar_valores_faltantes(qualifying_results: pd.DataFrame) -> pd.DataFrame:
    """Wrapper para mantener compatibilidad con pipeline existente.

    Args:
        qualifying_results: DataFrame con resultados de clasificación

    Returns:
        DataFrame con valores imputados
    """
    return imputar_valores_faltantes_avanzado(qualifying_results)


def crear_features_avanzadas(df_master: pd.DataFrame) -> pd.DataFrame:
    """Crea features de ingeniería avanzadas para mejorar el modelado.

    Args:
        df_master: Dataset maestro

    Returns:
        DataFrame con features adicionales
    """
    df_enhanced = df_master.copy()

    # 1. RATIOS Y DIFERENCIAS ENTRE VARIABLES
    if 'qualifying_position' in df_enhanced.columns and 'grid' in df_enhanced.columns:
        # Diferencia entre posición de clasificación y parrilla (penalizaciones)
        df_enhanced['qualifying_grid_diff'] = df_enhanced['qualifying_position'] - df_enhanced['grid']
        df_enhanced['grid_penalty'] = (df_enhanced['qualifying_grid_diff'] > 0).astype(int)

    if 'best_qualifying_time' in df_enhanced.columns and 'avg_qualifying_time' in df_enhanced.columns:
        # Consistencia en clasificación (menor es mejor)
        df_enhanced['qualifying_consistency'] = (
            df_enhanced['avg_qualifying_time'] - df_enhanced['best_qualifying_time']
        )

    # 2. FEATURES DE RENDIMIENTO RELATIVO
    if 'constructor_avg_points' in df_enhanced.columns and 'avg_points_last_3' in df_enhanced.columns:
        # Rendimiento del piloto vs constructor
        df_enhanced['driver_vs_constructor_performance'] = (
            df_enhanced['avg_points_last_3'] / (df_enhanced['constructor_avg_points'] + 0.1)
        )

    if 'avg_position_at_circuit' in df_enhanced.columns and 'avg_position_last_3' in df_enhanced.columns:
        # Especialización en circuito
        df_enhanced['circuit_specialization'] = (
            df_enhanced['avg_position_last_3'] / (df_enhanced['avg_position_at_circuit'] + 0.1)
        )

    # 3. FEATURES DE EXPERIENCIA Y MOMENTUM
    if 'driver_race_count' in df_enhanced.columns:
        # Experiencia categórica
        df_enhanced['experience_level'] = pd.cut(
            df_enhanced['driver_race_count'],
            bins=[0, 50, 150, 300, np.inf],
            labels=['Rookie', 'Junior', 'Veteran', 'Legend']
        ).astype(str)

    if 'podiums_last_3' in df_enhanced.columns and 'dnfs_last_3' in df_enhanced.columns:
        # Momentum reciente (más podios, menos DNFs = mejor momentum)
        df_enhanced['recent_momentum'] = (
            df_enhanced['podiums_last_3'] - df_enhanced['dnfs_last_3']
        )

    # 4. FEATURES GEOGRÁFICAS Y TÉCNICAS
    if 'lat' in df_enhanced.columns and 'lng' in df_enhanced.columns:
        # Distancia desde Greenwich como proxy de zona horaria/clima
        df_enhanced['distance_from_greenwich'] = np.sqrt(
            df_enhanced['lat']**2 + df_enhanced['lng']**2
        )

    if 'alt' in df_enhanced.columns:
        # Circuito de alta altitud (>1000m)
        df_enhanced['high_altitude_circuit'] = (df_enhanced['alt'] > 1000).astype(int)

    # 5. FEATURES DE TEMPORADA
    if 'year' in df_enhanced.columns and 'round' in df_enhanced.columns:
        # Progreso en temporada (normalizado)
        season_progress = df_enhanced.groupby('year')['round'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )
        df_enhanced['season_progress'] = season_progress

        # Era de F1 (cambios reglamentarios importantes)
        df_enhanced['f1_era'] = pd.cut(
            df_enhanced['year'],
            bins=[0, 1993, 2005, 2013, 2021, np.inf],
            labels=['Classic', 'Modern', 'Hybrid_Early', 'Hybrid_Late', 'Current']
        ).astype(str)

    return df_enhanced


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


def detectar_multicolinealidad(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Detecta y maneja multicolinealidad automáticamente.

    Args:
        df: DataFrame con features numéricas
        threshold: Umbral de correlación para considerar multicolinealidad

    Returns:
        DataFrame sin multicolinealidad excesiva
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # Calcular matriz de correlación
    corr_matrix = df_clean[numeric_cols].corr().abs()

    # Encontrar pares altamente correlacionados
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

    # Eliminar variables redundantes priorizando las más interpretables
    cols_to_remove = set()
    priority_keep = [
        'avg_lap_time', 'podium', 'qualifying_position', 'grid',
        'avg_position_last_3', 'driver_race_count', 'constructor_avg_points'
    ]

    for col1, col2, corr_value in high_corr_pairs:
        if col1 not in cols_to_remove and col2 not in cols_to_remove:
            # Mantener la variable más interpretable o importante
            if col1 in priority_keep and col2 not in priority_keep:
                cols_to_remove.add(col2)
            elif col2 in priority_keep and col1 not in priority_keep:
                cols_to_remove.add(col1)
            elif 'avg' in col1 and 'best' in col2:
                cols_to_remove.add(col2)  # Mantener avg sobre best
            elif 'seconds' in col1 and 'time' in col2:
                cols_to_remove.add(col1)  # Mantener time sobre seconds
            else:
                # Si no hay preferencia clara, mantener el primero
                cols_to_remove.add(col2)

    # Remover columnas identificadas
    df_clean = df_clean.drop(columns=list(cols_to_remove))

    print(f"Removed {len(cols_to_remove)} highly correlated features: {list(cols_to_remove)}")

    return df_clean


def aplicar_validacion_temporal(df: pd.DataFrame, test_years: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica validación temporal para evitar data leakage.

    Args:
        df: DataFrame con datos
        test_years: Número de años más recientes para test

    Returns:
        Tuple con datasets de train y test temporalmente separados
    """
    if 'year' not in df.columns:
        print("Warning: No 'year' column found. Using random split.")
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=0.2, random_state=42)

    # Obtener años únicos y ordenar
    years = sorted(df['year'].unique())
    if len(years) < test_years + 2:
        print(f"Warning: Not enough years for temporal validation. Using last {len(years)//4} years for test.")
        test_years = max(1, len(years) // 4)

    # Separar por años
    test_start_year = years[-test_years]
    train_data = df[df['year'] < test_start_year].copy()
    test_data = df[df['year'] >= test_start_year].copy()

    print(f"Temporal split: Train years {years[0]}-{test_start_year-1}, Test years {test_start_year}-{years[-1]}")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    return train_data, test_data


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

        # Añadir features avanzadas
        regression_data = crear_features_avanzadas(regression_data)

        # Seleccionar features finales incluyendo las nuevas
        new_features = [
            'qualifying_grid_diff', 'grid_penalty', 'qualifying_consistency',
            'driver_vs_constructor_performance', 'circuit_specialization',
            'recent_momentum', 'distance_from_greenwich', 'high_altitude_circuit',
            'season_progress', 'qualified_for_q3'
        ]

        regression_features_final = [col for col in regression_features if col in regression_data.columns]
        regression_features_final.extend([col for col in new_features if col in regression_data.columns])
        regression_features_final.extend(important_onehot)
        regression_features_final.append(regression_target)

        df_regression = regression_data[regression_features_final].copy()

        # Aplicar detección de multicolinealidad
        df_regression = detectar_multicolinealidad(df_regression, threshold=0.85)

        # Limpieza final
        threshold = len(df_regression.columns) * 0.6  # Más permisivo después de feature engineering
        df_regression = df_regression.dropna(thresh=threshold)

    else:
        df_regression = pd.DataFrame()

    # Dataset de clasificación
    classification_data = model_ready_data.copy()
    classification_data = crear_features_avanzadas(classification_data)

    new_features_class = [
        'qualifying_grid_diff', 'grid_penalty', 'driver_vs_constructor_performance',
        'circuit_specialization', 'recent_momentum', 'high_altitude_circuit',
        'season_progress', 'qualified_for_q3'
    ]

    classification_features_final = [col for col in classification_features if col in classification_data.columns]
    classification_features_final.extend([col for col in new_features_class if col in classification_data.columns])
    classification_features_final.extend(important_onehot)
    classification_features_final.append(classification_target)

    df_classification = classification_data[classification_features_final].copy()
    df_classification = detectar_multicolinealidad(df_classification, threshold=0.85)

    threshold = len(df_classification.columns) * 0.6
    df_classification = df_classification.dropna(thresh=threshold)

    return df_regression, df_classification