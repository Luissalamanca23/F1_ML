"""
Nodes del Pipeline de Preparación de Datos para Regresión F1
=============================================================

Este módulo contiene las 11 funciones (nodes) que transforman los datos crudos
desde 01_raw hasta el dataset final listo para modelado.

Autor: Pipeline F1 ML
Fecha: 2025-10-25
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# NODE 1: CARGA DE DATOS CRUDOS
# =============================================================================

def load_raw_data(
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame,
    driver_rankings: pd.DataFrame,
    constructor_rankings: pd.DataFrame,
    track_information: pd.DataFrame,
    race_schedule: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Carga las 6 tablas crudas desde 01_raw.

    Args:
        race_results: Resultados de carreras
        qualifying_results: Resultados de clasificación
        driver_rankings: Rankings de pilotos
        constructor_rankings: Rankings de constructores
        track_information: Información de circuitos
        race_schedule: Calendario de carreras

    Returns:
        Dict con las 6 tablas cargadas
    """
    logger.info("[OK] Cargando 6 tablas desde 01_raw")
    logger.info(f"  - Race_Results: {race_results.shape}")
    logger.info(f"  - Qualifying_Results: {qualifying_results.shape}")
    logger.info(f"  - Driver_Rankings: {driver_rankings.shape}")
    logger.info(f"  - Constructor_Rankings: {constructor_rankings.shape}")
    logger.info(f"  - Track_Information: {track_information.shape}")
    logger.info(f"  - Race_Schedule: {race_schedule.shape}")

    return {
        "race_results": race_results,
        "qualifying_results": qualifying_results,
        "driver_rankings": driver_rankings,
        "constructor_rankings": constructor_rankings,
        "track_information": track_information,
        "race_schedule": race_schedule,
    }


# =============================================================================
# NODE 2: MERGE DE TABLAS
# =============================================================================

def merge_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge de 6 tablas relacionales.

    Secuencia de merges:
    1. Race_Results ← Race_Schedule (on raceId)
    2. Race_Results ← Qualifying_Results (on raceId, driverId)
    3. Race_Results ← Driver_Rankings (on raceId, driverId)
    4. Race_Results ← Constructor_Rankings (on raceId, constructorId)
    5. Race_Results ← Track_Information (on circuitId)

    Args:
        tables: Diccionario con las 6 tablas

    Returns:
        DataFrame merged
    """
    logger.info("[MERGE] Realizando merge de 6 tablas...")

    # Empezar con Race_Results
    df = tables["race_results"].copy()
    logger.info(f"  Tabla base (Race_Results): {df.shape}")

    # Merge 1: Race_Schedule (para obtener year, round, circuitId)
    df = df.merge(
        tables["race_schedule"][["raceId", "year", "round", "circuitId"]],
        on="raceId",
        how="left",
    )
    logger.info(f"  Después de merge con Race_Schedule: {df.shape}")

    # Merge 2: Qualifying_Results (posición en clasificación)
    df = df.merge(
        tables["qualifying_results"][["raceId", "driverId", "position"]].rename(
            columns={"position": "position_quali"}
        ),
        on=["raceId", "driverId"],
        how="left",
    )
    logger.info(f"  Después de merge con Qualifying_Results: {df.shape}")

    # Merge 3: Driver_Rankings (puntos, posición, victorias del piloto)
    df = df.merge(
        tables["driver_rankings"][
            ["raceId", "driverId", "points", "position", "wins"]
        ].rename(
            columns={
                "points": "puntos_campeonato_piloto",
                "position": "posicion_campeonato_piloto",
                "wins": "victorias_piloto",
            }
        ),
        on=["raceId", "driverId"],
        how="left",
    )
    logger.info(f"  Después de merge con Driver_Rankings: {df.shape}")

    # Merge 4: Constructor_Rankings (puntos, posición del constructor)
    df = df.merge(
        tables["constructor_rankings"][
            ["raceId", "constructorId", "points", "position", "wins"]
        ].rename(
            columns={
                "points": "puntos_campeonato_constructor",
                "position": "posicion_campeonato_constructor",
                "wins": "victorias_constructor",
            }
        ),
        on=["raceId", "constructorId"],
        how="left",
    )
    logger.info(f"  Después de merge con Constructor_Rankings: {df.shape}")

    # Merge 5: Track_Information (características del circuito)
    df = df.merge(
        tables["track_information"][
            [
                "circuitId",
                "name",
                "location",
                "country",
                "lat",
                "lng",
                "alt",
            ]
        ].rename(
            columns={
                "name": "nombre_circuito",
                "location": "ubicacion_circuito",
                "country": "pais_circuito",
                "lat": "latitud",
                "lng": "longitud",
                "alt": "altitud",
            }
        ),
        on="circuitId",
        how="left",
    )
    logger.info(f"  [OK] Merge completo: {df.shape}")

    return df


# =============================================================================
# NODE 3: FILTRO TEMPORAL (ERA MODERNA)
# =============================================================================

def filter_modern_era(
    df: pd.DataFrame, params: Dict[str, int]
) -> pd.DataFrame:
    """
    Filtra solo carreras de la era moderna (2010-2024).

    Args:
        df: DataFrame merged
        params: Dict con 'modern_era_start' y 'modern_era_end'

    Returns:
        DataFrame filtrado
    """
    year_start = params["modern_era_start"]
    year_end = params["modern_era_end"]

    logger.info(f"[FILTER] Filtrando era moderna ({year_start}-{year_end})...")
    logger.info(f"  Filas antes del filtro: {len(df)}")

    df_filtered = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()

    logger.info(f"  Filas después del filtro: {len(df_filtered)}")
    logger.info(f"  Carreras eliminadas: {len(df) - len(df_filtered)}")

    return df_filtered


# =============================================================================
# NODE 4: ELIMINAR DNFs
# =============================================================================

def remove_dnfs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina pilotos que no terminaron la carrera (DNF = Did Not Finish).

    Criterio: posición válida (numérica, entre 1-30)

    Args:
        df: DataFrame filtrado por era

    Returns:
        DataFrame sin DNFs
    """
    logger.info("[DNF] Eliminando DNFs (Did Not Finish)...")
    logger.info(f"  Filas antes: {len(df)}")

    # Convertir position a numérico
    df_clean = df.copy()
    df_clean["position"] = pd.to_numeric(df_clean["position"], errors="coerce")

    # Filtrar solo posiciones válidas (1-30)
    df_clean = df_clean[
        (df_clean["position"].notna())
        & (df_clean["position"] >= 1)
        & (df_clean["position"] <= 30)
    ].copy()

    logger.info(f"  Filas después: {len(df_clean)}")
    logger.info(f"  DNFs eliminados: {len(df) - len(df_clean)}")

    return df_clean


# =============================================================================
# NODE 5: ELIMINAR VARIABLES CON DATA LEAKAGE
# =============================================================================

def drop_leakage_features(df: pd.DataFrame, params: Dict[str, list]) -> pd.DataFrame:
    """
    Elimina las 16 variables que contienen data leakage.

    Args:
        df: DataFrame sin DNFs
        params: Dict con lista de 'leakage_features'

    Returns:
        DataFrame sin variables prohibidas
    """
    leakage_features = params["leakage_features"]

    logger.info(f"[LEAKAGE] Eliminando {len(leakage_features)} variables con data leakage...")

    # Filtrar solo las columnas que existen en el DataFrame
    cols_to_drop = [col for col in leakage_features if col in df.columns]

    logger.info(f"  Variables encontradas y eliminadas: {len(cols_to_drop)}")
    logger.info(f"  Columnas antes: {df.shape[1]}")

    df_clean = df.drop(columns=cols_to_drop)

    logger.info(f"  Columnas después: {df_clean.shape[1]}")

    return df_clean


# =============================================================================
# NODE 6: IMPUTAR VALORES FALTANTES
# =============================================================================

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en position_quali con la mediana del grid.

    Justificación: Si no hay qualifying position, usar grid position como proxy.

    Args:
        df: DataFrame con features permitidas

    Returns:
        DataFrame sin valores faltantes críticos
    """
    logger.info("[IMPUTE] Imputando valores faltantes...")

    df_imputed = df.copy()

    # Contar NaNs en position_quali
    nan_count = df_imputed["position_quali"].isna().sum()

    if nan_count > 0:
        logger.info(f"  NaNs en position_quali: {nan_count}")

        # Imputar con mediana del grid (o mediana global si grid también es NaN)
        for idx in df_imputed[df_imputed["position_quali"].isna()].index:
            if pd.notna(df_imputed.loc[idx, "grid"]):
                df_imputed.loc[idx, "position_quali"] = df_imputed.loc[idx, "grid"]
            else:
                # Fallback: mediana global de position_quali
                df_imputed.loc[idx, "position_quali"] = df_imputed[
                    "position_quali"
                ].median()

        logger.info(f"  [OK] NaNs imputados: {nan_count}")
    else:
        logger.info("  No hay NaNs en position_quali")

    return df_imputed


# =============================================================================
# NODE 7: SPLIT TRAIN/TEST
# =============================================================================

def split_train_test(
    df: pd.DataFrame, params: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en train (80%) y test (20%).

    Args:
        df: DataFrame limpio e imputado
        params: Dict con 'test_size' y 'random_state'

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    logger.info("[SPLIT] Dividiendo train/test...")

    # Separar features y target
    X = df.drop("position", axis=1)
    y = df["position"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        shuffle=True,
    )

    logger.info(f"  Train: {X_train.shape[0]} muestras")
    logger.info(f"  Test: {X_test.shape[0]} muestras")
    logger.info(f"  Target media (train): {y_train.mean():.2f}")
    logger.info(f"  Target media (test): {y_test.mean():.2f}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# NODE 8: FEATURE ENGINEERING (9 FEATURES AVANZADAS)
# =============================================================================

def create_advanced_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea 9 features avanzadas mediante ingeniería de features.

    Features creadas:
    1. diferencia_grid_quali - Indica penalizaciones
    2. ratio_puntos_piloto_constructor - Importancia relativa
    3. quali_x_puntos_constructor - Interacción
    4. progreso_temporada - Normalizado (0-1)
    5. diff_posicion_piloto_constructor - Gap posiciones
    6. log_puntos_piloto - Log de puntos (reduce skewness)
    7. log_puntos_constructor - Log de puntos constructor
    8. quali_x_grid - Consistencia
    9. altitud_extrema - Binaria (>1000m)

    Args:
        X_train: Features de entrenamiento
        X_test: Features de test

    Returns:
        Tupla (X_train_fe, X_test_fe) con nuevas features
    """
    logger.info("[FE] Aplicando Feature Engineering (9 features avanzadas)...")

    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()

    for df_fe in [X_train_fe, X_test_fe]:
        # 1. Diferencia grid vs quali
        df_fe["diferencia_grid_quali"] = df_fe["grid"] - df_fe["position_quali"]

        # 2. Ratio puntos piloto/constructor
        df_fe["ratio_puntos_piloto_constructor"] = np.where(
            df_fe["puntos_campeonato_constructor"] > 0,
            df_fe["puntos_campeonato_piloto"]
            / df_fe["puntos_campeonato_constructor"],
            0,
        )

        # 3. Interacción quali × puntos constructor
        df_fe["quali_x_puntos_constructor"] = (
            df_fe["position_quali"] * df_fe["puntos_campeonato_constructor"]
        )

        # 4. Progreso temporada (normalizado)
        df_fe["progreso_temporada"] = df_fe["round"] / 24.0

        # 5. Diferencia posición piloto-constructor
        df_fe["diff_posicion_piloto_constructor"] = (
            df_fe["posicion_campeonato_piloto"]
            - df_fe["posicion_campeonato_constructor"]
        )

        # 6 y 7. Log de puntos
        df_fe["log_puntos_piloto"] = np.log1p(df_fe["puntos_campeonato_piloto"])
        df_fe["log_puntos_constructor"] = np.log1p(
            df_fe["puntos_campeonato_constructor"]
        )

        # 8. Interacción quali × grid
        df_fe["quali_x_grid"] = df_fe["position_quali"] * df_fe["grid"]

        # 9. Altitud extrema (binaria)
        df_fe["altitud_extrema"] = (df_fe["altitud"] > 1000).astype(int)

    logger.info(f"  [OK] Features añadidas: 9")
    logger.info(f"  Dimensiones train: {X_train_fe.shape}")
    logger.info(f"  Dimensiones test: {X_test_fe.shape}")

    return X_train_fe, X_test_fe


# =============================================================================
# NODE 9: TARGET ENCODING PARA IDs
# =============================================================================

def apply_target_encoding(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """
    Aplica Target Encoding a las columnas de IDs (alta cardinalidad).

    Columnas: driverId, constructorId, circuitId

    Args:
        X_train: Features de entrenamiento
        X_test: Features de test
        y_train: Target de entrenamiento

    Returns:
        Tupla (X_train_encoded, X_test_encoded, encoder)
    """
    logger.info("[ENCODE] Aplicando Target Encoding a IDs...")

    cols_ids = ["driverId", "constructorId", "circuitId"]

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    # Crear y fitear encoder (SOLO en train)
    encoder = TargetEncoder(cols=cols_ids)
    encoder.fit(X_train_enc[cols_ids], y_train)

    # Transformar train y test
    X_train_enc[cols_ids] = encoder.transform(X_train_enc[cols_ids])
    X_test_enc[cols_ids] = encoder.transform(X_test_enc[cols_ids])

    logger.info(f"  [OK] Columnas encoded: {cols_ids}")

    return X_train_enc, X_test_enc, encoder


# =============================================================================
# NODE 10: ONE-HOT ENCODING PARA TEXTO
# =============================================================================

def apply_onehot_encoding(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica One-Hot Encoding a las columnas de texto categóricas.

    Columnas: nombre_circuito, ubicacion_circuito, pais_circuito

    Args:
        X_train: Features de entrenamiento
        X_test: Features de test

    Returns:
        Tupla (X_train_onehot, X_test_onehot)
    """
    logger.info("[ONEHOT] Aplicando One-Hot Encoding a texto...")

    cols_texto = ["nombre_circuito", "ubicacion_circuito", "pais_circuito"]

    # Aplicar get_dummies
    X_train_onehot = pd.get_dummies(X_train, columns=cols_texto, drop_first=True)
    X_test_onehot = pd.get_dummies(X_test, columns=cols_texto, drop_first=True)

    # Asegurar mismas columnas en train y test
    for col in X_train_onehot.columns:
        if col not in X_test_onehot.columns:
            X_test_onehot[col] = 0

    for col in X_test_onehot.columns:
        if col not in X_train_onehot.columns:
            X_train_onehot[col] = 0

    # Reordenar columnas
    X_test_onehot = X_test_onehot[X_train_onehot.columns]

    logger.info(f"  [OK] Dimensiones train: {X_train_onehot.shape}")
    logger.info(f"  [OK] Dimensiones test: {X_test_onehot.shape}")

    return X_train_onehot, X_test_onehot


# =============================================================================
# NODE 11: SCALING (STANDARDSCALER)
# =============================================================================

def apply_scaling(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Aplica StandardScaler a todas las features.

    CRÍTICO: El scaler se fitea SOLO en train para evitar data leakage.

    Args:
        X_train: Features de entrenamiento
        X_test: Features de test

    Returns:
        Tupla (X_train_scaled, X_test_scaled, scaler)
    """
    logger.info("[SCALE] Aplicando StandardScaler...")

    # Crear scaler y fitear SOLO en train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"  [OK] Train escalado: {X_train_scaled.shape}")
    logger.info(f"  [OK] Test escalado: {X_test_scaled.shape}")
    logger.info(f"  Media train (post-scaling): {X_train_scaled.mean():.6f}")
    logger.info(f"  Std train (post-scaling): {X_train_scaled.std():.6f}")

    return X_train_scaled, X_test_scaled, scaler
