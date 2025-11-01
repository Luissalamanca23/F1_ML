#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTOR DE PRODUCCIÓN - MODELO F1
====================================

Script para hacer predicciones de posición final usando el modelo entrenado.

Uso:
    python predictor_produccion.py --driver_id 844 --constructor_id 131 \
                                   --circuit_id 18 --year 2025 --quali_position 2

Autor: Pipeline F1 ML
Fecha: 2025-10-27
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import json
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_DIR = BASE_DIR / "data" / "06_models"
DATOS_DIR = BASE_DIR / "data" / "05_model_input"
METRICAS_DIR = BASE_DIR / "data" / "07_model_output"

# =============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# =============================================================================

def crear_features_avanzadas(df):
    """Aplica el mismo feature engineering que en entrenamiento."""
    df_fe = df.copy()

    # 1. Diferencia grid vs position_quali
    df_fe['diferencia_grid_quali'] = df_fe['grid'] - df_fe['position_quali']

    # 2. Ratio puntos piloto / constructor
    df_fe['ratio_puntos_piloto_constructor'] = np.where(
        df_fe['puntos_campeonato_constructor'] > 0,
        df_fe['puntos_campeonato_piloto'] / df_fe['puntos_campeonato_constructor'],
        0
    )

    # 3. Interacción quali * puntos constructor
    df_fe['quali_x_puntos_constructor'] = df_fe['position_quali'] * df_fe['puntos_campeonato_constructor']

    # 4. Progreso temporada
    df_fe['progreso_temporada'] = df_fe['round'] / 24.0

    # 5. Diferencia posición campeonato
    df_fe['diff_posicion_piloto_constructor'] = df_fe['posicion_campeonato_piloto'] - df_fe['posicion_campeonato_constructor']

    # 6. Log de puntos
    df_fe['log_puntos_piloto'] = np.log1p(df_fe['puntos_campeonato_piloto'])
    df_fe['log_puntos_constructor'] = np.log1p(df_fe['puntos_campeonato_constructor'])

    # 7. Interacción quali × grid
    df_fe['quali_x_grid'] = df_fe['position_quali'] * df_fe['grid']

    # 8. Altitud extrema
    df_fe['altitud_extrema'] = (df_fe['altitud'] > 1000).astype(int)

    return df_fe


def preparar_datos_prediccion(datos_entrada):
    """
    Prepara los datos de entrada para predicción.

    Flujo:
    1. Feature Engineering (9 features avanzadas)
    2. Target Encoding (driverId, constructorId, circuitId)
    3. One-Hot Encoding (nombre_circuito, ubicacion_circuito, pais_circuito)
    4. Alinear columnas con las esperadas por el scaler
    5. StandardScaler

    Args:
        datos_entrada (dict): Diccionario con features necesarias

    Returns:
        np.ndarray: Array escalado listo para predicción
    """
    # Crear DataFrame
    df = pd.DataFrame([datos_entrada])

    # PASO 1: Aplicar feature engineering (9 features)
    df_fe = crear_features_avanzadas(df)

    # PASO 2: Cargar y aplicar target encoding a IDs
    target_encoder = joblib.load(DATOS_DIR / "regresion_target_encoder.pkl")
    cols_ids = ['driverId', 'constructorId', 'circuitId']
    df_fe[cols_ids] = target_encoder.transform(df_fe[cols_ids])

    # PASO 3: Aplicar One-Hot Encoding a columnas de texto
    cols_texto = ['nombre_circuito', 'ubicacion_circuito', 'pais_circuito']
    df_fe_encoded = pd.get_dummies(df_fe, columns=cols_texto, drop_first=True)

    # PASO 4: Cargar scaler
    scaler = joblib.load(DATOS_DIR / "regresion_scaler.pkl")

    # PASO 5: Asegurar que tenemos todas las columnas esperadas
    # (en producción, algunas columnas one-hot pueden faltar)
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df_fe_encoded.columns:
            df_fe_encoded[col] = 0

    # Reordenar columnas para que coincidan con el scaler
    df_fe_encoded = df_fe_encoded[expected_cols]

    # PASO 6: Escalar con StandardScaler
    X_scaled = scaler.transform(df_fe_encoded)

    return X_scaled


def predecir_posicion(datos_entrada):
    """
    Hace predicción de posición final.

    Args:
        datos_entrada (dict): Features de entrada

    Returns:
        dict: Predicción y estadísticas
    """
    # Cargar modelo final
    modelo = joblib.load(MODELOS_DIR / "regresion_modelo_final.pkl")

    # Cargar nombre del modelo
    try:
        with open(MODELOS_DIR / "regresion_nombre_modelo_final.txt", "r") as f:
            nombre_modelo = f.read().strip()
    except FileNotFoundError:
        nombre_modelo = "Modelo desconocido"

    # Cargar métricas finales del modelo
    try:
        with open(METRICAS_DIR / "regresion_metricas_finales.json", "r") as f:
            metricas = json.load(f)
            mae = metricas.get("mae_test", 2.5)
            r2 = metricas.get("r2_test", 0.0)
            rmse = metricas.get("rmse_test", 3.0)
    except FileNotFoundError:
        mae = 2.5  # valor por defecto conservador
        r2 = 0.0
        rmse = 3.0

    # Preparar datos
    X = preparar_datos_prediccion(datos_entrada)

    # Predecir
    posicion_predicha = modelo.predict(X)[0]

    # Redondear a posición entera más cercana
    posicion_redondeada = int(round(posicion_predicha))

    # Intervalo de confianza aproximado (± MAE)
    posicion_min = max(1, int(np.floor(posicion_predicha - mae)))
    posicion_max = min(24, int(np.ceil(posicion_predicha + mae)))

    return {
        'posicion_predicha': posicion_predicha,
        'posicion_redondeada': posicion_redondeada,
        'intervalo_confianza': (posicion_min, posicion_max),
        'mae_esperado': mae,
        'r2_test': r2,
        'rmse_test': rmse,
        'nombre_modelo': nombre_modelo
    }


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predictor de posición final F1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
    python predictor_produccion.py --driver_id 844 --constructor_id 131 \
                                   --circuit_id 18 --year 2025 --quali_position 2 \
                                   --round 5 --grid 2
        """
    )

    # Argumentos requeridos
    parser.add_argument('--driver_id', type=int, required=True, help='ID del piloto')
    parser.add_argument('--constructor_id', type=int, required=True, help='ID del constructor/equipo')
    parser.add_argument('--circuit_id', type=int, required=True, help='ID del circuito')
    parser.add_argument('--year', type=int, required=True, help='Año de la carrera')
    parser.add_argument('--quali_position', type=float, required=True, help='Posición en qualifying')

    # Argumentos opcionales
    parser.add_argument('--round', type=int, default=1, help='Número de carrera en temporada (default: 1)')
    parser.add_argument('--grid', type=float, default=None, help='Posición en parrilla (default: igual a quali)')
    parser.add_argument('--puntos_piloto', type=float, default=0.0, help='Puntos del piloto (default: 0)')
    parser.add_argument('--puntos_constructor', type=float, default=0.0, help='Puntos del constructor (default: 0)')
    parser.add_argument('--posicion_piloto', type=float, default=10.0, help='Posición piloto en campeonato (default: 10)')
    parser.add_argument('--posicion_constructor', type=float, default=5.0, help='Posición constructor en campeonato (default: 5)')
    parser.add_argument('--victorias_constructor', type=int, default=0, help='Victorias del constructor (default: 0)')
    parser.add_argument('--circuito_nombre', type=str, default='Unknown Circuit', help='Nombre del circuito')
    parser.add_argument('--circuito_ubicacion', type=str, default='Unknown', help='Ubicación del circuito')
    parser.add_argument('--circuito_pais', type=str, default='Unknown', help='País del circuito')
    parser.add_argument('--latitud', type=float, default=0.0, help='Latitud del circuito')
    parser.add_argument('--longitud', type=float, default=0.0, help='Longitud del circuito')
    parser.add_argument('--altitud', type=float, default=0.0, help='Altitud del circuito')

    args = parser.parse_args()

    # Si grid no se especifica, usar quali_position
    grid = args.grid if args.grid is not None else args.quali_position

    # Construir diccionario de entrada
    datos_entrada = {
        'driverId': args.driver_id,
        'constructorId': args.constructor_id,
        'circuitId': args.circuit_id,
        'year': args.year,
        'round': args.round,
        'grid': grid,
        'position_quali': args.quali_position,
        'puntos_campeonato_piloto': args.puntos_piloto,
        'posicion_campeonato_piloto': args.posicion_piloto,
        'puntos_campeonato_constructor': args.puntos_constructor,
        'posicion_campeonato_constructor': args.posicion_constructor,
        'victorias_constructor': args.victorias_constructor,
        'nombre_circuito': args.circuito_nombre,
        'ubicacion_circuito': args.circuito_ubicacion,
        'pais_circuito': args.circuito_pais,
        'latitud': args.latitud,
        'longitud': args.longitud,
        'altitud': args.altitud
    }

    print("=" * 80)
    print("PREDICTOR F1 - POSICIÓN FINAL")
    print("=" * 80)
    print(f"\nDatos de entrada:")
    print(f"  Piloto ID: {args.driver_id}")
    print(f"  Constructor ID: {args.constructor_id}")
    print(f"  Circuito: {args.circuito_nombre} (ID: {args.circuit_id})")
    print(f"  Año: {args.year}, Carrera #{args.round}")
    print(f"  Qualifying: P{int(args.quali_position)}")
    print(f"  Grid: P{int(grid)}")

    # Hacer predicción
    print(f"\n{'Procesando...':<80}")
    resultado = predecir_posicion(datos_entrada)

    # Mostrar resultado
    print(f"\n{'=' * 80}")
    print("RESULTADO DE LA PREDICCIÓN")
    print("=" * 80)
    print(f"\n  Modelo utilizado: {resultado['nombre_modelo']}")
    print(f"  R² Test: {resultado['r2_test']:.4f}")
    print(f"  MAE Test: {resultado['mae_esperado']:.2f} posiciones")
    print(f"  RMSE Test: {resultado['rmse_test']:.2f} posiciones")
    print(f"\n  Posición predicha: P{resultado['posicion_redondeada']}")
    print(f"  Valor exacto: {resultado['posicion_predicha']:.2f}")
    print(f"  Intervalo de confianza (±{resultado['mae_esperado']:.2f} posiciones): P{resultado['intervalo_confianza'][0]} - P{resultado['intervalo_confianza'][1]}")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
