"""
This is a boilerplate pipeline 'clustering_pilotos'
generated using Kedro 0.19.10
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_driver_data(race_results: pd.DataFrame, driver_details: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Preprocesa los datos de pilotos para clustering.
    Calcula métricas agregadas por piloto (Ratios, Racecraft).
    
    Args:
        race_results: DataFrame con resultados de carreras.
        driver_details: DataFrame con detalles de pilotos.
        params: Diccionario de parámetros.
        
    Returns:
        DataFrame con características agregadas por piloto.
    """
    # Manejo de nulos (Kedro ya carga como NaN si está configurado, pero reforzamos)
    race_results = race_results.replace('\\N', np.nan)
    
    # Merge para tener nombres y nacionalidades
    df = race_results.merge(
        driver_details[['driverId', 'driverRef', 'nationality', 'dob']], 
        on='driverId', 
        how='left'
    )
    
    # Conversiones de tipo
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce').fillna(20)
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    
    # Filtrar pilotos con pocas carreras
    min_races = params.get('min_races', 20)
    driver_counts = df['driverId'].value_counts()
    active_drivers = driver_counts[driver_counts >= min_races].index
    df_filtered = df[df['driverId'].isin(active_drivers)].copy()
    
    # Agregaciones básicas
    stats = df_filtered.groupby('driverId').agg(
        total_races=('raceId', 'count'),
        avg_grid=('grid', 'mean'),
        avg_finish=('positionOrder', 'mean'),
        wins=('positionOrder', lambda x: (x == 1).sum()),
        podiums=('positionOrder', lambda x: (x <= 3).sum()),
        top10=('positionOrder', lambda x: (x <= 10).sum()),
        dnfs=('position', lambda x: x.isna().sum())
    ).reset_index()
    
    # Cálculo de Ratios
    stats['win_rate'] = stats['wins'] / stats['total_races']
    stats['podium_rate'] = stats['podiums'] / stats['total_races']
    stats['top10_rate'] = stats['top10'] / stats['total_races']
    stats['dnf_rate'] = stats['dnfs'] / stats['total_races']
    
    # Métrica de Racecraft (Ganancia de posiciones en carreras terminadas)
    finished_races = df_filtered.dropna(subset=['position'])
    
    # Fix para FutureWarning de pandas groupby.apply
    def calculate_positions_gained(x):
        return (x['grid'] - x['position']).mean()

    racecraft = finished_races.groupby('driverId').apply(calculate_positions_gained).reset_index(name='avg_positions_gained')
    
    stats = stats.merge(racecraft, on='driverId', how='left')
    stats['avg_positions_gained'] = stats['avg_positions_gained'].fillna(0)
    
    # Unir metadatos finales
    driver_names = df_filtered[['driverId', 'driverRef', 'nationality']].drop_duplicates()
    final_df = stats.merge(driver_names, on='driverId', how='left')
    
    return final_df

def train_driver_clustering(df_features: pd.DataFrame, params: dict) -> tuple:
    """
    Entrena el modelo KMeans para segmentación de pilotos.
    
    Args:
        df_features: DataFrame con features de pilotos.
        params: Diccionario de parámetros.
        
    Returns:
        Tupla (Modelo KMeans entrenado, DataFrame con etiquetas de cluster).
    """
    features = params.get('features', [
        'avg_grid', 'avg_finish', 'win_rate', 'podium_rate', 
        'top10_rate', 'dnf_rate', 'avg_positions_gained'
    ])
    
    X = df_features[features]
    
    # Escalado RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenamiento con K óptimo (definido en parámetros o encontrado)
    k = params.get('n_clusters', 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Asignar resultados
    df_clustered = df_features.copy()
    df_clustered['cluster'] = labels
    
    # Calcular PCA para visualización posterior (opcional guardarlo en el df)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    df_clustered['pca_1'] = pca_res[:, 0]
    df_clustered['pca_2'] = pca_res[:, 1]
    
    return kmeans, df_clustered

def evaluate_driver_clustering(df_clustered: pd.DataFrame, params: dict) -> dict:
    """
    Evalúa el clustering y genera métricas.
    
    Args:
        df_clustered: DataFrame con etiquetas de cluster.
        params: Diccionario de parámetros.
        
    Returns:
        Diccionario con métricas (Silhouette, etc.).
    """
    features = params.get('features', [
        'avg_grid', 'avg_finish', 'win_rate', 'podium_rate', 
        'top10_rate', 'dnf_rate', 'avg_positions_gained'
    ])
    
    # Reconstruir X scaled para métricas (idealmente pasar el scaler, pero aquí lo recreamos rápido)
    # Nota: Para consistencia estricta deberíamos pasar el scaler, pero RobustScaler es determinista.
    X = df_clustered[features]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    labels = df_clustered['cluster']
    
    sil = silhouette_score(X_scaled, labels)
    cal = calinski_harabasz_score(X_scaled, labels)
    
    # Perfilamiento de clusters
    summary = df_clustered.groupby('cluster')[features].mean()
    
    metrics = {
        "n_clusters": len(labels.unique()),
        "silhouette_score": float(sil),
        "calinski_harabasz_score": float(cal),
        "cluster_profiles": summary.to_dict()
    }
    
    return metrics