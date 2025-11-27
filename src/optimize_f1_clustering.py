import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración
pd.set_option('display.max_columns', None)
np.random.seed(42)

# Rutas
DATA_DIR = 'data/01_raw'
OUTPUT_DIR = 'data/processed'
REPORT_DIR = 'reports/clustering'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    """Carga los datos necesarios manejando valores nulos."""
    print("Cargando datasets...")
    results = pd.read_csv(os.path.join(DATA_DIR, 'Race_Results.csv'), na_values=['\\\\N', 'NULL'])
    drivers = pd.read_csv(os.path.join(DATA_DIR, 'Driver_Details.csv'), na_values=['\\\\N', 'NULL'])
    
    # Merge básico para tener nombres
    df = results.merge(drivers[['driverId', 'driverRef', 'nationality', 'dob']], on='driverId', how='left')
    return df

def preprocess_features(df):
    """Ingeniería de características más robusta."""
    print("Procesando características...")
    
    # 1. Limpieza básica
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce').fillna(20) # Asumir fondo parrilla si nulo
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce') # Esta columna suele estar completa en Ergast
    
    # 2. Definir métricas por piloto
    # Filtrar pilotos con muy pocas carreras (ruido)
    min_races = 20
    driver_counts = df['driverId'].value_counts()
    active_drivers = driver_counts[driver_counts >= min_races].index
    
    df_filtered = df[df['driverId'].isin(active_drivers)].copy()
    
    # Agregaciones
    stats = df_filtered.groupby('driverId').agg(
        total_races=('raceId', 'count'),
        avg_grid=('grid', 'mean'),
        avg_finish=('positionOrder', 'mean'), # positionOrder es oficial, incluye retiros como últimos lugares
        wins=('positionOrder', lambda x: (x == 1).sum()),
        podiums=('positionOrder', lambda x: (x <= 3).sum()),
        top10=('positionOrder', lambda x: (x <= 10).sum()),
        dnfs=('position', lambda x: x.isna().sum()) # position es nula si no clasificó
    ).reset_index()
    
    # 3. Ratios (Mejor que totales para comparar épocas/longevidad)
    stats['win_rate'] = stats['wins'] / stats['total_races']
    stats['podium_rate'] = stats['podiums'] / stats['total_races']
    stats['top10_rate'] = stats['top10'] / stats['total_races']
    stats['dnf_rate'] = stats['dnfs'] / stats['total_races']
    
    # 4. Métricas de Calidad pura (Racecraft)
    # Ganancia de posiciones promedio (Grid - Finish)
    # Nota: positionOrder penaliza DNF poniéndolos al final, lo cual es bueno para una métrica global de "éxito"
    # pero malo para evaluar "habilidad pura". 
    # Vamos a calcular posiciones ganadas SOLO en carreras terminadas para medir "habilidad de adelantamiento"
    finished_races = df_filtered.dropna(subset=['position'])
    racecraft = finished_races.groupby('driverId').apply(
        lambda x: (x['grid'] - x['position']).mean()
    ).reset_index(name='avg_positions_gained')
    
    stats = stats.merge(racecraft, on='driverId', how='left')
    stats['avg_positions_gained'] = stats['avg_positions_gained'].fillna(0)
    
    # Unir nombres para referencia final
    driver_names = df_filtered[['driverId', 'driverRef', 'nationality']].drop_duplicates()
    final_df = stats.merge(driver_names, on='driverId', how='left')
    
    return final_df

def run_clustering(df_features):
    """Ejecuta KMeans y encuentra el óptimo."""
    print("Ejecutando clustering...")
    
    # Seleccionar columnas numéricas para el modelo
    features = ['avg_grid', 'avg_finish', 'win_rate', 'podium_rate', 
                'top10_rate', 'dnf_rate', 'avg_positions_gained']
    
    X = df_features[features]
    
    # Escalado RobustScaler (mejor para outliers que StandardScaler)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Búsqueda de K óptimo
    best_k = 0
    best_score = -1
    best_model = None
    results = []
    
    print(f"{ 'K':<5} {'Silhouette':<15} {'Calinski':<15}")
    print("-" * 35)
    
    for k in range(3, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        
        print(f"{k:<5} {sil:<15.4f} {cal:<15.4f}")
        
        results.append({'k': k, 'silhouette': sil, 'calinski': cal})
        
        if sil > best_score:
            best_score = sil
            best_k = k
            best_model = kmeans
            best_labels = labels
            
    print("-" * 35)
    print(f"Mejor K seleccionado: {best_k} (Silhouette: {best_score:.4f})")
    
    # Asignar clusters
    df_features['cluster'] = best_labels
    
    # Análisis de Componentes Principales para visualización (2D)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    df_features['pca_1'] = pca_res[:, 0]
    df_features['pca_2'] = pca_res[:, 1]
    
    return df_features, best_k, features

def evaluate_clusters(df, k, feature_cols):
    """Genera reporte de los clusters encontrados."""
    print("\n--- Análisis de Clusters ---")
    
    summary = df.groupby('cluster')[feature_cols].mean()
    print(summary)
    
    # Top drivers por cluster (para validación semántica)
    print("\n--- Pilotos Representativos por Cluster ---")
    for i in range(k):
        print(f"\nCluster {i}:")
        # Mostrar pilotos con más carreras en ese cluster como representativos
        cluster_drivers = df[df['cluster'] == i].sort_values('total_races', ascending=False).head(5)
        print(cluster_drivers[['driverRef', 'total_races', 'win_rate', 'avg_finish']])
        
    return summary

def plot_results(df, k):
    """Genera visualizaciones."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster', palette='viridis', s=100, alpha=0.8)
    
    # Etiquetar algunos pilotos famosos
    famous_drivers = ['hamilton', 'schumacher', 'senna', 'max_verstappen', 'alonso', 'latifi', 'mazepin']
    for _, row in df.iterrows():
        if row['driverRef'] in famous_drivers:
            plt.text(row['pca_1']+0.1, row['pca_2'], row['driverRef'], fontsize=9, fontweight='bold')
            
    plt.title(f'Clustering de Pilotos F1 (PCA) - K={k}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.savefig(os.path.join(REPORT_DIR, 'cluster_pca.png'))
    print(f"\nGráfico guardado en {os.path.join(REPORT_DIR, 'cluster_pca.png')}")

def save_results(df):
    """Guarda los resultados finales."""
    path = os.path.join(OUTPUT_DIR, 'driver_clusters.csv')
    df.to_csv(path, index=False)
    print(f"Resultados guardados en {path}")

def main():
    try:
        raw_df = load_data()
        processed_df = preprocess_features(raw_df)
        clustered_df, best_k, feature_cols = run_clustering(processed_df)
        evaluate_clusters(clustered_df, best_k, feature_cols)
        plot_results(clustered_df, best_k)
        save_results(clustered_df)
        print("\nProceso completado con éxito.")
    except Exception as e:
        print(f"Error fatal: {e}")

if __name__ == "__main__":
    main()
