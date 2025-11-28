import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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
    
    return df_features, best_k, features, X_scaled, best_model

def detect_anomalies(df_features, X_scaled, kmeans_model, feature_cols):
    """Detecta anomalías usando múltiples técnicas."""
    print("\n--- Detección de Anomalías ---")

    # 1. Isolation Forest (detección global)
    print("Ejecutando Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df_features['anomaly_isolation'] = iso_forest.fit_predict(X_scaled)
    # -1 para anomalías, 1 para normales
    df_features['is_anomaly_isolation'] = df_features['anomaly_isolation'] == -1

    # Score de anomalía (más negativo = más anómalo)
    df_features['anomaly_score_isolation'] = iso_forest.score_samples(X_scaled)

    # 2. Distancia al centroide del cluster
    print("Calculando distancias a centroides...")
    centroids = kmeans_model.cluster_centers_
    distances = []

    for idx, row in df_features.iterrows():
        cluster_id = row['cluster']
        centroid = centroids[cluster_id]
        # Encontrar el índice original en X_scaled
        point_idx = df_features.index.get_loc(idx)
        point = X_scaled[point_idx]
        distance = np.linalg.norm(point - centroid)
        distances.append(distance)

    df_features['distance_to_centroid'] = distances

    # Detectar anomalías por distancia (> percentil 90 dentro de cada cluster)
    df_features['is_anomaly_distance'] = False
    for cluster_id in df_features['cluster'].unique():
        cluster_mask = df_features['cluster'] == cluster_id
        threshold = df_features.loc[cluster_mask, 'distance_to_centroid'].quantile(0.90)
        anomaly_mask = (cluster_mask) & (df_features['distance_to_centroid'] > threshold)
        df_features.loc[anomaly_mask, 'is_anomaly_distance'] = True

    # 3. Local Outlier Factor
    print("Ejecutando Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    df_features['anomaly_lof'] = lof.fit_predict(X_scaled)
    df_features['is_anomaly_lof'] = df_features['anomaly_lof'] == -1
    df_features['lof_score'] = lof.negative_outlier_factor_

    # 4. Consenso: anomalía si al menos 2 métodos la detectan
    df_features['anomaly_consensus'] = (
        df_features['is_anomaly_isolation'].astype(int) +
        df_features['is_anomaly_distance'].astype(int) +
        df_features['is_anomaly_lof'].astype(int)
    )
    df_features['is_anomaly_final'] = df_features['anomaly_consensus'] >= 2

    # Estadísticas
    n_isolation = df_features['is_anomaly_isolation'].sum()
    n_distance = df_features['is_anomaly_distance'].sum()
    n_lof = df_features['is_anomaly_lof'].sum()
    n_final = df_features['is_anomaly_final'].sum()

    print(f"\nAnomalías detectadas:")
    print(f"  - Isolation Forest: {n_isolation} pilotos")
    print(f"  - Distancia a centroide: {n_distance} pilotos")
    print(f"  - Local Outlier Factor: {n_lof} pilotos")
    print(f"  - Consenso (≥2 métodos): {n_final} pilotos")

    return df_features

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

def analyze_anomalies(df):
    """Genera reporte detallado de anomalías."""
    print("\n--- Reporte de Anomalías ---")

    anomalies = df[df['is_anomaly_final'] == True].copy()

    if len(anomalies) == 0:
        print("No se detectaron anomalías con el criterio de consenso.")
        return

    # Ordenar por número de métodos que detectaron la anomalía
    anomalies = anomalies.sort_values('anomaly_consensus', ascending=False)

    print(f"\nSe detectaron {len(anomalies)} pilotos anómalos:\n")

    # Mostrar información detallada
    cols_to_show = ['driverRef', 'nationality', 'cluster', 'total_races',
                    'win_rate', 'podium_rate', 'avg_finish', 'avg_positions_gained',
                    'anomaly_consensus', 'distance_to_centroid']

    print(anomalies[cols_to_show].to_string(index=False))

    # Análisis por cluster
    print("\n--- Distribución de Anomalías por Cluster ---")
    anomaly_dist = anomalies.groupby('cluster').size()
    for cluster_id, count in anomaly_dist.items():
        total_in_cluster = (df['cluster'] == cluster_id).sum()
        pct = (count / total_in_cluster) * 100
        print(f"Cluster {cluster_id}: {count}/{total_in_cluster} pilotos ({pct:.1f}%)")

    # Top anomalías más extremas
    print("\n--- Top 5 Anomalías Más Extremas (por Isolation Forest) ---")
    top_anomalies = df.nsmallest(5, 'anomaly_score_isolation')
    for _, row in top_anomalies.iterrows():
        print(f"\n{row['driverRef']} ({row['nationality']}):")
        print(f"  Cluster: {row['cluster']}")
        print(f"  Total carreras: {row['total_races']}")
        print(f"  Win rate: {row['win_rate']:.3f}")
        print(f"  Podium rate: {row['podium_rate']:.3f}")
        print(f"  Avg finish: {row['avg_finish']:.2f}")
        print(f"  Avg positions gained: {row['avg_positions_gained']:.2f}")
        print(f"  Anomaly score: {row['anomaly_score_isolation']:.3f}")
        print(f"  Métodos que lo detectaron: {row['anomaly_consensus']}/3")

    return anomalies

def plot_results(df, k):
    """Genera visualizaciones de clustering y anomalías."""

    # Gráfico 1: Clustering básico con PCA
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
    plt.savefig(os.path.join(REPORT_DIR, 'cluster_pca.png'), dpi=150, bbox_inches='tight')
    print(f"\nGráfico guardado en {os.path.join(REPORT_DIR, 'cluster_pca.png')}")
    plt.close()

    # Gráfico 2: Clusters con anomalías resaltadas
    fig, ax = plt.subplots(figsize=(14, 10))

    # Puntos normales
    normal = df[df['is_anomaly_final'] == False]
    anomalies = df[df['is_anomaly_final'] == True]

    # Plotear normales
    sns.scatterplot(data=normal, x='pca_1', y='pca_2', hue='cluster',
                    palette='viridis', s=100, alpha=0.6, ax=ax, legend='full')

    # Plotear anomalías con marcador diferente
    if len(anomalies) > 0:
        ax.scatter(anomalies['pca_1'], anomalies['pca_2'],
                   c='red', s=300, alpha=0.7, marker='X',
                   edgecolors='darkred', linewidths=2, label='Anomalías')

        # Etiquetar anomalías
        for _, row in anomalies.iterrows():
            ax.annotate(row['driverRef'],
                       (row['pca_1'], row['pca_2']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_title(f'Clustering con Detección de Anomalías - K={k}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Componente Principal 1', fontsize=12)
    ax.set_ylabel('Componente Principal 2', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(REPORT_DIR, 'cluster_anomalies_pca.png'), dpi=150, bbox_inches='tight')
    print(f"Gráfico de anomalías guardado en {os.path.join(REPORT_DIR, 'cluster_anomalies_pca.png')}")
    plt.close()

    # Gráfico 3: Distribución de scores de anomalía
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Isolation Forest scores
    axes[0, 0].hist(df['anomaly_score_isolation'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['anomaly_score_isolation'].quantile(0.1), color='red', linestyle='--',
                       label='Threshold (10%)')
    axes[0, 0].set_title('Distribución de Anomaly Scores (Isolation Forest)', fontweight='bold')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Distance to centroid
    axes[0, 1].hist(df['distance_to_centroid'], bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribución de Distancias al Centroide', fontweight='bold')
    axes[0, 1].set_xlabel('Distancia al Centroide')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].grid(True, alpha=0.3)

    # LOF scores
    axes[1, 0].hist(df['lof_score'], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['lof_score'].quantile(0.1), color='red', linestyle='--',
                       label='Threshold (10%)')
    axes[1, 0].set_title('Distribución de LOF Scores', fontweight='bold')
    axes[1, 0].set_xlabel('LOF Score (Negative Outlier Factor)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Consensus bar chart
    consensus_counts = df['anomaly_consensus'].value_counts().sort_index()
    axes[1, 1].bar(consensus_counts.index, consensus_counts.values,
                   color=['green', 'yellow', 'orange', 'red'][:len(consensus_counts)],
                   alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Consenso de Métodos de Detección', fontweight='bold')
    axes[1, 1].set_xlabel('Número de Métodos que Detectaron Anomalía')
    axes[1, 1].set_ylabel('Cantidad de Pilotos')
    axes[1, 1].set_xticks([0, 1, 2, 3])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'anomaly_scores_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"Distribución de scores guardada en {os.path.join(REPORT_DIR, 'anomaly_scores_distribution.png')}")
    plt.close()

def save_results(df):
    """Guarda los resultados finales."""
    # Guardar dataset completo
    path = os.path.join(OUTPUT_DIR, 'driver_clusters.csv')
    df.to_csv(path, index=False)
    print(f"\nResultados guardados en {path}")

    # Guardar solo anomalías
    anomalies = df[df['is_anomaly_final'] == True]
    if len(anomalies) > 0:
        anomalies_path = os.path.join(OUTPUT_DIR, 'driver_anomalies.csv')
        anomalies.to_csv(anomalies_path, index=False)
        print(f"Anomalías guardadas en {anomalies_path}")

def main():
    try:
        # 1. Cargar y procesar datos
        raw_df = load_data()
        processed_df = preprocess_features(raw_df)

        # 2. Ejecutar clustering
        clustered_df, best_k, feature_cols, X_scaled, kmeans_model = run_clustering(processed_df)

        # 3. Evaluar clusters
        evaluate_clusters(clustered_df, best_k, feature_cols)

        # 4. Detectar anomalías
        clustered_df = detect_anomalies(clustered_df, X_scaled, kmeans_model, feature_cols)

        # 5. Analizar anomalías
        analyze_anomalies(clustered_df)

        # 6. Generar visualizaciones
        plot_results(clustered_df, best_k)

        # 7. Guardar resultados
        save_results(clustered_df)

        print("\n" + "="*60)
        print("Proceso completado con éxito.")
        print("="*60)
    except Exception as e:
        print(f"Error fatal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
