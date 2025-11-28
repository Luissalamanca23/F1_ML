import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
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
    """Carga los datos de resultados y equipos."""
    print("Cargando datasets...")
    # Usar \N para manejar los nulos de datasets tipo Ergast
    results = pd.read_csv(os.path.join(DATA_DIR, 'Race_Results.csv'), na_values=['\\N', 'NULL'])
    teams = pd.read_csv(os.path.join(DATA_DIR, 'Team_Details.csv'), na_values=['\\N', 'NULL'])
    
    # Unir para tener nombres de equipos
    df = results.merge(teams[['constructorId', 'constructorRef', 'nationality']], on='constructorId', how='left')
    return df

def preprocess_constructor_features(df):
    """
    Ingeniería de características enfocada en el rendimiento de EQUIPOS (Constructores).
    Objetivo: Identificar jerarquías históricas (Dominantes, Midfield, Backmarkers).
    """
    print("Generando features de Constructores...")
    
    # Limpieza
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce').fillna(20)
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
    
    # Agrupación por Carrera y Equipo (para métricas combinadas de sus 2 pilotos)
    # Primero, agregamos por constructor globalmente
    
    # Filtrar equipos efímeros (menos de 30 carreras) para evitar ruido
    race_counts = df.groupby('constructorId')['raceId'].nunique()
    active_teams = race_counts[race_counts >= 30].index
    df_filtered = df[df['constructorId'].isin(active_teams)].copy()
    
    # Calcular métricas
    stats = df_filtered.groupby('constructorId').agg(
        total_entries=('raceId', 'count'), # Total de coches inscritos (2 por carrera usualmente)
        races_participated=('raceId', 'nunique'), # Carreras reales
        total_points=('points', 'sum'),
        wins=('positionOrder', lambda x: (x == 1).sum()),
        podiums=('positionOrder', lambda x: (x <= 3).sum()),
        top10=('positionOrder', lambda x: (x <= 10).sum()),
        dnfs=('position', lambda x: x.isna().sum()), # position es null en DNF
        avg_grid=('grid', 'mean')
    ).reset_index()
    
    # Métricas de Dominio y Fiabilidad
    stats['avg_points_per_race'] = stats['total_points'] / stats['races_participated']
    stats['win_rate'] = stats['wins'] / stats['races_participated']
    stats['podium_rate'] = stats['podiums'] / stats['races_participated']
    stats['dnf_rate'] = stats['dnfs'] / stats['total_entries'] # DNF por coche inscrito
    
    # Métrica de "Doble Amenaza" (Qué tan seguido meten ambos coches en puntos)
    # Esto requiere agrupar por raceId primero, pero usaremos una proxy: top10_rate ajustado
    stats['scoring_consistency'] = stats['top10'] / stats['total_entries']
    
    # Unir nombres
    team_names = df_filtered[['constructorId', 'constructorRef', 'nationality']].drop_duplicates()
    final_df = stats.merge(team_names, on='constructorId', how='left')
    
    return final_df

def run_clustering(df_features):
    print("Ejecutando KMeans para Constructores...")
    
    features = ['avg_points_per_race', 'win_rate', 'podium_rate', 
                'dnf_rate', 'scoring_consistency', 'avg_grid']
    
    X = df_features[features]
    
    # RobustScaler es vital aquí porque Mercedes/Ferrari tienen stats muy fuera de escala
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Búsqueda de K
    best_k = 0
    best_score = -1
    best_model = None
    
    print(f"{ 'K':<5} {'Silhouette':<15} {'Calinski':<15}")
    print("-" * 35)
    
    for k in range(3, 8): # Rango más pequeño para equipos (menos categorías naturales)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        
        print(f"{k:<5} {sil:<15.4f} {cal:<15.4f}")
        
        if sil > best_score:
            best_score = sil
            best_k = k
            best_model = kmeans
            best_labels = labels
            
    print("-" * 35)
    print(f"Mejor K seleccionado: {best_k} (Silhouette: {best_score:.4f})")
    
    df_features['cluster'] = best_labels
    
    # PCA para visualización
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    df_features['pca_1'] = pca_res[:, 0]
    df_features['pca_2'] = pca_res[:, 1]
    
    return df_features, best_k, features, best_score

def compare_models(current_score, current_k):
    """Compara con el modelo de Pilotos anterior (Hardcoded stats del run previo)."""
    
    # Datos del run anterior (guardados en memoria/contexto)
    prev_model_name = "Clustering Pilotos (Performance)"
    prev_k = 3
    prev_score = 0.4512
    
    print("\n" + "="*50)
    print("COMPARACIÓN DE MODELOS")
    print("="*50)
    print(f"{ 'Modelo':<30} {'K':<5} {'Silhouette Score':<15}")
    print("-" * 50)
    print(f"{prev_model_name:<30} {prev_k:<5} {prev_score:<15.4f}")
    print(f"{ 'Clustering Constructores':<30} {current_k:<5} {current_score:<15.4f}")
    print("-" * 50)
    
    diff = current_score - prev_score
    print(f"Diferencia de Calidad (Silhouette): {diff:+.4f}")
    if diff > 0:
        print(">>")
        print("El clustering de Constructores tiene grupos más definidos.")
    else:
        print(">>")
        print("El clustering de Pilotos generó grupos más compactos/separados.")

def evaluate_clusters(df, k, feature_cols):
    print("\n--- Perfil de Clusters de Equipos ---")
    summary = df.groupby('cluster')[feature_cols].mean()
    print(summary)
    
    print("\n--- Equipos Representativos ---")
    for i in range(k):
        print(f"\nCluster {i}:")
        # Ordenar por total de carreras para sacar los históricos
        teams = df[df['cluster'] == i].sort_values('total_entries', ascending=False).head(5)
        print(teams[['constructorRef', 'races_participated', 'win_rate', 'podium_rate']])

def plot_results(df, k):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster', palette='deep', s=100, alpha=0.8)
    
    # Etiquetas
    top_teams = ['ferrari', 'mclaren', 'williams', 'mercedes', 'red_bull', 'haas', 'minardi', 'tyrrell', 'lotus_f1']
    for _, row in df.iterrows():
        if row['constructorRef'] in top_teams:
            plt.text(row['pca_1'], row['pca_2']+0.1, row['constructorRef'], fontsize=9, fontweight='bold')
            
    plt.title(f'Clustering de Constructores F1 (PCA) - K={k}')
    plt.xlabel('Componente 1 (Rendimiento)')
    plt.ylabel('Componente 2 (Consistencia/Fiabilidad)')
    plt.savefig(os.path.join(REPORT_DIR, 'constructor_pca.png'))
    print(f"\nGráfico guardado en {os.path.join(REPORT_DIR, 'constructor_pca.png')}")

def main():
    try:
        raw_df = load_data()
        processed_df = preprocess_constructor_features(raw_df)
        clustered_df, best_k, feats, best_score = run_clustering(processed_df)
        
        evaluate_clusters(clustered_df, best_k, feats)
        compare_models(best_score, best_k)
        plot_results(clustered_df, best_k)
        
        # Guardar
        clustered_df.to_csv(os.path.join(OUTPUT_DIR, 'constructor_clusters.csv'), index=False)
        print(f"\nDatos guardados en {os.path.join(OUTPUT_DIR, 'constructor_clusters.csv')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
