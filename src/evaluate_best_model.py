import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

# Configuración
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_and_prep_data():
    """Carga los datos procesados y recrea las matrices de características escaladas."""
    
    # --- MODELO A: PILOTOS ---
    try:
        df_drivers = pd.read_csv('data/processed/driver_clusters.csv')
        cols_drivers = ['avg_grid', 'avg_finish', 'win_rate', 'podium_rate', 
                        'top10_rate', 'dnf_rate', 'avg_positions_gained']
        
        # Re-escalado (crítico para métricas justas)
        scaler_d = RobustScaler()
        X_drivers = scaler_d.fit_transform(df_drivers[cols_drivers])
        labels_drivers = df_drivers['cluster']
    except Exception as e:
        print(f"Error cargando Pilotos: {e}")
        return None, None, None, None

    # --- MODELO B: CONSTRUCTORES ---
    try:
        df_const = pd.read_csv('data/processed/constructor_clusters.csv')
        cols_const = ['avg_points_per_race', 'win_rate', 'podium_rate', 
                      'dnf_rate', 'scoring_consistency', 'avg_grid']
        
        scaler_c = RobustScaler()
        X_const = scaler_c.fit_transform(df_const[cols_const])
        labels_const = df_const['cluster']
    except Exception as e:
        print(f"Error cargando Constructores: {e}")
        return None, None, None, None
        
    return X_drivers, labels_drivers, X_const, labels_const

def calculate_kpis(X, labels, model_name):
    """Calcula los KPIs objetivos de clustering."""
    
    # 1. Davies-Bouldin (Menor es mejor): Ratio de similitud intra-cluster vs inter-cluster
    dbi = davies_bouldin_score(X, labels)
    
    # 2. Silhouette (Mayor es mejor): Qué tan bien encaja el objeto en su propio cluster
    sil = silhouette_score(X, labels)
    
    # 3. Calinski-Harabasz (Mayor es mejor): Ratio de dispersión
    chi = calinski_harabasz_score(X, labels)
    
    return {
        "Modelo": model_name,
        "N_Clusters": len(np.unique(labels)),
        "Davies-Bouldin (Minimizar)": dbi,
        "Silhouette (Maximizar)": sil,
        "Calinski-Harabasz (Maximizar)": chi
    }

def print_scorecard(metrics_a, metrics_b):
    """Imprime la tabla comparativa y declara el ganador."""
    
    df_metrics = pd.DataFrame([metrics_a, metrics_b])
    
    print("\n" + "="*80)
    print("SCORECARD DE VALIDACIÓN A/B - CLUSTERING F1")
    print("="*80)
    print(df_metrics.set_index("Modelo").round(4))
    print("-" * 80)
    
    # Lógica de decisión del ganador
    # Asignamos puntos: 1 punto por ganar en cada métrica
    score_a = 0
    score_b = 0
    
    # DBI (Menor gana)
    if metrics_a["Davies-Bouldin (Minimizar)"] < metrics_b["Davies-Bouldin (Minimizar)"]:
        score_a += 1
        winner_dbi = "Pilotos"
    else:
        score_b += 1
        winner_dbi = "Constructores"
        
    # Silhouette (Mayor gana)
    if metrics_a["Silhouette (Maximizar)"] > metrics_b["Silhouette (Maximizar)"]:
        score_a += 1
        winner_sil = "Pilotos"
    else:
        score_b += 1
        winner_sil = "Constructores"
        
    # CHI (Mayor gana)
    if metrics_a["Calinski-Harabasz (Maximizar)"] > metrics_b["Calinski-Harabasz (Maximizar)"]:
        score_a += 1
        winner_chi = "Pilotos"
    else:
        score_b += 1
        winner_chi = "Constructores"
        
    print(f"\nANÁLISIS DE VICTORIA POR KPI:")
    print(f"1. Precisión de Separación (DBI): Ganador -> {winner_dbi.upper()}")
    print(f"2. Cohesión de Grupo (Silhouette): Ganador -> {winner_sil.upper()}")
    print(f"3. Definición Matemática (CHI):   Ganador -> {winner_chi.upper()}")
    
    print("\n" + "="*80)
    print(f"VEREDICTO FINAL: EL MEJOR MODELO ES: {'PILOTOS (MODELO A)' if score_a > score_b else 'CONSTRUCTORES (MODELO B)'}")
    print("="*80)
    
    if score_a > score_b:
        print("JUSTIFICACIÓN: El modelo de Pilotos es matemáticamente más robusto.")
        print("Sus grupos están más definidos y separados entre sí, lo que indica")
        print("que las categorías de talento humano (Élite vs Resto) son más")
        print("evidentes en los datos que las categorías de equipos.")
    else:
        print("JUSTIFICACIÓN: El modelo de Constructores es superior.")
        print("Logra capturar matices más complejos con una segmentación más fina,")
        print("superando la penalización natural de tener más clusters.")

if __name__ == "__main__":
    X_d, labels_d, X_c, labels_c = load_and_prep_data()
    
    if X_d is not None and X_c is not None:
        metrics_d = calculate_kpis(X_d, labels_d, "A: Pilotos")
        metrics_c = calculate_kpis(X_c, labels_c, "B: Constructores")
        print_scorecard(metrics_d, metrics_c)
