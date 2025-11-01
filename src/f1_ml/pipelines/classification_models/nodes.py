"""
Nodos para el pipeline de modelos de clasificación (predicción de podio)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger(__name__)


def apply_smote_balancing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplicar SMOTE para balancear clases"""
    smote = SMOTE(
        sampling_strategy=params['smote_sampling_strategy'],
        random_state=params['random_state']
    )

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    logger.info(f"SMOTE aplicado: {len(y_train)} -> {len(y_train_smote)} registros")
    logger.info(f"Balance antes: {np.bincount(y_train)}")
    logger.info(f"Balance después: {np.bincount(y_train_smote)}")

    return X_train_smote, y_train_smote


def train_classification_models_base(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict
) -> Dict[str, Any]:
    """Entrenar 6 modelos base con parámetros default"""
    random_state = params['random_state']
    models = {}

    logger.info("Entrenando modelos base...")

    # 1. Logistic Regression
    logger.info("Entrenando Logistic Regression...")
    models['LogisticRegression'] = LogisticRegression(
        random_state=random_state, max_iter=1000
    )
    models['LogisticRegression'].fit(X_train, y_train)

    # 2. Random Forest
    logger.info("Entrenando Random Forest...")
    models['RandomForest'] = RandomForestClassifier(
        random_state=random_state, n_estimators=100, class_weight='balanced'
    )
    models['RandomForest'].fit(X_train, y_train)

    # 3. XGBoost
    logger.info("Entrenando XGBoost...")
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    models['XGBoost'] = xgb.XGBClassifier(
        random_state=random_state,
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    models['XGBoost'].fit(X_train, y_train)

    # 4. LightGBM
    logger.info("Entrenando LightGBM...")
    models['LightGBM'] = lgb.LGBMClassifier(
        random_state=random_state,
        n_estimators=100,
        is_unbalance=True,
        verbose=-1
    )
    models['LightGBM'].fit(X_train, y_train)

    # 5. CatBoost
    logger.info("Entrenando CatBoost...")
    models['CatBoost'] = CatBoostClassifier(
        random_state=random_state,
        iterations=100,
        auto_class_weights='Balanced',
        verbose=0
    )
    models['CatBoost'].fit(X_train, y_train)

    # 6. Gradient Boosting
    logger.info("Entrenando Gradient Boosting...")
    models['GradientBoosting'] = GradientBoostingClassifier(
        random_state=random_state,
        n_estimators=100
    )
    models['GradientBoosting'].fit(X_train, y_train)

    logger.info("Entrenamiento de modelos base completado")
    return models


def evaluate_classification_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Evaluar modelos con F1, Precision, Recall, ROC-AUC"""
    results = []

    for name, model in models.items():
        logger.info(f"Evaluando {name}...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'model': name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        })

        logger.info(f"{name} - F1: {f1:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

    results_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
    return results_df


def select_top5_classification(
    results_df: pd.DataFrame,
    models: Dict[str, Any]
) -> Dict:
    """Seleccionar TOP 5 modelos por F1-Score"""
    top5_names = results_df.head(5)['model'].tolist()
    top5_models = {name: models[name] for name in top5_names}

    logger.info(f"TOP 5 modelos seleccionados: {top5_names}")

    return {
        'top5_names': top5_names,
        'top5_models': top5_models
    }


def optimize_classification_gridsearch(
    top5_dict: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict
) -> Dict[str, Any]:
    """Optimizar TOP 5 con GridSearch"""
    top5_names = top5_dict['top5_names']
    gridsearch_grids = params['gridsearch_grids']
    scoring = params['scoring']
    cv_folds = params['cv_folds']
    random_state = params['random_state']

    optimized_models = {}
    best_params = {}

    # Calcular scale_pos_weight para XGBoost si está en TOP 5
    if 'XGBoost' in top5_names:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count

        # Agregar a grid con variantes
        gridsearch_grids['XGBoost']['scale_pos_weight'] = [
            scale_pos_weight,
            scale_pos_weight * 0.8,
            scale_pos_weight * 1.2
        ]

    for model_name in top5_names:
        logger.info(f"Optimizando {model_name} con GridSearch...")

        # Crear modelo base
        if model_name == 'LogisticRegression':
            base_model = LogisticRegression(random_state=random_state)
        elif model_name == 'RandomForest':
            base_model = RandomForestClassifier(random_state=random_state)
        elif model_name == 'XGBoost':
            base_model = xgb.XGBClassifier(random_state=random_state, eval_metric='logloss')
        elif model_name == 'LightGBM':
            base_model = lgb.LGBMClassifier(random_state=random_state, verbose=-1)
        elif model_name == 'CatBoost':
            base_model = CatBoostClassifier(random_state=random_state, verbose=0)
        elif model_name == 'GradientBoosting':
            base_model = GradientBoostingClassifier(random_state=random_state)
        else:
            logger.warning(f"Modelo {model_name} no reconocido, saltando...")
            continue

        # GridSearch
        param_grid = gridsearch_grids.get(model_name, {})

        if not param_grid:
            logger.warning(f"No hay param_grid para {model_name}, usando modelo base")
            optimized_models[model_name] = base_model
            optimized_models[model_name].fit(X_train, y_train)
            continue

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        optimized_models[model_name] = grid_search.best_estimator_
        best_params[model_name] = grid_search.best_params_

        logger.info(f"{model_name} - Mejores parámetros: {grid_search.best_params_}")
        logger.info(f"{model_name} - Mejor F1 (CV): {grid_search.best_score_:.4f}")

    return {
        'models': optimized_models,
        'best_params': best_params
    }


def evaluate_optimized_classification(
    optimized_dict: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Evaluar modelos optimizados"""
    models = optimized_dict['models']
    results = []

    for name, model in models.items():
        logger.info(f"Evaluando {name} optimizado...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'model': name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        })

        logger.info(f"{name} - F1: {f1:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"{name} - Confusion Matrix:\n{cm}")

    results_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
    return results_df


def save_best_classification_model(
    optimized_dict: Dict,
    results_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Any, str, Dict]:
    """Guardar mejor modelo por F1-test"""
    best_model_name = results_df.iloc[0]['model']
    best_model = optimized_dict['models'][best_model_name]

    logger.info(f"Mejor modelo: {best_model_name}")
    logger.info(f"F1-Score: {results_df.iloc[0]['f1_score']:.4f}")

    # Generar reporte de clasificación
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")

    # Métricas finales
    metricas_finales = {
        'model_name': best_model_name,
        'f1_score': float(results_df.iloc[0]['f1_score']),
        'precision': float(results_df.iloc[0]['precision']),
        'recall': float(results_df.iloc[0]['recall']),
        'roc_auc': float(results_df.iloc[0]['roc_auc']),
        'best_params': optimized_dict['best_params'].get(best_model_name, {})
    }

    return best_model, best_model_name, metricas_finales
