import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_config():
    """Carga la configuración de la API"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_hyperparameters(params, config):
    """Valida que los hiperparámetros estén dentro de rangos válidos"""
    errors = []
    hyperparams = config['hyperparameters']['clustering']
    
    for param_name, constraints in hyperparams.items():
        if param_name in params:
            value = params[param_name]
            param_type = constraints['type']
            
            # Validar tipo
            if param_type == 'integer' and not isinstance(value, int):
                errors.append(f"{param_name} debe ser un entero")
                continue
            elif param_type == 'float' and not isinstance(value, (int, float)):
                errors.append(f"{param_name} debe ser un número")
                continue
            
            # Validar rango
            if 'min' in constraints and value < constraints['min']:
                errors.append(f"{param_name} debe ser >= {constraints['min']}")
            if 'max' in constraints and value > constraints['max']:
                errors.append(f"{param_name} debe ser <= {constraints['max']}")
    
    return errors

def format_metrics_response(resultado, metricas_internas):
    """Formatea las métricas en un formato estándar"""
    return {
        'external_metrics': {
            'ari': {
                'value': float(resultado.ari),
                'name': 'Adjusted Rand Index',
                'interpretation': 'higher_better',
                'range': '[-1, 1]'
            },
            'nmi': {
                'value': float(resultado.nmi),
                'name': 'Normalized Mutual Information',
                'interpretation': 'higher_better',
                'range': '[0, 1]'
            },
            'ami': {
                'value': float(resultado.ami),
                'name': 'Adjusted Mutual Information', 
                'interpretation': 'higher_better',
                'range': '[0, 1]'
            }
        },
        'internal_metrics': {
            'silhouette': {
                'value': float(metricas_internas['silhouette_cosine']),
                'name': 'Silhouette Score (Cosine)',
                'interpretation': 'higher_better',
                'range': '[-1, 1]'
            },
            'davies_bouldin': {
                'value': float(metricas_internas['davies_bouldin']),
                'name': 'Davies-Bouldin Index',
                'interpretation': 'lower_better',
                'range': '[0, ∞]'
            },
            'calinski_harabasz': {
                'value': float(metricas_internas['calinski_harabasz']),
                'name': 'Calinski-Harabasz Index',
                'interpretation': 'higher_better',
                'range': '[0, ∞]'
            }
        }
    }

def get_model_recommendations(modelo_id, config):
    """Obtiene recomendaciones específicas para un modelo"""
    models = config['models']['available']
    model_info = next((m for m in models if m['id'] == modelo_id), None)
    
    if not model_info:
        return {}
    
    recommendations = {
        'recommended_k': model_info.get('recommended_k', 3),
        'model_specific_tips': []
    }
    
    # Consejos específicos por modelo
    if 'cnn' in modelo_id:
        recommendations['model_specific_tips'].extend([
            'CNN embeddings funcionan mejor con k=6 (2 datasets × 3 clases)',
            'Usar cluster_similarity_threshold más alto (0.8-0.9) para embeddings densos'
        ])
    elif 'sift' in modelo_id:
        recommendations['model_specific_tips'].extend([
            'SIFT tiene 512 dimensiones, puede beneficiarse de PCA',
            'Usar subcluster_similarity_threshold más bajo (0.7-0.8)'
        ])
    elif 'hu' in modelo_id:
        recommendations['model_specific_tips'].extend([
            'Momentos de Hu son invariantes, buenos para clustering robusto',
            'Solo 7 dimensiones, puede usar thresholds más estrictos'
        ])
    elif 'hog' in modelo_id:
        recommendations['model_specific_tips'].extend([
            'HOG captura gradientes locales, bueno para texturas',
            'Dimensionalidad variable según parámetros de extracción'
        ])
    
    return recommendations

def save_experiment_results(params, resultados, experiment_name=None):
    """Guarda los resultados de un experimento"""
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'results': resultados
    }
    
    # Crear directorio de experimentos si no existe
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_vectors', 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Guardar experimento
    experiment_file = os.path.join(experiments_dir, f"{experiment_name}.json")
    with open(experiment_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    return experiment_file

def get_experiment_history():
    """Obtiene el historial de experimentos"""
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_vectors', 'experiments')
    
    if not os.path.exists(experiments_dir):
        return []
    
    experiments = []
    for filename in os.listdir(experiments_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(experiments_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    experiment = json.load(f)
                    experiments.append({
                        'filename': filename,
                        'experiment_name': experiment.get('experiment_name', filename[:-5]),
                        'timestamp': experiment.get('timestamp'),
                        'num_results': len(experiment.get('results', []))
                    })
            except Exception:
                continue
    
    # Ordenar por timestamp descendente
    experiments.sort(key=lambda x: x['timestamp'] or '', reverse=True)
    return experiments

class APIError(Exception):
    """Excepción personalizada para errores de la API"""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code