from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Agregar src al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import evaluar_clustering, cargar_csv_features, cargar_npy_features, cargar_modelo_por_dataset
from clustering.online import online_capacity_links_with_metrics, _safe_internal_metrics
from clustering.models import OnlineClusteringAPI
from sklearn.preprocessing import StandardScaler, normalize
from api.utils import (
    load_config, validate_hyperparameters, format_metrics_response,
    get_model_recommendations, save_experiment_results, get_experiment_history, APIError
)

app = Flask(__name__)
CORS(app)

# Configuración
FEATURE_VECTORS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_vectors')
CONFIG = load_config()

# Instancia global de la API de clustering online
clustering_api = OnlineClusteringAPI(feature_vectors_dir=FEATURE_VECTORS_DIR)

@app.errorhandler(APIError)
def handle_api_error(error):
    return jsonify({
        'success': False,
        'error': error.message
    }), error.status_code

@app.route('/', methods=['GET'])
def home():
    """Endpoint de bienvenida"""
    return jsonify({
        'message': 'API de Clustering - Proyecto de Análisis de Imágenes',
        'version': '2.0',
        'features': [
            'Clustering Online con Similitud Coseno',
            'Actualización Dinámica de Hiperparámetros', 
            'Múltiples Modelos de Características',
            'Métricas de Evaluación Completas',
            'Soporte para Clases Desbalanceadas'
        ],
        'endpoints': {
            'GET /': 'Esta información',
            'GET /config': 'Configuración e hiperparámetros disponibles',
            'GET /modelos': 'Lista de modelos disponibles',
            'GET /modelo/<nombre>': 'Resultados de un modelo específico',
            'GET /modelo/<nombre>/recomendaciones': 'Recomendaciones para un modelo',
            'GET /resultados': 'Resumen de todos los resultados',
            'GET /experimentos': 'Historial de experimentos',
            'POST /clustering': 'Ejecutar clustering con modelo específico',
            '--- CLUSTERING ONLINE ---': '---',
            'GET /api/clustering/config': 'Configuración actual de clustering online',
            'PUT /api/clustering/config': 'Actualizar hiperparámetros dinámicamente',
            'POST /api/clustering/execute': 'Ejecutar clustering online',
            'POST /api/clustering/batch': 'Ejecutar múltiples experimentos de clustering',
            'GET /api/clustering/models': 'Listar modelos disponibles',
            'POST /api/clustering/validate': 'Validar parámetros antes de ejecutar'
        },
        'usage_example': {
            'execute_clustering': {
                'url': '/api/clustering/execute',
                'method': 'POST',
                'body': {
                    'model_id': 'momentos_hu',
                    'dataset_num': 1,
                    'use_flexible': True,
                    'hyperparameters': {
                        'k': 3,
                        'm': 50,
                        'cluster_similarity_threshold': 0.75
                    }
                }
            }
        }
    })
            'POST /clustering': 'Ejecutar clustering con parámetros personalizados',
            'POST /evaluar': 'Evaluar modelo con hiperparámetros específicos',
            'POST /experimento': 'Guardar experimento con nombre personalizado'
        },
        'total_models': len(CONFIG['models']['available']),
        'hyperparameters': list(CONFIG['hyperparameters']['clustering'].keys())
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Obtiene la configuración completa de hiperparámetros y modelos"""
    return jsonify({
        'success': True,
        'config': CONFIG
    })

@app.route('/modelo/<modelo_id>/recomendaciones', methods=['GET'])
def get_recommendations(modelo_id):
    """Obtiene recomendaciones específicas para un modelo"""
    try:
        recommendations = get_model_recommendations(modelo_id, CONFIG)
        if not recommendations:
            raise APIError(f'Modelo {modelo_id} no encontrado', 404)
        
        return jsonify({
            'success': True,
            'modelo_id': modelo_id,
            'recomendaciones': recommendations
        })
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(str(e), 500)

@app.route('/experimentos', methods=['GET'])
def get_experiments():
    """Obtiene el historial de experimentos"""
    try:
        experiments = get_experiment_history()
        return jsonify({
            'success': True,
            'experimentos': experiments,
            'total': len(experiments)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/experimento', methods=['POST'])
def save_experiment():
    """Guarda un experimento con nombre personalizado"""
    try:
        data = request.get_json()
        
        required_fields = ['experiment_name', 'parameters', 'results']
        for field in required_fields:
            if field not in data:
                raise APIError(f'Campo requerido faltante: {field}')
        
        experiment_file = save_experiment_results(
            data['parameters'],
            data['results'],
            data['experiment_name']
        )
        
        return jsonify({
            'success': True,
            'message': 'Experimento guardado exitosamente',
            'experiment_file': os.path.basename(experiment_file),
            'timestamp': datetime.now().isoformat()
        })
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(str(e), 500)

@app.route('/modelos', methods=['GET'])
def listar_modelos():
    """Lista todos los modelos de características disponibles organizados por dataset"""
    try:
        modelos = []
        
        # Modelos CSV por dataset
        archivos_csv = [
            ("momentos_clasicos.csv", "Momentos Clásicos", "Momentos regulares, centrales y normalizados"),
            ("momentos_hu.csv", "Momentos de Hu", "7 momentos invariantes de Hu"),
            ("momentos_zernike.csv", "Momentos de Zernike", "Momentos de Zernike para análisis de forma"),
            ("sift_features.csv", "SIFT Features", "Scale-Invariant Feature Transform"),
            ("hog_features.csv", "HOG Features", "Histogram of Oriented Gradients")
        ]
        
        for dataset_num in [1, 2]:
            dataset_dir = os.path.join(FEATURE_VECTORS_DIR, f"dataset_{dataset_num}")
            
            for archivo, nombre, descripcion in archivos_csv:
                filepath = os.path.join(dataset_dir, archivo)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    modelos.append({
                        'id': f"{archivo.replace('.csv', '')}_dataset_{dataset_num}",
                        'nombre': f"{nombre} - Dataset {dataset_num}",
                        'descripcion': descripcion,
                        'tipo': 'CSV',
                        'archivo': archivo,
                        'dataset': dataset_num,
                        'n_muestras': len(df),
                        'n_features': len([col for col in df.columns if col not in ['label', 'clase_name', 'dataset']]),
                        'disponible': True
                    })
            
            # Modelo CNN por dataset
            embeddings_path = os.path.join(dataset_dir, "Embeddings_cnn.npy")
            labels_path = os.path.join(dataset_dir, "Labels_cnn.npy")
            
            if os.path.exists(embeddings_path) and os.path.exists(labels_path):
                embeddings = np.load(embeddings_path)
                modelos.append({
                    'id': f'cnn_embeddings_dataset_{dataset_num}',
                    'nombre': f'CNN Embeddings (ResNet50) - Dataset {dataset_num}',
                    'descripcion': 'Embeddings de red convolucional preentrenada',
                    'tipo': 'NPY',
                    'archivo': 'Embeddings_cnn.npy',
                    'dataset': dataset_num,
                    'n_muestras': embeddings.shape[0],
                    'n_features': embeddings.shape[1],
                    'disponible': True
                })
        
        return jsonify({
            'success': True,
            'modelos': modelos,
            'total': len(modelos)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/modelo/<modelo_id>', methods=['GET'])
def obtener_modelo(modelo_id):
    """Obtiene información detallada de un modelo específico"""
    try:
        # Parsear el modelo_id para extraer dataset
        # Formato: {modelo_name}_dataset_{dataset_num}
        if '_dataset_' not in modelo_id:
            return jsonify({
                'success': False,
                'error': f'ID de modelo inválido: {modelo_id}. Debe incluir dataset.'
            }), 400
        
        parts = modelo_id.rsplit('_dataset_', 1)
        if len(parts) != 2:
            return jsonify({
                'success': False,
                'error': f'ID de modelo inválido: {modelo_id}'
            }), 400
        
        modelo_base = parts[0]
        dataset_num = parts[1]
        
        # Validar dataset
        if dataset_num not in ['1', '2']:
            return jsonify({
                'success': False,
                'error': f'Dataset inválido: {dataset_num}'
            }), 400
        
        dataset_dir = os.path.join(FEATURE_VECTORS_DIR, f"dataset_{dataset_num}")
        
        # Mapeo de IDs base a archivos
        modelos_info = {
            'momentos_clasicos': ('momentos_clasicos.csv', 'Momentos Clásicos'),
            'momentos_hu': ('momentos_hu.csv', 'Momentos de Hu'),
            'momentos_zernike': ('momentos_zernike.csv', 'Momentos de Zernike'),
            'sift_features': ('sift_features.csv', 'SIFT Features'),
            'hog_features': ('hog_features.csv', 'HOG Features'),
            'cnn_embeddings': ('Embeddings_cnn.npy', 'CNN Embeddings')
        }
        
        if modelo_base not in modelos_info:
            return jsonify({
                'success': False,
                'error': f'Modelo base {modelo_base} no encontrado'
            }), 404
        
        archivo, nombre = modelos_info[modelo_base]
        
        if modelo_base == 'cnn_embeddings':
            # Cargar datos CNN
            embeddings_path = os.path.join(dataset_dir, "Embeddings_cnn.npy")
            labels_path = os.path.join(dataset_dir, "Labels_cnn.npy")
            
            if not (os.path.exists(embeddings_path) and os.path.exists(labels_path)):
                return jsonify({
                    'success': False,
                    'error': f'Archivos CNN no encontrados para dataset {dataset_num}'
                }), 404
            
            embeddings = np.load(embeddings_path)
            labels = np.load(labels_path)
            
            # Estadísticas básicas
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            return jsonify({
                'success': True,
                'modelo': {
                    'id': modelo_id,
                    'nombre': f"{nombre} - Dataset {dataset_num}",
                    'tipo': 'NPY',
                    'dataset': int(dataset_num),
                    'n_muestras': embeddings.shape[0],
                    'n_features': embeddings.shape[1],
                    'distribucion_clases': dict(zip(unique_labels.tolist(), counts.tolist())),
                    'estadisticas': {
                        'media': float(np.mean(embeddings)),
                        'std': float(np.std(embeddings)),
                        'min': float(np.min(embeddings)),
                        'max': float(np.max(embeddings))
                    }
                }
            })
        else:
            # Cargar datos CSV
            filepath = os.path.join(dataset_dir, archivo)
            if not os.path.exists(filepath):
                return jsonify({
                    'success': False,
                    'error': f'Archivo {archivo} no encontrado para dataset {dataset_num}'
                }), 404
            
            df = pd.read_csv(filepath)
            feature_cols = [col for col in df.columns if col not in ['label', 'clase_name', 'dataset']]
            
            # Estadísticas básicas
            X = df[feature_cols].values
            y = df['label'].values
            unique_labels, counts = np.unique(y, return_counts=True)
            
            return jsonify({
                'success': True,
                'modelo': {
                    'id': modelo_id,
                    'nombre': f"{nombre} - Dataset {dataset_num}",
                    'tipo': 'CSV',
                    'dataset': int(dataset_num),
                    'n_muestras': len(df),
                    'n_features': len(feature_cols),
                    'feature_names': feature_cols,
                    'distribucion_clases': dict(zip(unique_labels.tolist(), counts.tolist())),
                    'estadisticas': {
                        'media': float(np.mean(X)),
                        'std': float(np.std(X)),
                        'min': float(np.min(X)),
                        'max': float(np.max(X))
                    }
                }
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/resultados', methods=['GET'])
def obtener_resultados():
    """Obtiene resumen de resultados de clustering si existen"""
    try:
        results_path = os.path.join(FEATURE_VECTORS_DIR, "clustering_evaluation_results.csv")
        
        if not os.path.exists(results_path):
            return jsonify({
                'success': False,
                'message': 'No hay resultados de clustering disponibles. Ejecuta model.py primero.'
            })
        
        df_resultados = pd.read_csv(results_path)
        
        # Convertir a formato JSON
        resultados = []
        for _, row in df_resultados.iterrows():
            resultados.append({
                'metodo': row['Método'],
                'muestras': int(row['Muestras']),
                'features': int(row['Features']),
                'metricas': {
                    'ari': float(row['ARI']),
                    'nmi': float(row['NMI']),
                    'ami': float(row['AMI']),
                    'silhouette': float(row['Silhouette']),
                    'davies_bouldin': float(row['Davies-Bouldin']),
                    'calinski_harabasz': float(row['C-Harabasz'])
                }
            })
        
        # Encontrar el mejor modelo para cada métrica
        df_num = df_resultados.copy()
        for col in ['ARI', 'NMI', 'AMI', 'Silhouette', 'C-Harabasz']:
            df_num[col] = pd.to_numeric(df_num[col])
        df_num['Davies-Bouldin'] = pd.to_numeric(df_num['Davies-Bouldin'])
        
        mejores = {
            'mejor_ari': df_num.loc[df_num['ARI'].idxmax(), 'Método'],
            'mejor_nmi': df_num.loc[df_num['NMI'].idxmax(), 'Método'],
            'mejor_silhouette': df_num.loc[df_num['Silhouette'].idxmax(), 'Método'],
            'mejor_davies_bouldin': df_num.loc[df_num['Davies-Bouldin'].idxmin(), 'Método'],  # Menor es mejor
            'mejor_calinski_harabasz': df_num.loc[df_num['C-Harabasz'].idxmax(), 'Método']
        }
        
        return jsonify({
            'success': True,
            'resultados': resultados,
            'mejores_modelos': mejores,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(results_path)).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clustering', methods=['POST'])
def ejecutar_clustering():
    """Ejecuta clustering con hiperparámetros personalizados"""
    try:
        data = request.get_json()
        
        # Validar parámetros requeridos
        if 'modelo_id' not in data:
            raise APIError('Parámetro requerido faltante: modelo_id')
        
        modelo_id = data['modelo_id']
        
        # Extraer hiperparámetros
        hyperparams = {}
        for param in CONFIG['hyperparameters']['clustering'].keys():
            if param in data:
                hyperparams[param] = data[param]
        
        # Validar hiperparámetros
        errors = validate_hyperparameters(hyperparams, CONFIG)
        if errors:
            raise APIError(f'Errores de validación: {"; ".join(errors)}')
        
        # Valores por defecto
        k = hyperparams.get('k', CONFIG['hyperparameters']['clustering']['k']['default'])
        m = data.get('m', None)  # Se calculará automáticamente si no se proporciona
        random_state = hyperparams.get('random_state', CONFIG['hyperparameters']['clustering']['random_state']['default'])
        cluster_similarity_threshold = hyperparams.get('cluster_similarity_threshold', CONFIG['hyperparameters']['clustering']['cluster_similarity_threshold']['default'])
        subcluster_similarity_threshold = hyperparams.get('subcluster_similarity_threshold', CONFIG['hyperparameters']['clustering']['subcluster_similarity_threshold']['default'])
        pair_similarity_maximum = hyperparams.get('pair_similarity_maximum', CONFIG['hyperparameters']['clustering']['pair_similarity_maximum']['default'])
        
        # Cargar datos del modelo
        X, y, n_features = cargar_datos_modelo(modelo_id)
        
        if X is None:
            raise APIError(f'No se pudo cargar el modelo {modelo_id}', 404)
        
        # Preprocesamiento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_norm = normalize(X_scaled, norm='l2')
        
        # Calcular m si no se proporcionó
        if m is None:
            m = len(X_norm) // k
        
        # Ajustar datos para n = k*m
        n_adjusted = k * m
        if len(X_norm) > n_adjusted:
            X_norm = X_norm[:n_adjusted]
            y = y[:n_adjusted]
        elif len(X_norm) < n_adjusted:
            raise APIError(f'No hay suficientes muestras. Necesitas al menos {n_adjusted} muestras para k={k}, m={m}')
        
        # Ejecutar clustering
        resultado = online_capacity_links_with_metrics(
            X_norm,
            y_true=y,
            k=k,
            m=m,
            shuffle_data=True,
            random_state=random_state,
            cluster_similarity_threshold=cluster_similarity_threshold,
            subcluster_similarity_threshold=subcluster_similarity_threshold,
            pair_similarity_maximum=pair_similarity_maximum,
        )
        
        # Métricas internas
        metricas_internas = _safe_internal_metrics(X_norm, resultado.labels)
        
        # Formatear métricas usando la utilidad
        metrics_formatted = format_metrics_response(resultado, metricas_internas)
        
        # Obtener recomendaciones del modelo
        recommendations = get_model_recommendations(modelo_id, CONFIG)
        
        # Formatear respuesta
        response = {
            'success': True,
            'modelo': {
                'id': modelo_id,
                'n_muestras_usadas': len(X_norm),
                'n_features': n_features,
                'recomendaciones': recommendations
            },
            'parametros': {
                'k': k,
                'm': m,
                'random_state': random_state,
                'cluster_similarity_threshold': cluster_similarity_threshold,
                'subcluster_similarity_threshold': subcluster_similarity_threshold,
                'pair_similarity_maximum': pair_similarity_maximum
            },
            'metricas': metrics_formatted,
            'clusters': {
                'distribucion': dict(resultado.counts),
                'total_clusters': len(resultado.counts),
                'balanceado': len(set(resultado.counts.values())) == 1
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(str(e), 500)

@app.route('/evaluar', methods=['POST'])
def evaluar_todos():
    """Evalúa todos los modelos con hiperparámetros específicos"""
    try:
        data = request.get_json() or {}
        
        # Hiperparámetros con valores por defecto
        k = data.get('k', 3)
        random_state = data.get('random_state', 42)
        cluster_similarity_threshold = data.get('cluster_similarity_threshold', 0.75)
        subcluster_similarity_threshold = data.get('subcluster_similarity_threshold', 0.85)
        pair_similarity_maximum = data.get('pair_similarity_maximum', 0.95)
        
        resultados = []
        
        # Lista de modelos a evaluar
        modelos = [
            ('momentos_clasicos', 'Momentos Clásicos'),
            ('momentos_hu', 'Momentos de Hu'),
            ('momentos_zernike', 'Momentos de Zernike'),
            ('sift_features', 'SIFT Features'),
            ('hog_features', 'HOG Features'),
            ('cnn_embeddings', 'CNN Embeddings (ResNet50)')
        ]
        
        for modelo_id, nombre in modelos:
            try:
                X, y, n_features = cargar_datos_modelo(modelo_id)
                
                if X is None:
                    continue
                
                # Preprocesamiento
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_norm = normalize(X_scaled, norm='l2')
                
                # Ajustar k para CNN (6 clases) vs otros (3 clases)
                k_modelo = 6 if modelo_id == 'cnn_embeddings' else k
                m = len(X_norm) // k_modelo
                n_adjusted = k_modelo * m
                
                if len(X_norm) >= n_adjusted:
                    X_norm = X_norm[:n_adjusted]
                    y = y[:n_adjusted]
                    
                    # Ejecutar clustering
                    resultado = online_capacity_links_with_metrics(
                        X_norm,
                        y_true=y,
                        k=k_modelo,
                        m=m,
                        shuffle_data=True,
                        random_state=random_state,
                        cluster_similarity_threshold=cluster_similarity_threshold,
                        subcluster_similarity_threshold=subcluster_similarity_threshold,
                        pair_similarity_maximum=pair_similarity_maximum,
                    )
                    
                    # Métricas internas
                    metricas_internas = _safe_internal_metrics(X_norm, resultado.labels)
                    
                    resultados.append({
                        'modelo_id': modelo_id,
                        'nombre': nombre,
                        'n_muestras': len(X_norm),
                        'n_features': n_features,
                        'k_usado': k_modelo,
                        'metricas': {
                            'ari': float(resultado.ari),
                            'nmi': float(resultado.nmi),
                            'ami': float(resultado.ami),
                            'silhouette': float(metricas_internas['silhouette_cosine']),
                            'davies_bouldin': float(metricas_internas['davies_bouldin']),
                            'calinski_harabasz': float(metricas_internas['calinski_harabasz'])
                        }
                    })
                
            except Exception as e:
                resultados.append({
                    'modelo_id': modelo_id,
                    'nombre': nombre,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'parametros_usados': {
                'k': k,
                'random_state': random_state,
                'cluster_similarity_threshold': cluster_similarity_threshold,
                'subcluster_similarity_threshold': subcluster_similarity_threshold,
                'pair_similarity_maximum': pair_similarity_maximum
            },
            'resultados': resultados,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def cargar_datos_modelo(modelo_id):
    """Función auxiliar para cargar datos de un modelo específico usando la nueva estructura por dataset"""
    try:
        # Parsear el modelo_id para extraer dataset
        if '_dataset_' not in modelo_id:
            raise ValueError(f"ID de modelo inválido: {modelo_id}. Debe incluir dataset.")
        
        parts = modelo_id.rsplit('_dataset_', 1)
        if len(parts) != 2:
            raise ValueError(f"ID de modelo inválido: {modelo_id}")
        
        modelo_base = parts[0]
        dataset_num = parts[1]
        
        # Validar dataset
        if dataset_num not in ['1', '2']:
            raise ValueError(f"Dataset inválido: {dataset_num}")
        
        # Usar la nueva función para cargar por dataset
        from model import cargar_modelo_por_dataset
        return cargar_modelo_por_dataset(modelo_id, int(dataset_num), FEATURE_VECTORS_DIR)
        
    except Exception as e:
        print(f"Error cargando modelo {modelo_id}: {e}")
        return None, None, None


# ============================================================================
# ENDPOINTS PARA CLUSTERING ONLINE
# ============================================================================

@app.route('/api/clustering/config', methods=['GET'])
def get_clustering_config():
    """Obtiene la configuración actual de clustering online"""
    try:
        return jsonify({
            'success': True,
            'hyperparameters': clustering_api.get_hyperparameters(),
            'available_models': clustering_api.list_available_models(),
            'datasets': [1, 2]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clustering/config', methods=['PUT'])
def update_clustering_config():
    """Actualiza hiperparámetros de clustering dinámicamente"""
    try:
        data = request.get_json()
        if not data:
            raise APIError("No se proporcionaron datos", 400)
        
        # Actualizar hiperparámetros
        clustering_api.update_hyperparameters(**data)
        
        return jsonify({
            'success': True,
            'message': 'Hiperparámetros actualizados correctamente',
            'updated_parameters': clustering_api.get_hyperparameters()
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Error de validación: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clustering/execute', methods=['POST'])
def execute_clustering():
    """Ejecuta clustering online con parámetros especificados"""
    try:
        data = request.get_json()
        if not data:
            raise APIError("No se proporcionaron datos", 400)
        
        # Validar parámetros requeridos
        required_params = ['model_id', 'dataset_num']
        for param in required_params:
            if param not in data:
                raise APIError(f"Parámetro requerido faltante: {param}", 400)
        
        model_id = data['model_id']
        dataset_num = int(data['dataset_num'])
        use_flexible = data.get('use_flexible', True)  # Por defecto usar versión flexible
        
        # Actualizar hiperparámetros si se proporcionan
        if 'hyperparameters' in data:
            clustering_api.update_hyperparameters(**data['hyperparameters'])
        
        # Ejecutar clustering
        result = clustering_api.cluster(
            model_id=model_id,
            dataset_num=dataset_num, 
            use_flexible=use_flexible
        )
        
        # Guardar resultados automáticamente
        if result['success']:
            output_dir = "resultados"
            os.makedirs(output_dir, exist_ok=True)
            clustering_api.save_results(result, output_dir)
            
            # Agregar timestamp
            result['timestamp'] = datetime.now().isoformat()
            
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Error de validación: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clustering/batch', methods=['POST'])
def execute_batch_clustering():
    """Ejecuta clustering en múltiples configuraciones (comparación de modelos)"""
    try:
        data = request.get_json()
        if not data:
            raise APIError("No se proporcionaron datos", 400)
        
        experiments = data.get('experiments', [])
        if not experiments:
            raise APIError("Se requiere al menos un experimento", 400)
        
        results = []
        
        for i, experiment in enumerate(experiments):
            try:
                # Validar experimento
                if 'model_id' not in experiment or 'dataset_num' not in experiment:
                    results.append({
                        'experiment_id': i,
                        'success': False,
                        'error': 'Faltan parámetros requeridos (model_id, dataset_num)'
                    })
                    continue
                
                # Actualizar hiperparámetros si están especificados
                if 'hyperparameters' in experiment:
                    clustering_api.update_hyperparameters(**experiment['hyperparameters'])
                
                # Ejecutar clustering
                result = clustering_api.cluster(
                    model_id=experiment['model_id'],
                    dataset_num=int(experiment['dataset_num']),
                    use_flexible=experiment.get('use_flexible', True)
                )
                
                # Agregar ID del experimento
                result['experiment_id'] = i
                result['experiment_name'] = experiment.get('name', f"Experimento {i+1}")
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'experiment_id': i,
                    'success': False,
                    'error': str(e)
                })
        
        # Crear resumen comparativo
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            # Ordenar por ARI descendente
            successful_results.sort(
                key=lambda x: x.get('metrics', {}).get('external', {}).get('ARI', 0),
                reverse=True
            )
            
            summary = {
                'total_experiments': len(experiments),
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results),
                'best_result': successful_results[0] if successful_results else None,
                'ranking': [
                    {
                        'rank': i+1,
                        'experiment_name': r.get('experiment_name'),
                        'model': r.get('model', {}).get('name'),
                        'ari': r.get('metrics', {}).get('external', {}).get('ARI', 0),
                        'nmi': r.get('metrics', {}).get('external', {}).get('NMI', 0),
                        'silhouette': r.get('metrics', {}).get('internal', {}).get('Silhouette', 0)
                    }
                    for i, r in enumerate(successful_results[:10])  # Top 10
                ]
            }
        else:
            summary = {
                'total_experiments': len(experiments),
                'successful': 0,
                'failed': len(results),
                'best_result': None,
                'ranking': []
            }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clustering/models', methods=['GET'])
def list_clustering_models():
    """Lista todos los modelos disponibles para clustering"""
    try:
        return jsonify({
            'success': True,
            'models': clustering_api.list_available_models(),
            'total_models': len(clustering_api.available_models)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clustering/validate', methods=['POST'])
def validate_clustering_parameters():
    """Valida parámetros de clustering sin ejecutar el algoritmo"""
    try:
        data = request.get_json()
        if not data:
            raise APIError("No se proporcionaron datos", 400)
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validar modelo
        model_id = data.get('model_id')
        if model_id and model_id not in clustering_api.available_models:
            validation_results['errors'].append(f"Modelo '{model_id}' no disponible")
            validation_results['valid'] = False
        
        # Validar dataset
        dataset_num = data.get('dataset_num')
        if dataset_num and dataset_num not in [1, 2]:
            validation_results['errors'].append(f"Dataset {dataset_num} no válido (debe ser 1 o 2)")
            validation_results['valid'] = False
        
        # Validar hiperparámetros
        if 'hyperparameters' in data:
            try:
                # Crear copia temporal de la API para validación
                temp_api = OnlineClusteringAPI()
                temp_api.update_hyperparameters(**data['hyperparameters'])
            except ValueError as e:
                validation_results['errors'].append(f"Hiperparámetros inválidos: {str(e)}")
                validation_results['valid'] = False
        
        # Advertencias sobre configuración
        k = data.get('hyperparameters', {}).get('k', clustering_api.hyperparameters['k'])
        m = data.get('hyperparameters', {}).get('m', clustering_api.hyperparameters['m'])
        
        if k > 5:
            validation_results['warnings'].append("Un k alto puede reducir la calidad del clustering")
        if m < 10:
            validation_results['warnings'].append("Un m bajo puede crear clusters muy pequeños")
        
        return jsonify({
            'success': True,
            'validation': validation_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Iniciando API de Clustering...")
    print(f"📁 Directorio de características: {FEATURE_VECTORS_DIR}")
    print("🔗 Nuevos endpoints de clustering online disponibles:")
    print("   GET  /api/clustering/config - Configuración actual")
    print("   PUT  /api/clustering/config - Actualizar hiperparámetros")
    print("   POST /api/clustering/execute - Ejecutar clustering")
    print("   POST /api/clustering/batch - Ejecutar múltiples experimentos")
    print("   GET  /api/clustering/models - Listar modelos")
    print("   POST /api/clustering/validate - Validar parámetros")
    app.run(debug=True, host='0.0.0.0', port=5000)