import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from collections import Counter
import json

# Importar funciones de clustering online
try:
    from clustering.online import (
        online_capacity_links_with_metrics,
        online_flexible_links_with_metrics,
        online_custom_capacity_flexible_links_with_metrics,
        OnlineBalancedLinksResult,
        _safe_internal_metrics
    )
except ImportError:
    # Si falla la importación relativa, intentar importación desde el directorio actual
    from .online import (
        online_capacity_links_with_metrics,
        online_flexible_links_with_metrics,
        online_custom_capacity_flexible_links_with_metrics,
        OnlineBalancedLinksResult,
        _safe_internal_metrics
    )

def convert_numpy_types(obj):
    """
    Convierte tipos numpy a tipos nativos de Python para serialización JSON
    """
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class DatasetEvaluator:
    """
    Clase para evaluar un dataset específico con clustering online
    """
    def __init__(self, dataset_num: int, feature_vectors_dir: str = "src/feature_vectors"):
        self.dataset_num = dataset_num
        self.feature_vectors_dir = feature_vectors_dir
        self.dataset_dir = os.path.join(feature_vectors_dir, f"dataset_{dataset_num}")
        self.results = {}
        
    def _cargar_csv_features(self, filename: str):
        """Carga un archivo CSV de características"""
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠ No se encontró: {filepath}")
            return None, None
            
        df = pd.read_csv(filepath)
        
        # Separar features de labels
        if 'label' in df.columns:
            labels = df['label'].values
            features = df.drop(['label', 'clase_name', 'dataset'], axis=1, errors='ignore').values
        elif 'Clase' in df.columns:
            labels = df['Clase'].values
            features = df.drop(['Clase'], axis=1, errors='ignore').values
        else:
            print(f"⚠ No se encontró columna de etiquetas en {filename}")
            return None, None
            
        return features, labels
    
    def _cargar_npy_embeddings(self):
        """Carga embeddings CNN desde archivos .npy"""
        embeddings_path = os.path.join(self.dataset_dir, "Embeddings_cnn.npy")
        labels_path = os.path.join(self.dataset_dir, "Labels_cnn.npy")
        
        if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
            print(f"⚠ No se encontraron embeddings CNN para dataset {self.dataset_num}")
            return None, None
            
        embeddings = np.load(embeddings_path)
        labels_raw = np.load(labels_path)
        
        # Convertir labels string a numéricos (d1_c0 -> 0, d1_c1 -> 1, etc.)
        labels = np.array([int(label.split('_')[1][1]) for label in labels_raw])
        
        return embeddings, labels
    
    def _preprocesar_datos(self, X: np.ndarray, normalize_data: bool = True, apply_pca: bool = False, n_components: int = 50):
        """Preprocesa los datos para clustering"""
        # Normalización estándar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Normalización unitaria si se solicita
        if normalize_data:
            X_scaled = normalize(X_scaled, norm='l2')
        
        # PCA si se solicita
        if apply_pca and X_scaled.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
            print(f"   PCA aplicado: {X.shape[1]} -> {X_scaled.shape[1]} dimensiones")
            print(f"   Varianza explicada: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_scaled
    
    def evaluar_caracteristicas_csv(self, k: int = 3, m_factor: int = 1, 
                                   normalize_data: bool = True, apply_pca: bool = False):
        """
        Evalúa todas las características CSV disponibles
        
        Parameters:
        -----------
        k : int
            Número de clusters
        m_factor : int 
            Factor para calcular m (tamaño por cluster). m = (n // k) * m_factor
        normalize_data : bool
            Si aplicar normalización unitaria
        apply_pca : bool
            Si aplicar PCA para reducir dimensionalidad
        """
        print(f"\n{'='*60}")
        print(f"EVALUANDO CARACTERÍSTICAS CSV - DATASET {self.dataset_num}")
        print(f"{'='*60}")
        
        archivos_csv = [
            ("momentos_clasicos.csv", "Momentos Clásicos"),
            ("momentos_hu.csv", "Momentos de Hu"),
            ("momentos_zernike.csv", "Momentos de Zernike"),
            ("sift_features.csv", "SIFT Features"),
            ("hog_features.csv", "HOG Features")
        ]
        
        for filename, description in archivos_csv:
            print(f"\n--- {description} ---")
            
            X, y = self._cargar_csv_features(filename)
            if X is None or y is None:
                continue
                
            print(f"   Datos cargados: {X.shape} | Clases: {np.unique(y)}")
            
            # Verificar si n es múltiplo de k
            n = len(X)
            if n % k != 0:
                # Ajustar n para que sea múltiplo de k
                n_adjusted = (n // k) * k
                indices = np.random.choice(n, n_adjusted, replace=False)
                X = X[indices]
                y = y[indices]
                print(f"   Ajustado a: {X.shape} para que n sea múltiplo de k")
            
            m = len(X) // k
            
            # Preprocesar datos
            X_processed = self._preprocesar_datos(X, normalize_data=normalize_data, apply_pca=apply_pca)
            
            try:
                # Aplicar clustering online
                result = online_capacity_links_with_metrics(
                    X=X_processed,
                    y_true=y,
                    k=k,
                    m=m,
                    shuffle_data=True,
                    random_state=42
                )
                
                # Métricas internas
                internal_metrics = _safe_internal_metrics(X_processed, result.labels)
                
                # Guardar resultados
                resultado = {
                    'description': description,
                    'data_shape': X.shape,
                    'processed_shape': X_processed.shape,
                    'n_clusters': k,
                    'cluster_size': m,
                    'counts': dict(result.counts),
                    'external_metrics': {
                        'nmi': float(result.nmi),
                        'ami': float(result.ami),
                        'ari': float(result.ari)
                    },
                    'internal_metrics': internal_metrics
                }
                
                self.results[filename] = resultado
                
                # Mostrar resultados
                print(f"   Clusters formados: {dict(result.counts)}")
                print(f"   NMI: {result.nmi:.4f} | AMI: {result.ami:.4f} | ARI: {result.ari:.4f}")
                print(f"   Silhouette: {internal_metrics['silhouette_cosine']:.4f}")
                
            except Exception as e:
                print(f"   Error en clustering: {e}")
                continue
    
    def evaluar_embeddings_cnn(self, k: int = 3, m_factor: int = 1):
        """
        Evalúa embeddings CNN
        """
        print(f"\n{'='*60}")
        print(f"EVALUANDO EMBEDDINGS CNN - DATASET {self.dataset_num}")
        print(f"{'='*60}")
        
        X, y = self._cargar_npy_embeddings()
        if X is None or y is None:
            return
            
        print(f"   Embeddings cargados: {X.shape} | Clases: {np.unique(y)}")
        
        # Verificar si n es múltiplo de k
        n = len(X)
        if n % k != 0:
            n_adjusted = (n // k) * k
            indices = np.random.choice(n, n_adjusted, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"   Ajustado a: {X.shape} para que n sea múltiplo de k")
        
        m = len(X) // k
        
        # Preprocesar (los embeddings CNN ya están en buen formato)
        X_processed = self._preprocesar_datos(X, normalize_data=True, apply_pca=False)
        
        try:
            result = online_capacity_links_with_metrics(
                X=X_processed,
                y_true=y,
                k=k,
                m=m,
                shuffle_data=True,
                random_state=42
            )
            
            internal_metrics = _safe_internal_metrics(X_processed, result.labels)
            
            resultado = {
                'description': 'CNN Embeddings (ResNet50)',
                'data_shape': X.shape,
                'processed_shape': X_processed.shape,
                'n_clusters': k,
                'cluster_size': m,
                'counts': dict(result.counts),
                'external_metrics': {
                    'nmi': float(result.nmi),
                    'ami': float(result.ami),
                    'ari': float(result.ari)
                },
                'internal_metrics': internal_metrics
            }
            
            self.results['embeddings_cnn'] = resultado
            
            print(f"   Clusters formados: {dict(result.counts)}")
            print(f"   NMI: {result.nmi:.4f} | AMI: {result.ami:.4f} | ARI: {result.ari:.4f}")
            print(f"   Silhouette: {internal_metrics['silhouette_cosine']:.4f}")
            
        except Exception as e:
            print(f"   Error en clustering CNN: {e}")
    
    def guardar_resultados(self, output_dir: str = "resultados"):
        """Guarda los resultados en un archivo JSON"""
        if not self.results:
            print("⚠ No hay resultados para guardar")
            return
            
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
            
        output_path = os.path.join(output_dir, f"clustering_results_dataset_{self.dataset_num}.json")
        
        # Convertir tipos numpy a tipos nativos de Python
        results_converted = convert_numpy_types(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ Resultados guardados en: {output_path}")
    
    def mostrar_resumen(self):
        """Muestra un resumen de todos los resultados"""
        if not self.results:
            print("⚠ No hay resultados disponibles")
            return
            
        print(f"\n{'='*60}")
        print(f"RESUMEN DE RESULTADOS - DATASET {self.dataset_num}")
        print(f"{'='*60}")
        
        # Ordenar por NMI (mejor primero)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('external_metrics', {}).get('nmi', 0),
            reverse=True
        )
        
        print(f"{'Método':<25} {'NMI':<8} {'AMI':<8} {'ARI':<8} {'Silhouette':<12}")
        print("-" * 65)
        
        for key, result in sorted_results:
            desc = result.get('description', key)[:24]
            ext_metrics = result.get('external_metrics', {})
            int_metrics = result.get('internal_metrics', {})
            
            nmi = ext_metrics.get('nmi', 0)
            ami = ext_metrics.get('ami', 0)
            ari = ext_metrics.get('ari', 0)
            sil = int_metrics.get('silhouette_cosine', 0)
            
            print(f"{desc:<25} {nmi:<8.4f} {ami:<8.4f} {ari:<8.4f} {sil:<12.4f}")


def evaluar_todos_los_datasets(k: int = 3, feature_vectors_dir: str = "src/feature_vectors", 
                              resultados_dir: str = "resultados"):
    """
    Función principal para evaluar todos los datasets con clustering online
    
    Parameters:
    -----------
    k : int
        Número de clusters (debe ser 3 para los datasets actuales)
    feature_vectors_dir : str
        Directorio donde están los feature vectors
    resultados_dir : str
        Directorio donde guardar los resultados de evaluación
    """
    print(f"\n{'='*80}")
    print(f"EVALUACIÓN COMPLETA DE CLUSTERING ONLINE")
    print(f"Clusters objetivo: {k} | Features: {feature_vectors_dir} | Resultados: {resultados_dir}")
    print(f"{'='*80}")
    
    resultados_generales = {}
    
    for dataset_num in [1, 2]:
        print(f"\n🔄 Procesando Dataset {dataset_num}...")
        
        evaluator = DatasetEvaluator(dataset_num, feature_vectors_dir)
        
        # Evaluar características CSV
        evaluator.evaluar_caracteristicas_csv(
            k=k, 
            normalize_data=True, 
            apply_pca=True  # Aplicar PCA para características muy dimensionales
        )
        
        # Evaluar embeddings CNN
        evaluator.evaluar_embeddings_cnn(k=k)
        
        # Mostrar resumen
        evaluator.mostrar_resumen()
        
        # Guardar resultados
        evaluator.guardar_resultados(resultados_dir)
        
        resultados_generales[f"dataset_{dataset_num}"] = evaluator.results
    
    # Crear resumen global
    crear_resumen_global(resultados_generales, resultados_dir)
    
    print(f"\n{'='*80}")
    print(f"EVALUACIÓN COMPLETADA ✅")
    print(f"Resultados guardados en: {resultados_dir}")
    print(f"{'='*80}")
    

def crear_resumen_global(resultados: dict, output_dir: str):
    """Crea un resumen comparativo global de todos los datasets"""
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"RESUMEN GLOBAL COMPARATIVO")
    print(f"{'='*80}")
    
    # Crear tabla comparativa
    all_results = []
    
    for dataset_key, dataset_results in resultados.items():
        dataset_num = dataset_key.split('_')[1]
        
        for method_key, method_result in dataset_results.items():
            all_results.append({
                'dataset': f"Dataset {dataset_num}",
                'metodo': method_result.get('description', method_key),
                'nmi': method_result.get('external_metrics', {}).get('nmi', 0),
                'ami': method_result.get('external_metrics', {}).get('ami', 0),
                'ari': method_result.get('external_metrics', {}).get('ari', 0),
                'silhouette': method_result.get('internal_metrics', {}).get('silhouette_cosine', 0)
            })
    
    # Ordenar por NMI
    all_results.sort(key=lambda x: x['nmi'], reverse=True)
    
    print(f"{'Dataset':<12} {'Método':<25} {'NMI':<8} {'AMI':<8} {'ARI':<8} {'Silhouette':<12}")
    print("-" * 85)
    
    for result in all_results:
        print(f"{result['dataset']:<12} {result['metodo'][:24]:<25} "
              f"{result['nmi']:<8.4f} {result['ami']:<8.4f} {result['ari']:<8.4f} {result['silhouette']:<12.4f}")
    
    # Guardar resumen global
    output_path = os.path.join(output_dir, "clustering_evaluation_results.csv")
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✅ Resumen global guardado en: {output_path}")
    print(f"📁 Todos los resultados de clustering disponibles en: {output_dir}/")


class OnlineClusteringAPI:
    """
    API completa para clustering online con parámetros dinámicos
    """
    def __init__(self, feature_vectors_dir: str = "src/feature_vectors"):
        self.feature_vectors_dir = feature_vectors_dir
        self.hyperparameters = {
            'k': 3,
            'm': 50,
            'cluster_similarity_threshold': 0.75,
            'subcluster_similarity_threshold': 0.85,
            'pair_similarity_maximum': 0.95,
            'random_state': 42,
            'normalize_data': True,
            'apply_pca': False,
            'pca_components': 50
        }
        self.available_models = {
            'momentos_clasicos': {
                'name': 'Momentos Clásicos',
                'file': 'momentos_clasicos.csv',
                'description': 'Momentos regulares, centrales y normalizados'
            },
            'momentos_hu': {
                'name': 'Momentos de Hu',
                'file': 'momentos_hu.csv', 
                'description': '7 momentos invariantes de Hu'
            },
            'momentos_zernike': {
                'name': 'Momentos de Zernike',
                'file': 'momentos_zernike.csv',
                'description': 'Momentos de Zernike para análisis de forma'
            },
            'sift_features': {
                'name': 'SIFT Features',
                'file': 'sift_features.csv',
                'description': 'Scale-Invariant Feature Transform'
            },
            'hog_features': {
                'name': 'HOG Features', 
                'file': 'hog_features.csv',
                'description': 'Histogram of Oriented Gradients'
            },
            'cnn_embeddings': {
                'name': 'CNN Embeddings (ResNet50)',
                'file': 'Embeddings_cnn.npy',
                'description': 'Embeddings de red convolucional preentrenada'
            }
        }
        
    def update_hyperparameters(self, **kwargs):
        """
        Actualiza hiperparámetros dinámicamente
        
        Parámetros soportados:
        - k: número de clusters (int)
        - m: capacidad máxima por cluster (int) 
        - cluster_similarity_threshold: umbral similitud clusters (float)
        - subcluster_similarity_threshold: umbral similitud subclusters (float)
        - pair_similarity_maximum: similitud máxima entre pares (float)
        - random_state: semilla aleatoria (int)
        - normalize_data: aplicar normalización unitaria (bool)
        - apply_pca: aplicar PCA (bool)
        - pca_components: componentes PCA (int)
        """
        for key, value in kwargs.items():
            if key in self.hyperparameters:
                # Validaciones básicas
                if key in ['k', 'm', 'random_state', 'pca_components'] and not isinstance(value, int):
                    raise ValueError(f"{key} debe ser un entero")
                if key in ['cluster_similarity_threshold', 'subcluster_similarity_threshold', 'pair_similarity_maximum']:
                    if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                        raise ValueError(f"{key} debe ser un float entre 0 y 1")
                if key in ['k', 'm'] and value <= 0:
                    raise ValueError(f"{key} debe ser mayor que 0")
                    
                self.hyperparameters[key] = value
                print(f"✅ Actualizado {key} = {value}")
            else:
                print(f"⚠ Hiperparámetro '{key}' no reconocido. Disponibles: {list(self.hyperparameters.keys())}")
                
    def get_hyperparameters(self):
        """Retorna los hiperparámetros actuales"""
        return self.hyperparameters.copy()
        
    def list_available_models(self):
        """Lista los modelos de características disponibles"""
        return {k: {'name': v['name'], 'description': v['description']} 
                for k, v in self.available_models.items()}
                
    def _load_data(self, model_id: str, dataset_num: int):
        """Carga datos según el modelo y dataset especificados"""
        if model_id not in self.available_models:
            raise ValueError(f"Modelo '{model_id}' no disponible. Modelos: {list(self.available_models.keys())}")
            
        model_info = self.available_models[model_id]
        dataset_dir = os.path.join(self.feature_vectors_dir, f"dataset_{dataset_num}")
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset {dataset_num} no encontrado en {dataset_dir}")
            
        if model_id == 'cnn_embeddings':
            # Cargar embeddings CNN desde archivos .npy
            embeddings_path = os.path.join(dataset_dir, "Embeddings_cnn.npy")
            labels_path = os.path.join(dataset_dir, "Labels_cnn.npy")
            
            if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
                raise FileNotFoundError(f"Embeddings CNN no encontrados para dataset {dataset_num}")
                
            X = np.load(embeddings_path)
            labels_raw = np.load(labels_path)
            # Convertir labels string a numéricos
            y = np.array([int(label.split('_')[1][1]) for label in labels_raw])
            
        else:
            # Cargar características CSV
            filepath = os.path.join(dataset_dir, model_info['file'])
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Archivo {model_info['file']} no encontrado en dataset {dataset_num}")
                
            df = pd.read_csv(filepath)
            
            # Separar features de labels
            if 'label' in df.columns:
                y = df['label'].values
                X = df.drop(['label', 'clase_name', 'dataset'], axis=1, errors='ignore').values
            elif 'Clase' in df.columns:
                y = df['Clase'].values
                X = df.drop(['Clase'], axis=1, errors='ignore').values
            else:
                raise ValueError(f"No se encontró columna de etiquetas en {model_info['file']}")
                
        return X, y, model_info
        
    def _preprocess_data(self, X: np.ndarray):
        """Preprocesa los datos según la configuración actual"""
        # Normalización estándar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Normalización unitaria si se solicita
        if self.hyperparameters['normalize_data']:
            X_scaled = normalize(X_scaled, norm='l2')
            
        # PCA si se solicita
        if self.hyperparameters['apply_pca'] and X_scaled.shape[1] > self.hyperparameters['pca_components']:
            pca = PCA(n_components=self.hyperparameters['pca_components'], random_state=self.hyperparameters['random_state'])
            X_scaled = pca.fit_transform(X_scaled)
            return X_scaled, {'pca_applied': True, 'explained_variance': float(pca.explained_variance_ratio_.sum())}
            
        return X_scaled, {'pca_applied': False}
        
    def cluster(self, model_id: str, dataset_num: int, use_flexible: bool = True):
        """
        Ejecuta clustering online con el modelo y dataset especificados
        
        Parameters:
        -----------
        model_id : str
            ID del modelo de características a utilizar
        dataset_num : int
            Número del dataset (1 o 2)
        use_flexible : bool
            Si usar versión flexible (permite clases desbalanceadas) o restrictiva
            
        Returns:
        --------
        dict: Resultados del clustering con métricas completas
        """
        # Cargar datos
        X, y_true, model_info = self._load_data(model_id, dataset_num)
        original_shape = X.shape
        
        # Preprocesar datos
        X_processed, preprocessing_info = self._preprocess_data(X)
        
        # Configurar parámetros
        k = self.hyperparameters['k']
        m = self.hyperparameters['m']
        
        print(f"🔄 Ejecutando clustering online...")
        print(f"   Modelo: {model_info['name']}")
        print(f"   Dataset: {dataset_num}")
        print(f"   Datos: {original_shape} -> {X_processed.shape}")
        print(f"   Parámetros: k={k}, m={m}")
        print(f"   Algoritmo: {'Flexible' if use_flexible else 'Restrictivo'}")
        
        try:
            if use_flexible:
                # Usar versión flexible (permite clases desbalanceadas)
                result = online_flexible_links_with_metrics(
                    X=X_processed,
                    y_true=y_true,
                    k=k,
                    m=m,
                    shuffle_data=True,
                    random_state=self.hyperparameters['random_state'],
                    cluster_similarity_threshold=self.hyperparameters['cluster_similarity_threshold'],
                    subcluster_similarity_threshold=self.hyperparameters['subcluster_similarity_threshold'],
                    pair_similarity_maximum=self.hyperparameters['pair_similarity_maximum']
                )
            else:
                # Usar versión restrictiva (requiere n = k*m)
                n = len(X_processed)
                if n != k * m:
                    # Ajustar datos para que n = k*m
                    n_target = k * m
                    if n < n_target:
                        raise ValueError(f"Dataset muy pequeño: {n} < {n_target} (k*m)")
                    # Tomar muestra aleatoria
                    indices = np.random.choice(n, n_target, replace=False)
                    X_processed = X_processed[indices]
                    y_true = y_true[indices]
                    
                result = online_capacity_links_with_metrics(
                    X=X_processed,
                    y_true=y_true,
                    k=k,
                    m=m,
                    shuffle_data=True,
                    random_state=self.hyperparameters['random_state'],
                    cluster_similarity_threshold=self.hyperparameters['cluster_similarity_threshold'],
                    subcluster_similarity_threshold=self.hyperparameters['subcluster_similarity_threshold'],
                    pair_similarity_maximum=self.hyperparameters['pair_similarity_maximum']
                )
                
            # Calcular métricas internas
            internal_metrics = _safe_internal_metrics(X_processed, result.labels)
            
            # Estructurar respuesta completa
            clustering_result = {
                'success': True,
                'model': {
                    'id': model_id,
                    'name': model_info['name'],
                    'description': model_info['description']
                },
                'dataset': dataset_num,
                'algorithm': 'Online LINKS-like (Flexible)' if use_flexible else 'Online LINKS-like (Restrictive)',
                'data_info': {
                    'original_shape': original_shape,
                    'processed_shape': X_processed.shape,
                    'preprocessing': preprocessing_info,
                    'true_classes': len(np.unique(y_true))
                },
                'hyperparameters': self.hyperparameters.copy(),
                'clustering_results': {
                    'labels': result.labels.tolist(),
                    'cluster_counts': dict(result.counts),
                    'n_clusters_formed': len(result.counts),
                    'cluster_sizes': [result.counts[i] for i in range(len(result.counts))]
                },
                'metrics': {
                    'external': {
                        'ARI': float(result.ari),
                        'NMI': float(result.nmi), 
                        'AMI': float(result.ami)
                    },
                    'internal': {
                        'Silhouette': float(internal_metrics['silhouette_cosine']),
                        'Davies_Bouldin': float(internal_metrics['davies_bouldin']),
                        'Calinski_Harabasz': float(internal_metrics['calinski_harabasz'])
                    }
                }
            }
            
            print(f"✅ Clustering completado")
            print(f"   Clusters formados: {len(result.counts)}")
            print(f"   Distribución: {dict(result.counts)}")
            print(f"   Métricas externas - ARI: {result.ari:.4f}, NMI: {result.nmi:.4f}, AMI: {result.ami:.4f}")
            print(f"   Métricas internas - Silhouette: {internal_metrics['silhouette_cosine']:.4f}")
            
            return clustering_result
            
        except Exception as e:
            print(f"❌ Error en clustering: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': {'id': model_id, 'name': model_info['name']},
                'dataset': dataset_num
            }
            
    def cluster_with_custom_capacities(self, model_id: str, dataset_num: int, cluster_capacities: list[int]):
        """
        Ejecuta clustering online con capacidades personalizadas por cluster
        
        Parameters:
        -----------
        model_id : str
            ID del modelo de características a utilizar
        dataset_num : int
            Número del dataset (1 o 2)
        cluster_capacities : list[int]
            Lista con la capacidad máxima para cada cluster
            Por ejemplo: [40, 30, 30] para 3 clusters con diferentes capacidades
            
        Returns:
        --------
        dict: Resultados del clustering con métricas completas
        """
        # Validar capacidades
        if not cluster_capacities or any(c <= 0 for c in cluster_capacities):
            raise ValueError("Todas las capacidades deben ser > 0")
            
        # Cargar datos
        X, y_true, model_info = self._load_data(model_id, dataset_num)
        original_shape = X.shape
        
        # Preprocesar datos
        X_processed, preprocessing_info = self._preprocess_data(X)
        
        k = len(cluster_capacities)
        total_capacity = sum(cluster_capacities)
        
        print(f"🔄 Ejecutando clustering con capacidades personalizadas...")
        print(f"   Modelo: {model_info['name']}")
        print(f"   Dataset: {dataset_num}")
        print(f"   Datos: {original_shape} -> {X_processed.shape}")
        print(f"   Clusters: {k}")
        print(f"   Capacidades: {cluster_capacities} (total: {total_capacity})")
        
        try:
            # Usar versión con capacidades personalizadas
            result = online_custom_capacity_flexible_links_with_metrics(
                X=X_processed,
                y_true=y_true,
                cluster_capacities=cluster_capacities,
                shuffle_data=True,
                random_state=self.hyperparameters['random_state'],
                cluster_similarity_threshold=self.hyperparameters['cluster_similarity_threshold'],
                subcluster_similarity_threshold=self.hyperparameters['subcluster_similarity_threshold'],
                pair_similarity_maximum=self.hyperparameters['pair_similarity_maximum'],
                allow_overflow=True
            )
                
            # Calcular métricas internas
            internal_metrics = _safe_internal_metrics(X_processed, result.labels)
            
            # Estructurar respuesta completa
            clustering_result = {
                'success': True,
                'model': {
                    'id': model_id,
                    'name': model_info['name'],
                    'description': model_info['description']
                },
                'dataset': dataset_num,
                'algorithm': f'Online LINKS-like (Capacidades Personalizadas: {cluster_capacities})',
                'data_info': {
                    'original_shape': original_shape,
                    'processed_shape': X_processed.shape,
                    'preprocessing': preprocessing_info,
                    'true_classes': len(np.unique(y_true))
                },
                'hyperparameters': {
                    **self.hyperparameters.copy(),
                    'cluster_capacities': cluster_capacities,
                    'total_capacity': total_capacity
                },
                'clustering_results': {
                    'labels': result.labels.tolist(),
                    'cluster_counts': dict(result.counts),
                    'n_clusters_formed': len(result.counts),
                    'cluster_sizes': [result.counts[i] for i in range(len(result.counts))],
                    'target_capacities': cluster_capacities,
                    'capacity_utilization': [
                        f"{result.counts.get(i, 0)}/{cluster_capacities[i]}" 
                        for i in range(len(cluster_capacities))
                    ]
                },
                'metrics': {
                    'external': {
                        'ARI': float(result.ari),
                        'NMI': float(result.nmi), 
                        'AMI': float(result.ami)
                    },
                    'internal': {
                        'Silhouette': float(internal_metrics['silhouette_cosine']),
                        'Davies_Bouldin': float(internal_metrics['davies_bouldin']),
                        'Calinski_Harabasz': float(internal_metrics['calinski_harabasz'])
                    }
                }
            }
            
            print(f"✅ Clustering con capacidades personalizadas completado")
            print(f"   Clusters formados: {len(result.counts)}")
            print(f"   Distribución real: {dict(result.counts)}")
            print(f"   Capacidades objetivo: {cluster_capacities}")
            utilization = [f"{result.counts.get(i, 0)}/{cluster_capacities[i]}" for i in range(len(cluster_capacities))]
            print(f"   Utilización: {utilization}")
            print(f"   Métricas externas - ARI: {result.ari:.4f}, NMI: {result.nmi:.4f}, AMI: {result.ami:.4f}")
            print(f"   Métricas internas - Silhouette: {internal_metrics['silhouette_cosine']:.4f}")
            
            return clustering_result
            
        except Exception as e:
            print(f"❌ Error en clustering con capacidades personalizadas: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': {'id': model_id, 'name': model_info['name']},
                'dataset': dataset_num,
                'cluster_capacities': cluster_capacities
            }
            
    def save_results(self, results: dict, output_dir: str = "resultados"):
        """Guarda resultados de clustering en archivo JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        if results['success']:
            filename = f"online_clustering_{results['model']['id']}_dataset_{results['dataset']}.json"
        else:
            filename = f"online_clustering_error_{results['model']['id']}_dataset_{results['dataset']}.json"
            
        filepath = os.path.join(output_dir, filename)
        
        # Convertir tipos numpy
        results_converted = convert_numpy_types(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Resultados guardados en: {filepath}")
        return filepath


if __name__ == "__main__":
    # Evaluar todos los datasets
    evaluar_todos_los_datasets(k=3, resultados_dir="resultados")