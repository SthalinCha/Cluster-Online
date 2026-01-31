# API de Clustering - Documentación

## Descripción
API REST para evaluar algoritmos de clustering online en diferentes vectores de características extraídos de imágenes.

## Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install flask flask-cors pandas numpy scikit-learn scipy
```

### 2. Ejecutar la API
```bash
# Desde el directorio backend/
python src/run_api.py
```

La API estará disponible en: `http://localhost:5000`

## Endpoints Disponibles

### 📋 Información General

#### `GET /`
Información básica de la API y lista de endpoints disponibles.

#### `GET /config`
Configuración completa de hiperparámetros y modelos disponibles.
```json
{
  "success": true,
  "config": {
    "hyperparameters": {...},
    "models": {...},
    "metrics": {...}
  }
}
```

### 🤖 Modelos

#### `GET /modelos`
Lista todos los modelos de características disponibles.
```json
{
  "success": true,
  "modelos": [
    {
      "id": "momentos_clasicos",
      "nombre": "Momentos Clásicos",
      "descripcion": "Momentos regulares, centrales y normalizados",
      "tipo": "CSV",
      "n_muestras": 150,
      "n_features": 24,
      "disponible": true
    }
  ]
}
```

#### `GET /modelo/<modelo_id>`
Información detallada de un modelo específico.
```json
{
  "success": true,
  "modelo": {
    "id": "momentos_hu",
    "nombre": "Momentos de Hu",
    "n_muestras": 150,
    "n_features": 7,
    "estadisticas": {...},
    "distribucion_clases": {...}
  }
}
```

#### `GET /modelo/<modelo_id>/recomendaciones`
Recomendaciones específicas para un modelo.
```json
{
  "success": true,
  "recomendaciones": {
    "recommended_k": 3,
    "model_specific_tips": [
      "Momentos de Hu son invariantes...",
      "Solo 7 dimensiones, puede usar thresholds más estrictos"
    ]
  }
}
```

### 📊 Resultados y Evaluaciones

#### `GET /resultados`
Resumen de resultados de clustering si están disponibles.
```json
{
  "success": true,
  "resultados": [...],
  "mejores_modelos": {
    "mejor_ari": "CNN Embeddings (ResNet50)",
    "mejor_silhouette": "Momentos de Hu"
  }
}
```

#### `POST /clustering`
Ejecuta clustering con hiperparámetros personalizados en un modelo específico.

**Request Body:**
```json
{
  "modelo_id": "momentos_hu",
  "k": 3,
  "cluster_similarity_threshold": 0.75,
  "subcluster_similarity_threshold": 0.85,
  "pair_similarity_maximum": 0.95,
  "random_state": 42
}
```

**Response:**
```json
{
  "success": true,
  "modelo": {...},
  "parametros": {...},
  "metricas": {
    "external_metrics": {
      "ari": {"value": 0.85, "interpretation": "higher_better"},
      "nmi": {"value": 0.78, "interpretation": "higher_better"}
    },
    "internal_metrics": {
      "silhouette": {"value": 0.65, "interpretation": "higher_better"},
      "davies_bouldin": {"value": 0.45, "interpretation": "lower_better"}
    }
  },
  "clusters": {...}
}
```

#### `POST /evaluar`
Evalúa todos los modelos disponibles con hiperparámetros específicos.

**Request Body (opcional):**
```json
{
  "k": 3,
  "cluster_similarity_threshold": 0.8,
  "random_state": 42
}
```

### 🧪 Experimentos

#### `GET /experimentos`
Historial de experimentos guardados.

#### `POST /experimento`
Guarda un experimento con nombre personalizado.

**Request Body:**
```json
{
  "experiment_name": "test_hyperparams_v1",
  "parameters": {...},
  "results": [...]
}
```

## Hiperparámetros Disponibles

### Clustering Online
- **k**: Número de clusters (default: 3, rango: 2-10)
- **cluster_similarity_threshold**: Umbral de similitud entre clusters (default: 0.75, rango: 0.1-0.99)
- **subcluster_similarity_threshold**: Umbral de similitud para subclusters (default: 0.85, rango: 0.1-0.99)
- **pair_similarity_maximum**: Similitud máxima entre pares (default: 0.95, rango: 0.1-0.99)
- **random_state**: Semilla aleatoria (default: 42, rango: 0-9999)

## Modelos de Características

1. **Momentos Clásicos** (`momentos_clasicos`)
   - 24 características: momentos regulares, centrales y normalizados
   - Recomendado k=3

2. **Momentos de Hu** (`momentos_hu`)
   - 7 características: momentos invariantes
   - Recomendado k=3

3. **Momentos de Zernike** (`momentos_zernike`)
   - Variables características según parámetros
   - Recomendado k=3

4. **SIFT Features** (`sift_features`)
   - 512 características: estadísticas de descriptores SIFT
   - Recomendado k=3

5. **HOG Features** (`hog_features`)
   - Variables características: histograma de gradientes
   - Recomendado k=3

6. **CNN Embeddings** (`cnn_embeddings`)
   - 2048 características: embeddings ResNet50
   - Recomendado k=6 (2 datasets × 3 clases)

## Métricas de Evaluación

### Métricas Externas (vs etiquetas verdaderas)
- **ARI**: Adjusted Rand Index [-1, 1] (mayor es mejor)
- **NMI**: Normalized Mutual Information [0, 1] (mayor es mejor)
- **AMI**: Adjusted Mutual Information [0, 1] (mayor es mejor)

### Métricas Internas
- **Silhouette**: Cohesión vs separación [-1, 1] (mayor es mejor)
- **Davies-Bouldin**: Relación intra/inter cluster [0, ∞] (menor es mejor)
- **Calinski-Harabasz**: Varianza inter vs intra [0, ∞] (mayor es mejor)

## Ejemplos de Uso

### Evaluar un modelo específico
```bash
curl -X POST http://localhost:5000/clustering \
  -H "Content-Type: application/json" \
  -d '{
    "modelo_id": "momentos_hu",
    "k": 4,
    "cluster_similarity_threshold": 0.8
  }'
```

### Obtener recomendaciones
```bash
curl http://localhost:5000/modelo/cnn_embeddings/recomendaciones
```

### Evaluar todos los modelos
```bash
curl -X POST http://localhost:5000/evaluar \
  -H "Content-Type: application/json" \
  -d '{
    "k": 3,
    "random_state": 123
  }'
```

## Códigos de Error

- **400**: Error de validación de parámetros
- **404**: Modelo o recurso no encontrado
- **500**: Error interno del servidor

## Notas Importantes

1. **Prerequisitos**: Ejecutar primero `main.py` para generar los vectores de características
2. **Balanceo**: El algoritmo requiere n = k*m muestras exactas
3. **Preprocesamiento**: Los datos se escalán y normalizan automáticamente
4. **Experimentos**: Se guardan automáticamente en `src/feature_vectors/experiments/`