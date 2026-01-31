from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from clustering.models import OnlineClusteringAPI, convert_numpy_types
except ImportError:
    print("Warning: Could not import clustering modules - running in basic mode")
    OnlineClusteringAPI = None
    convert_numpy_types = None

app = FastAPI(title="Clustering API", description="API de Clustering Online", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global de la API de clustering
try:
    if OnlineClusteringAPI:
        clustering_api = OnlineClusteringAPI(feature_vectors_dir="src/feature_vectors")
    else:
        clustering_api = None
except Exception as e:
    print(f"Warning: Could not initialize clustering API: {e}")
    clustering_api = None

class ClusteringRequest(BaseModel):
    model_id: str
    dataset_num: int
    use_flexible: Optional[bool] = True
    use_custom_capacities: Optional[bool] = False
    cluster_capacities: Optional[list[int]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class HyperparametersUpdate(BaseModel):
    k: Optional[int] = None
    m: Optional[int] = None
    cluster_similarity_threshold: Optional[float] = None
    subcluster_similarity_threshold: Optional[float] = None
    pair_similarity_maximum: Optional[float] = None
    random_state: Optional[int] = None
    normalize_data: Optional[bool] = None
    apply_pca: Optional[bool] = None
    pca_components: Optional[int] = None

@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "API de Clustering Online",
        "version": "1.0.0",
        "status": "working",
        "endpoints": ["/health", "/config", "/cluster", "/models"]
    }

@app.get("/config")
async def get_config():
    """Obtiene la configuración actual"""
    if not clustering_api:
        raise HTTPException(status_code=500, detail="Clustering API no inicializada")
    
    return {
        "success": True,
        "hyperparameters": clustering_api.get_hyperparameters(),
        "available_models": clustering_api.list_available_models()
    }

@app.put("/config")
async def update_config(params: HyperparametersUpdate):
    """Actualiza hiperparámetros"""
    if not clustering_api:
        raise HTTPException(status_code=500, detail="Clustering API no inicializada")
    
    try:
        # Convertir a dict solo los valores no None
        update_dict = {k: v for k, v in params.dict().items() if v is not None}
        
        if not update_dict:
            raise HTTPException(status_code=400, detail="No se proporcionaron parámetros para actualizar")
        
        clustering_api.update_hyperparameters(**update_dict)
        
        return {
            "success": True,
            "message": "Parámetros actualizados",
            "updated_parameters": clustering_api.get_hyperparameters()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster")
async def execute_clustering(request: ClusteringRequest):
    """Ejecuta clustering online"""
    if not clustering_api:
        raise HTTPException(status_code=500, detail="Clustering API no inicializada")
    
    try:
        # Actualizar hiperparámetros si se proporcionan
        if request.hyperparameters:
            clustering_api.update_hyperparameters(**request.hyperparameters)
        
        # Intentar clustering real primero
        try:
            if request.use_custom_capacities and request.cluster_capacities:
                # Clustering con capacidades personalizadas
                result = clustering_api.cluster_with_custom_capacities(
                    model_id=request.model_id,
                    dataset_num=request.dataset_num,
                    cluster_capacities=request.cluster_capacities
                )
            else:
                # Clustering normal
                result = clustering_api.cluster(
                    model_id=request.model_id,
                    dataset_num=request.dataset_num,
                    use_flexible=request.use_flexible
                )
            
            # Convertir tipos numpy a tipos nativos de Python
            if convert_numpy_types:
                result = convert_numpy_types(result)
            
            # Remover las etiquetas individuales completamente
            if 'clustering_results' in result and 'labels' in result['clustering_results']:
                labels = result['clustering_results']['labels']
                result['clustering_results']['total_samples'] = len(labels)
                # Remover completamente las etiquetas - no mostrar preview
                del result['clustering_results']['labels']
            
            return result
        except (FileNotFoundError, ValueError) as e:
            # Si no hay datos reales, simular resultado
            import numpy as np
            
            # Determinar capacidades para simulación
            if request.use_custom_capacities and request.cluster_capacities:
                capacities = request.cluster_capacities
                k = len(capacities)
                total_samples = sum(capacities)
                cluster_sizes = capacities.copy()
                algorithm_name = f"Online LINKS-like Custom Capacities (Demo)"
            else:
                hyperparams = clustering_api.get_hyperparameters()
                k = hyperparams.get('k', 3)
                m = hyperparams.get('m', 50)
                total_samples = min(100, k * m)
                cluster_sizes = [total_samples // k] * k
                # Distribuir el resto
                remainder = total_samples % k
                for i in range(remainder):
                    cluster_sizes[i] += 1
                algorithm_name = "Online LINKS-like (Demo)"
            
            cluster_counts = {str(i): size for i, size in enumerate(cluster_sizes)}
            
            return {
                "success": True,
                "message": "Demo mode - datos simulados",
                "model": {
                    "id": request.model_id,
                    "name": f"Modelo {request.model_id}",
                    "description": "Datos simulados para demo"
                },
                "dataset": request.dataset_num,
                "algorithm": algorithm_name,
                "data_info": {
                    "original_shape": [total_samples, 7],
                    "processed_shape": [total_samples, 7],
                    "preprocessing": {"pca_applied": False},
                    "true_classes": k
                },
                "hyperparameters": clustering_api.get_hyperparameters(),
                "clustering_results": {
                    "total_samples": total_samples,
                    "cluster_counts": cluster_counts,
                    "n_clusters_formed": k,
                    "cluster_sizes": cluster_sizes,
                    "custom_capacities": capacities if request.use_custom_capacities else None
                },
                "metrics": {
                    "external": {
                        "ARI": 0.7523,
                        "NMI": 0.6891,
                        "AMI": 0.6234
                    },
                    "internal": {
                        "Silhouette": 0.5678,
                        "Davies_Bouldin": 1.2345,
                        "Calinski_Harabasz": 156.789
                    }
                },
                "note": f"Datos reales no encontrados para {request.model_id} dataset {request.dataset_num}. Mostrando resultado simulado."
            }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Lista modelos disponibles"""
    if not clustering_api:
        raise HTTPException(status_code=500, detail="Clustering API no inicializada")
    
    return {
        "success": True,
        "models": clustering_api.list_available_models()
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy", 
        "clustering_api": clustering_api is not None,
        "message": "API is running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)