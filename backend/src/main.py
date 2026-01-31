import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.adquisicion_datos import descargar_dataset_1, descargar_dataset_2
from preprocesamiento.preprocesamiento import procesar_imagenes
from feature_extraction.feature_processor import extraer_todas_las_caracteristicas
from cnn.embeddings import extraer_embeddings_cnn
from clustering.models import evaluar_todos_los_datasets, OnlineClusteringAPI

def verificar_caracteristicas_extraidas(out_dir):
    """
    Verifica si ya existen archivos CSV de características para ambos datasets
    """
    archivos_esperados = [
        "momentos_clasicos.csv",
        "momentos_hu.csv", 
        "momentos_zernike.csv",
        "sift_features.csv",
        "hog_features.csv"
    ]
    
    # Verificar archivos para ambos datasets
    for dataset_num in [1, 2]:
        dataset_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        
        # Si el directorio del dataset no existe, faltan archivos
        if not os.path.exists(dataset_dir):
            return False
            
        # Verificar cada archivo CSV en el dataset
        for archivo in archivos_esperados:
            ruta_archivo = os.path.join(dataset_dir, archivo)
            if not os.path.exists(ruta_archivo):
                return False
    
    return True

def verificar_embeddings_cnn(out_dir):
    """
    Verifica si ya existen archivos de embeddings CNN para ambos datasets
    """
    archivos_esperados = [
        "Embeddings_cnn.npy",
        "Labels_cnn.npy"
    ]
    
    # Verificar archivos para ambos datasets
    for dataset_num in [1, 2]:
        dataset_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        
        # Si el directorio del dataset no existe, faltan archivos
        if not os.path.exists(dataset_dir):
            return False
            
        # Verificar cada archivo NPY en el dataset
        for archivo in archivos_esperados:
            ruta_archivo = os.path.join(dataset_dir, archivo)
            if not os.path.exists(ruta_archivo):
                return False
    
    return True

def verificar_resultados_clustering(resultados_dir):
    """
    Verifica si ya existen archivos de resultados de clustering
    """
    archivos_esperados = [
        "clustering_results_dataset_1.json",
        "clustering_results_dataset_2.json",
        "clustering_evaluation_results.csv"
    ]
    
    for archivo in archivos_esperados:
        ruta_archivo = os.path.join(resultados_dir, archivo)
        if not os.path.exists(ruta_archivo):
            return False
    return True

def verificar_imagenes_procesadas(carpeta_procesadas):
    """
    Verifica si ya existen imágenes procesadas en las subcarpetas
    Retorna True si ya hay imágenes procesadas
    """
    tipos = ["binaria", "contraste", "gris"]
    clases = ["0", "1", "2"]
    
    for tipo in tipos:
        for clase in clases:
            ruta_tipo_clase = os.path.join(carpeta_procesadas, tipo, clase)
            if os.path.exists(ruta_tipo_clase):
                # Verificar si tiene imágenes
                archivos = [f for f in os.listdir(ruta_tipo_clase) 
                           if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
                if len(archivos) > 0:
                    return True
    return False

def preparar_rutas(dataset_num, dataset_func):
    """
    Devuelve un diccionario de rutas de cada clase dentro del dataset.
    Si ya existe la carpeta de crudas, no descarga.
    """
    carpeta_dataset = f"datasets/dataset_{dataset_num}"
    carpeta_crudas = os.path.join(carpeta_dataset, "imagenes_crudas")

    if not os.path.exists(carpeta_crudas):
        print(f"⬇ Descargando Dataset {dataset_num}...")
        rutas = dataset_func()
    else:
        print(f"✅ Dataset {dataset_num} ya existe, no se descarga.")
        # Crear diccionario de clases basado en subcarpetas
        rutas = {clase: os.path.join(carpeta_crudas, clase)
                 for clase in os.listdir(carpeta_crudas)
                 if os.path.isdir(os.path.join(carpeta_crudas, clase))}

    # Asegurar que la carpeta de procesadas exista
    carpeta_procesadas = os.path.join(carpeta_dataset, "imagenes_procesadas")
    os.makedirs(carpeta_procesadas, exist_ok=True)

    return rutas, carpeta_procesadas

def api_demo_clustering_online(feature_dir, resultados_dir):
    """
    Demostración de la nueva API de clustering online
    """
    try:
        print("🔧 Inicializando API de Clustering Online...")
        api = OnlineClusteringAPI(feature_vectors_dir=feature_dir)
        
        # Mostrar configuración actual
        print("📋 Configuración actual:")
        params = api.get_hyperparameters()
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Mostrar modelos disponibles
        print("\n📊 Modelos disponibles:")
        models = api.list_available_models()
        for model_id, info in models.items():
            print(f"   {model_id}: {info['name']}")
        
        # Prueba con un modelo rápido (Momentos de Hu)
        print("\n🧪 Ejecutando prueba con Momentos de Hu (Dataset 1)...")
        
        # Configurar parámetros optimizados
        api.update_hyperparameters(
            k=3,
            m=40,
            cluster_similarity_threshold=0.8,
            use_flexible=True
        )
        
        resultado = api.cluster(
            model_id='momentos_hu',
            dataset_num=1,
            use_flexible=True
        )
        
        if resultado['success']:
            print("✅ Demo exitosa!")
            print(f"   Clusters formados: {resultado['clustering_results']['n_clusters_formed']}")
            print(f"   Distribución: {resultado['clustering_results']['cluster_counts']}")
            print(f"   ARI: {resultado['metrics']['external']['ARI']:.4f}")
            print(f"   NMI: {resultado['metrics']['external']['NMI']:.4f}")
            print(f"   Silhouette: {resultado['metrics']['internal']['Silhouette']:.4f}")
            
            # Guardar resultado de la demo
            api.save_results(resultado, resultados_dir)
            
        else:
            print(f"❌ Error en demo: {resultado['error']}")
            
        print("💡 La API está lista para usar en src/api/app.py")
        
    except Exception as e:
        print(f"⚠️ Error en demo de API: {e}")
        print("💡 La evaluación tradicional se completó correctamente")

def main():
    print("=== Preparando datasets ===\n")

    # ------------------------------
    # Dataset 1
    # ------------------------------
    rutas1, carpeta_proc1 = preparar_rutas(1, descargar_dataset_1)
    print("\nDataset 1 listo:")
    for etiqueta, ruta in rutas1.items():
        print(f"Clase {etiqueta}: {ruta}")

    # ------------------------------
    # Dataset 2
    # ------------------------------
    rutas2, carpeta_proc2 = preparar_rutas(2, descargar_dataset_2)
    print("\nDataset 2 listo:")
    for etiqueta, ruta in rutas2.items():
        print(f"Clase {etiqueta}: {ruta}")

    # ------------------------------
    # Procesar imágenes
    # ------------------------------
    print("\n=== Procesando imágenes ===")
    for dataset_rutas, carpeta_procesadas, nombre_dataset in zip(
        [rutas1, rutas2],
        [carpeta_proc1, carpeta_proc2],
        ["dataset_1", "dataset_2"]
    ):
        # Verificar si ya existen imágenes procesadas
        if verificar_imagenes_procesadas(carpeta_procesadas):
            print(f"✅ {nombre_dataset}: Las imágenes ya están procesadas, saltando procesamiento.")
            continue
            
        print(f"🔄 {nombre_dataset}: Procesando imágenes...")
        for etiqueta, carpeta_entrada in dataset_rutas.items():
            # Carpeta de salida dentro de imagenes_procesadas/tipo/clase
            carpeta_salida_gris = os.path.join(carpeta_procesadas, "gris", str(etiqueta))
            carpeta_salida_contraste = os.path.join(carpeta_procesadas, "contraste", str(etiqueta))
            carpeta_salida_binaria = os.path.join(carpeta_procesadas, "binaria", str(etiqueta))

            # Crear todas las carpetas necesarias
            os.makedirs(carpeta_salida_gris, exist_ok=True)
            os.makedirs(carpeta_salida_contraste, exist_ok=True)
            os.makedirs(carpeta_salida_binaria, exist_ok=True)

            # Procesar imágenes
            total = procesar_imagenes(
                carpeta_entrada,
                carpeta_salida_gris,
                carpeta_salida_contraste,
                carpeta_salida_binaria
            )
            print(f"[{nombre_dataset} Clase {etiqueta}] Procesadas {total} imágenes")

    print("\n✅ Todos los datasets procesados y organizados correctamente.")

    # ------------------------------
    # Extracción de características
    # ------------------------------
    print("\n=== Extrayendo características ===")
    feature_dir = "src/feature_vectors"
    
    if verificar_caracteristicas_extraidas(feature_dir):
        print("✅ Las características ya están extraídas para ambos datasets, saltando extracción.")
        print(f"   Verificado: {feature_dir}/dataset_1/ y {feature_dir}/dataset_2/")
    else:
        print("🔄 Extrayendo características para ambos datasets...")
        extraer_todas_las_caracteristicas(
            data_root="datasets",
            out_dir=feature_dir
        )

    # ------------------------------
    # Extracción de embeddings CNN
    # ------------------------------
    print("\n=== Extrayendo embeddings CNN ===")
    
    if verificar_embeddings_cnn(feature_dir):
        print("✅ Los embeddings CNN ya están extraídos para ambos datasets, saltando extracción.")
        print(f"   Verificado: {feature_dir}/dataset_1/ y {feature_dir}/dataset_2/")
    else:
        print("🔄 Extrayendo embeddings CNN para ambos datasets...")
        extraer_embeddings_cnn(
            data_root="datasets",
            out_dir=feature_dir
        )

    # ------------------------------
    # Evaluación con Clustering Online
    # ------------------------------
    print("\n=== Evaluando con Clustering Online ===")
    resultados_dir = "resultados"
    
    if verificar_resultados_clustering(resultados_dir):
        print("✅ Los resultados de clustering ya están generados.")
        print("💡 Para forzar una nueva evaluación, elimina la carpeta 'resultados/'")
    else:
        print("🔄 Evaluando datasets con clustering online...")
        print("   Método: LINKS-like con similitud coseno")
        print("   Parámetros: k=3 clusters, versión flexible para clases desbalanceadas")
        
        # Evaluación completa con método tradicional
        evaluar_todos_los_datasets(
            k=3,  # Número de clusters (3 clases por dataset)
            feature_vectors_dir=feature_dir,
            resultados_dir=resultados_dir
        )
        
        print("✅ Evaluación tradicional completada")
        
        # Demostración de la nueva API de clustering online
        print("\n🚀 Demostrando API de Clustering Online...")
        api_demo_clustering_online(feature_dir, resultados_dir)

    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETO EJECUTADO EXITOSAMENTE 🎉")
    print("="*60)
    print(f"📁 Features y embeddings en: {feature_dir}")
    print(f"📊 Resultados de clustering en: {resultados_dir}")
    print("📈 Revisa los archivos de clustering para ver el rendimiento de cada método:")
    print(f"   - {resultados_dir}/clustering_results_dataset_1.json")
    print(f"   - {resultados_dir}/clustering_results_dataset_2.json") 
    print(f"   - {resultados_dir}/clustering_evaluation_results.csv")
    print("\n🆕 Archivos de la API Online:")
    print(f"   - {resultados_dir}/online_clustering_*.json")
    print("\n🌐 Para usar la API web, ejecuta: python src/api/app.py")

if __name__ == "__main__":
    main()
