# src/cnn/embeddings.py
import os
from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Número máximo de imágenes por clase
MAX_IMGS = 200

def extraer_embeddings_cnn(data_root="datasets", out_dir="src/feature_vectors", max_imgs=MAX_IMGS, img_size=(256,256), batch_size=32):
    """
    Extrae embeddings CNN de ambos datasets y los guarda organizados por dataset en feature_vectors
    """
    import tensorflow as tf
    
    # Configuración para evitar problemas de memoria
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
    
    # Crear directorio de salida principal
    os.makedirs(out_dir, exist_ok=True)
    
    # Crear directorios para cada dataset
    for dataset_num in [1, 2]:
        dataset_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_dir, exist_ok=True)
    
    # Definir configuración de ambos datasets
    datasets_config = {
        1: {"0": {"prefix": "d1_c0", "class_name": "Dataset1_Clase0"},
            "1": {"prefix": "d1_c1", "class_name": "Dataset1_Clase1"}, 
            "2": {"prefix": "d1_c2", "class_name": "Dataset1_Clase2"}},
        2: {"0": {"prefix": "d2_c0", "class_name": "Dataset2_Clase0"},
            "1": {"prefix": "d2_c1", "class_name": "Dataset2_Clase1"},
            "2": {"prefix": "d2_c2", "class_name": "Dataset2_Clase2"}}
    }
    
    print("="*60)
    print("EXTRAYENDO EMBEDDINGS CNN")
    print("="*60)
    
    # Cargar modelo ResNet50 una sola vez
    print("\nCargando modelo ResNet50 preentrenado...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size[0], img_size[1], 3))
    print("Modelo cargado ✅")
    
    # Procesar cada dataset por separado
    for dataset_num in [1, 2]:
        print(f"\n--- Procesando Dataset {dataset_num} ---")
        dataset_path = Path(data_root) / f"dataset_{dataset_num}"
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        
        if not dataset_path.exists():
            print(f"⚠ No existe: {dataset_path}")
            continue
            
        clases = datasets_config[dataset_num]
        dataset_images = []
        dataset_labels = []
        dataset_paths = []
        
        # Recopilar imágenes de cada clase
        for clase_folder, cfg in clases.items():
            input_dir = dataset_path / "imagenes_crudas" / clase_folder
            if not input_dir.exists():
                print(f"⚠ Carpeta no encontrada: {input_dir}")
                continue
                
            image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
            imgs = [p for ext in image_extensions for p in input_dir.glob(ext)]
            
            if len(imgs) == 0:
                print(f"⚠ No se encontraron imágenes en: {input_dir}")
                continue
                
            # Seleccionar aleatoriamente max_imgs
            imgs_sel = random.sample(imgs, min(max_imgs, len(imgs)))
            
            dataset_images.extend(imgs_sel)
            dataset_labels.extend([f"d{dataset_num}_c{clase_folder}"] * len(imgs_sel))
            dataset_paths.extend([str(p) for p in imgs_sel])
            
            print(f"  Clase {clase_folder}: {len(imgs_sel)} imágenes")
        
        print(f"Total Dataset {dataset_num}: {len(dataset_images)} imágenes")
        
        # Extraer embeddings del dataset actual
        if dataset_images:
            embeddings, paths = procesar_imagenes_por_lotes(dataset_images, model, img_size, batch_size)
            
            # Convertir a arrays numpy
            final_embeddings = np.array(embeddings)
            final_labels = np.array(dataset_labels[:len(embeddings)])  # Ajustar por imágenes fallidas
            final_paths = np.array(paths)
            
            print(f"\nResumen Dataset {dataset_num}:")
            print(f"Embeddings generados: {len(final_embeddings)}")
            print(f"Shape de embeddings: {final_embeddings.shape}")
            print(f"Distribución por clases:")
            
            unique_labels, counts = np.unique(final_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"  {label}: {count} imágenes")
            
            # Guardar archivos para este dataset
            print(f"\nGuardando archivos en: {dataset_out_dir}")
            
            embeddings_path = os.path.join(dataset_out_dir, "Embeddings_cnn.npy")
            labels_path = os.path.join(dataset_out_dir, "Labels_cnn.npy")
            
            np.save(embeddings_path, final_embeddings)
            np.save(labels_path, final_labels)
            
            # Crear CSV con metadatos
            metadata = pd.DataFrame({
                'image_path': final_paths,
                'class': final_labels,
                'dataset': [int(label.split('_')[0][1]) for label in final_labels],
                'clase_num': [int(label.split('_')[1][1]) for label in final_labels]
            })
            metadata_path = os.path.join(dataset_out_dir, "embeddings_cnn_metadata.csv")
            metadata.to_csv(metadata_path, index=False)
            
            print(f"✅ Guardado: {embeddings_path}")
            print(f"✅ Guardado: {labels_path}")
            print(f"✅ Guardado: {metadata_path}")
            
            print(f"\nInformación de embeddings Dataset {dataset_num}:")
            print(f"   Dimensión por imagen: {final_embeddings.shape[1]} features")
            print(f"   Memoria aprox: {(final_embeddings.nbytes / 1024**2):.2f} MB")
    
    print(f"\n" + "="*60)
    print("EXTRACCIÓN CNN COMPLETADA")
    print("="*60)
    print(f"Los archivos se han guardado organizados por dataset en: {out_dir}")
    print(f"Estructura:")
    print(f"  {out_dir}/dataset_1/")
    print(f"  {out_dir}/dataset_2/")
    

def procesar_imagenes_por_lotes(image_paths, model, img_size, batch_size):
    """
    Procesa imágenes en lotes para extracción de embeddings
    """
    embeddings = []
    valid_paths = []
    total_imgs = len(image_paths)
    
    print(f"Procesando {total_imgs} imágenes en lotes de {batch_size}...")
    start_time = time.time()
    
    for i in range(0, total_imgs, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []
        
        for img_path in batch_paths:
            try:
                # Cargar y preprocesar imagen
                img = load_img(img_path, target_size=img_size)
                x = img_to_array(img)
                x = preprocess_input(x)
                batch_images.append(x)
                batch_valid_paths.append(str(img_path))
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
                continue
        
        if batch_images:  # Si hay imágenes válidas en el batch
            # Procesar batch completo
            batch_array = np.array(batch_images)
            batch_embeddings = model.predict(batch_array, verbose=0)
            
            # Agregar resultados
            embeddings.extend(batch_embeddings)
            valid_paths.extend(batch_valid_paths)
        
        # Mostrar progreso
        if (i // batch_size) % 5 == 0 and i > 0:
            elapsed = time.time() - start_time
            processed = min(i + batch_size, total_imgs)
            print(f"  Procesadas {processed}/{total_imgs} imágenes ({elapsed:.1f}s)")
    
    return embeddings, valid_paths

def extraer_embeddings(dataset_path, clases, max_imgs=MAX_IMGS, img_size=(256,256), batch_size=32):
    """
    dataset_path : Path o str : carpeta base 'data/dataset_X/'
    clases : dict : {'nombre_carpeta_cruda': {'prefix': 'prefijo', 'class_name': 'YOLO'}}
    """
    dataset_path = Path(dataset_path)
    all_images = []
    class_labels = []

    # ----------------------------
    # Recopilar imágenes y etiquetas
    # ----------------------------
    for carpeta, cfg in clases.items():
        input_dir = dataset_path / "imagenes_crudas" / carpeta
        if not input_dir.exists():
            print(f"⚠ Carpeta no encontrada: {input_dir}")
            continue
        image_extensions = ("*.jpg", "*.jpeg", "*.png")
        imgs = [p for ext in image_extensions for p in input_dir.glob(ext)]
        # Seleccionar aleatoriamente max_imgs
        imgs_sel = random.sample(imgs, min(max_imgs, len(imgs)))

        all_images.extend(imgs_sel)
        class_labels.extend([cfg['prefix']] * len(imgs_sel))

    print(f"\nTotal imágenes a procesar para embeddings: {len(all_images)}")

    # ----------------------------
    # Cargar modelo ResNet50
    # ----------------------------
    print("\nCargando modelo ResNet50 preentrenado...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size[0], img_size[1],3))
    print("Modelo cargado ✅")

    # ----------------------------
    # Listas para almacenar resultados
    # ----------------------------
    embeddings = []
    image_paths = []

    start_time = time.time()
    print("\nComenzando extracción de embeddings...")

    # ----------------------------
    # Procesar por lotes para mayor eficiencia
    # ----------------------------
    for i in range(0, len(all_images), batch_size):
        batch_paths = all_images[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            try:
                # Cargar y preprocesar imagen
                img = load_img(img_path, target_size=img_size)
                x = img_to_array(img)
                x = preprocess_input(x)
                batch_images.append(x)
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
                continue

        if batch_images:  # Si hay imágenes válidas en el batch
            # Procesar batch completo
            batch_array = np.array(batch_images)
            batch_embeddings = model.predict(batch_array, verbose=0)

            # Agregar resultados
            embeddings.extend(batch_embeddings)

            # Guardar rutas correspondientes
            image_paths.extend([str(p) for p in batch_paths])

        # Mostrar progreso cada 5 lotes
        if (i // batch_size) % 5 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"  Procesadas {min(i + batch_size, len(all_images))}/{len(all_images)} imágenes "
                  f"({elapsed:.1f}s)")

    # ----------------------------
    # Convertir a arrays numpy
    # ----------------------------
    embeddings = np.array(embeddings)
    class_labels = np.array(class_labels)
    image_paths = np.array(image_paths)

    print(f"\nExtracción completada!")
    print(f"════════════════════════════════════════")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Número de imágenes procesadas: {len(embeddings)}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")

    # ----------------------------
    # Distribución por clases
    # ----------------------------
    print(f"\nDistribución por clases:")
    unique_classes, counts = np.unique(class_labels, return_counts=True)
    for clase, count in zip(unique_classes, counts):
        print(f"  {clase}: {count} imágenes")

    # ----------------------------
    # Guardar resultados
    # ----------------------------
    print(f"\nGuardando resultados...")
    output_dir = dataset_path / "embeddings"
    output_dir.mkdir(exist_ok=True)

    # Guardar embeddings
    np.save(output_dir / "Embeddings.npy", embeddings)

    # Guardar etiquetas
    np.save(output_dir / "Labels.npy", class_labels)


    # Crear archivo CSV con metadatos
    metadata = pd.DataFrame({
        'image_path': image_paths,
        'class': class_labels
    })
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print(f"Resultados guardados en: {output_dir}")
    print(f"   - embeddings.npy (shape: {embeddings.shape})")
    print(f"   - class_labels.npy ({len(class_labels)} etiquetas)")
    print(f"   - image_paths.npy ({len(image_paths)} rutas)")
    print(f"   - metadata.csv")

    # ----------------------------
    # Información adicional
    # ----------------------------
    print(f"\nInformación de embeddings:")
    print(f"   Dimensión por imagen: {embeddings.shape[1]} features")
    print(f"   Memoria aprox: {(embeddings.nbytes / 1024**2):.2f} MB")

    return embeddings, class_labels, image_paths

if __name__ == "__main__":
    # Ejecutar extracción de embeddings CNN
    extraer_embeddings_cnn()

