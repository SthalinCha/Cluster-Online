import os
import math
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# Mapeo de clases a etiquetas numéricas
CLASE_LABELS = {
    "0": 0,
    "1": 1,
    "2": 2
}

def construir_dataset(paths, labels, feature_func, clase_names):
    """Construye dataset aplicando función de extracción de características"""
    features = []
    valid_labels = []
    valid_names = []
    
    for i, path in enumerate(paths):
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            feat = feature_func(img)
            features.append(feat)
            valid_labels.append(labels[i])
            valid_names.append(clase_names[i])
        except Exception as e:
            print(f"Error procesando {path}: {e}")
            continue
    
    # Crear DataFrame
    features_array = np.array(features)
    df = pd.DataFrame(features_array)
    df['label'] = valid_labels
    df['clase_name'] = valid_names
    
    return df

def features_hog(img_gray):
    """
    Extrae características HOG (Histogram of Oriented Gradients)
    Parámetros: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)
    Retorna: vector de características HOG
    """
    # Asegurar que es grayscale
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    
    # Extraer HOG
    features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    
    return features.astype(np.float64)

def generar_datasets_hog(data_root="data", out_dir="data/features"):
    """
    Extrae características HOG de las imágenes de contraste
    Genera dataset_hog.csv para cada dataset (1 y 2)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_num in [1, 2]:
        print("\n" + "=" * 60)
        print(f"PROCESANDO HOG - DATASET {dataset_num}")
        print("=" * 60)
        
        # Crear carpeta específica del dataset
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        # Leer imágenes de CONTRASTE del dataset específico
        paths, labels, clase_names = [], [], []
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f" No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            # Usar carpeta "contraste"
            contraste_path = os.path.join(clase_path, "contraste")
            if not os.path.isdir(contraste_path):
                continue
            
            if clase_folder not in CLASE_LABELS:
                continue
            
            label = CLASE_LABELS[clase_folder]
            
            for fn in os.listdir(contraste_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(contraste_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
        
        if len(paths) == 0:
            print(f" No se encontraron imágenes en dataset {dataset_num}")
            continue
        
        print(f" Total de imágenes: {len(paths)}")
        print(f"Clases encontradas: {set(clase_names)}")
        
        # --- Dataset: HOG ---
        print(f"\n Extrayendo características HOG...")
        df_hog = construir_dataset(paths, labels, features_hog, clase_names)
        
        hog_path = os.path.join(dataset_out_dir, "dataset_hog.csv")
        df_hog.to_csv(hog_path, index=False)
        print(f" Guardado: {hog_path}")
        print(f"   Tamaño: {df_hog.shape}")
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN HOG COMPLETADA")
    print("=" * 60)
    print(f"\nMapeo de clases:")
    for clase, label in sorted(CLASE_LABELS.items(), key=lambda x: x[1]):
        print(f"  {label}: {clase}")


if __name__ == "__main__":
    generar_datasets_hog()