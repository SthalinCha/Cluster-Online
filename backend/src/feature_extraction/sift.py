import os
import math
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# Mapeo de clases a etiquetas numéricas
CLASE_LABELS = {
    "1": 0,
    "2": 1,
    "3": 2
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

# Mapeo de clases a etiquetas numéricas
CLASE_LABELS = {
    "1": 0,
    "2": 1,
    "3": 2
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

def features_sift(img_gray):
    """
    Extrae características SIFT: estadísticas (mean, std, min, max) de los 128 descriptores
    Total: 512 características (128 dimensiones × 4 estadísticas)
    Si no hay keypoints: retorna vector de ceros
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    # Si no hay descriptores, retornar vector de ceros
    if descriptors is None:
        return np.zeros(512, dtype=np.float64)
    
    # descriptors shape: (num_keypoints, 128)
    descriptors = descriptors.astype(np.float64)
    
    # Calcular estadísticas sobre cada una de las 128 dimensiones
    mean_desc = np.mean(descriptors, axis=0)  # 128
    std_desc = np.std(descriptors, axis=0)    # 128
    min_desc = np.min(descriptors, axis=0)    # 128
    max_desc = np.max(descriptors, axis=0)    # 128
    
    # Concatenar: mean + std + min + max = 512 características
    sift_feats = np.concatenate([mean_desc, std_desc, min_desc, max_desc])
    
    return sift_feats




def generar_datasets_sift(data_root="data", out_dir="data/features"):
    """
    Extrae características SIFT de las imágenes de contraste
    Genera dataset_sift.csv para cada dataset (1 y 2)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_num in [1, 2]:
        print("\n" + "=" * 60)
        print(f"PROCESANDO SIFT - DATASET {dataset_num}")
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
            
            # CAMBIO: usar carpeta "contraste" en lugar de "binaria"
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
        
        # --- Dataset: SIFT ---
        print(f"\n Extrayendo características SIFT...")
        df_sift = construir_dataset(paths, labels, features_sift, clase_names)
        
        sift_path = os.path.join(dataset_out_dir, "dataset_sift.csv")
        df_sift.to_csv(sift_path, index=False)
        print(f" Guardado: {sift_path}")
        print(f"   Tamaño: {df_sift.shape}")
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN SIFT COMPLETADA")
    print("=" * 60)
    print(f"\nMapeo de clases:")
    for clase, label in sorted(CLASE_LABELS.items(), key=lambda x: x[1]):
        print(f"  {label}: {clase}")


if __name__ == "__main__":
    generar_datasets_sift()

