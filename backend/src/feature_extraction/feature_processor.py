import os
import sys
import numpy as np
import pandas as pd
import cv2

# Agregar src al path para importaciones absolutas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.moments import features_momentos, features_hu, features_zernike
from feature_extraction.sift import features_sift
from feature_extraction.hog import features_hog

# Mapeo de clases a etiquetas numéricas
CLASE_LABELS = {
    1: {"0": 0, "1": 1, "2": 2},  # Dataset 1
    2: {"0": 0, "1": 1, "2": 2}   # Dataset 2
}

def listar_imagenes_por_tipo(data_root, tipo_imagen):
    """
    Lista imágenes de un tipo específico (binaria/contraste)
    Estructura: datasets/dataset_X/imagenes_procesadas/tipo_imagen/clase/
    Retorna: paths, labels, clase_names, dataset_nums
    """
    paths, labels, clase_names, dataset_nums = [], [], [], []
    
    for dataset_num in [1, 2]:
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f"⚠ No existe: {dataset_path}")
            continue
        
        # La estructura es: imagenes_procesadas/tipo_imagen/clase/
        tipo_path = os.path.join(dataset_path, tipo_imagen)
        if not os.path.isdir(tipo_path):
            print(f"⚠ No existe carpeta {tipo_imagen}: {tipo_path}")
            continue
        
        for clase_folder in os.listdir(tipo_path):
            clase_path = os.path.join(tipo_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            if dataset_num not in CLASE_LABELS or clase_folder not in CLASE_LABELS[dataset_num]:
                print(f"⚠ Clase no reconocida: {clase_folder} en dataset {dataset_num}")
                continue
            
            label = CLASE_LABELS[dataset_num][clase_folder]
            
            for fn in os.listdir(clase_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(clase_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
                    dataset_nums.append(dataset_num)
    
    return paths, labels, clase_names, dataset_nums

def listar_imagenes_por_tipo_dataset(data_root, tipo_imagen, dataset_num):
    """
    Lista imágenes de un tipo específico para un dataset específico
    Estructura: datasets/dataset_X/imagenes_procesadas/tipo_imagen/clase/
    Retorna: paths, labels, clase_names, dataset_nums
    """
    paths, labels, clase_names, dataset_nums = [], [], [], []
    
    dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
    
    if not os.path.isdir(dataset_path):
        print(f"⚠ No existe: {dataset_path}")
        return paths, labels, clase_names, dataset_nums
    
    # La estructura es: imagenes_procesadas/tipo_imagen/clase/
    tipo_path = os.path.join(dataset_path, tipo_imagen)
    if not os.path.isdir(tipo_path):
        print(f"⚠ No existe carpeta {tipo_imagen}: {tipo_path}")
        return paths, labels, clase_names, dataset_nums
    
    for clase_folder in os.listdir(tipo_path):
        clase_path = os.path.join(tipo_path, clase_folder)
        if not os.path.isdir(clase_path):
            continue
        
        if dataset_num not in CLASE_LABELS or clase_folder not in CLASE_LABELS[dataset_num]:
            print(f"⚠ Clase no reconocida: {clase_folder} en dataset {dataset_num}")
            continue
        
        label = CLASE_LABELS[dataset_num][clase_folder]
        
        for fn in os.listdir(clase_path):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                paths.append(os.path.join(clase_path, fn))
                labels.append(label)
                clase_names.append(clase_folder)
                dataset_nums.append(dataset_num)
    
    return paths, labels, clase_names, dataset_nums

def leer_imagen(path, size=(256, 256)):
    """Lee imagen en escala de grises"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def construir_dataset_features(paths, labels, clase_names, dataset_nums, feature_func, feature_names=None):
    """
    Construye dataset aplicando función de extracción de características
    """
    features = []
    valid_labels = []
    valid_clase_names = []
    valid_dataset_nums = []
    
    for i, path in enumerate(paths):
        try:
            img = leer_imagen(path)
            feat = feature_func(img)
            features.append(feat)
            valid_labels.append(labels[i])
            valid_clase_names.append(clase_names[i])
            valid_dataset_nums.append(dataset_nums[i])
        except Exception as e:
            print(f"Error procesando {path}: {e}")
            continue
    
    # Crear DataFrame
    features_array = np.array(features)
    
    if feature_names is not None:
        df = pd.DataFrame(features_array, columns=feature_names)
    else:
        # Crear nombres genéricos
        num_features = features_array.shape[1]
        df = pd.DataFrame(features_array, columns=[f"feature_{i}" for i in range(num_features)])
    
    # Agregar metadatos
    df['label'] = valid_labels
    df['clase_name'] = valid_clase_names
    df['dataset'] = valid_dataset_nums
    
    return df

def extraer_todas_las_caracteristicas(data_root="datasets", out_dir="src/feature_vectors"):
    """
    Extrae todas las características y las guarda en CSVs separados organizados por dataset
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Crear directorios para cada dataset
    for dataset_num in [1, 2]:
        dataset_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_dir, exist_ok=True)
    
    print("="*60)
    print("EXTRACCIÓN DE CARACTERÍSTICAS")
    print("="*60)
    
    # Procesar cada dataset por separado
    for dataset_num in [1, 2]:
        print(f"\n--- PROCESANDO DATASET {dataset_num} ---")
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        
        # ============================================================
        # CARACTERÍSTICAS DE MOMENTOS (usando imágenes BINARIAS)
        # ============================================================
        print(f"\n1. Procesando imágenes BINARIAS para momentos - Dataset {dataset_num}...")
        paths_bin, labels_bin, clases_bin, datasets_bin = listar_imagenes_por_tipo_dataset(data_root, "binaria", dataset_num)
        
        if len(paths_bin) > 0:
            print(f"   - Encontradas {len(paths_bin)} imágenes binarias")
            
            # Momentos clásicos
            print("   - Extrayendo momentos clásicos...")
            feature_names_momentos = [
                "m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
                "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
                "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"
            ]
            df_momentos = construir_dataset_features(
                paths_bin, labels_bin, clases_bin, datasets_bin,
                features_momentos, feature_names_momentos
            )
            
            momentos_path = os.path.join(dataset_out_dir, "momentos_clasicos.csv")
            df_momentos.to_csv(momentos_path, index=False)
            print(f"     ✅ Guardado: {momentos_path} | Tamaño: {df_momentos.shape}")
            
            # Momentos de Hu
            print("   - Extrayendo momentos de Hu...")
            feature_names_hu = [f"hu_{i}" for i in range(7)]
            df_hu = construir_dataset_features(
                paths_bin, labels_bin, clases_bin, datasets_bin,
                lambda img: features_hu(img, log_scale=True), feature_names_hu
            )
            
            hu_path = os.path.join(dataset_out_dir, "momentos_hu.csv")
            df_hu.to_csv(hu_path, index=False)
            print(f"     ✅ Guardado: {hu_path} | Tamaño: {df_hu.shape}")
            
            # Momentos de Zernike
            print("   - Extrayendo momentos de Zernike...")
            df_zernike = construir_dataset_features(
                paths_bin, labels_bin, clases_bin, datasets_bin,
                lambda img: features_zernike(img, radius=21, degree=8)
            )
            
            zernike_path = os.path.join(dataset_out_dir, "momentos_zernike.csv")
            df_zernike.to_csv(zernike_path, index=False)
            print(f"     ✅ Guardado: {zernike_path} | Tamaño: {df_zernike.shape}")
        
        else:
            print(f"   ⚠ No se encontraron imágenes binarias para dataset {dataset_num}")
        
        # ============================================================
        # CARACTERÍSTICAS SIFT y HOG (usando imágenes de CONTRASTE)
        # ============================================================
        print(f"\n2. Procesando imágenes de CONTRASTE para SIFT y HOG - Dataset {dataset_num}...")
        paths_cont, labels_cont, clases_cont, datasets_cont = listar_imagenes_por_tipo_dataset(data_root, "contraste", dataset_num)
        
        if len(paths_cont) > 0:
            print(f"   - Encontradas {len(paths_cont)} imágenes de contraste")
            
            # SIFT
            print("   - Extrayendo características SIFT...")
            feature_names_sift = []
            for stat in ["mean", "std", "min", "max"]:
                for i in range(128):
                    feature_names_sift.append(f"sift_{stat}_{i}")
            
            df_sift = construir_dataset_features(
                paths_cont, labels_cont, clases_cont, datasets_cont,
                features_sift, feature_names_sift
            )
            
            sift_path = os.path.join(dataset_out_dir, "sift_features.csv")
            df_sift.to_csv(sift_path, index=False)
            print(f"     ✅ Guardado: {sift_path} | Tamaño: {df_sift.shape}")
            
            # HOG
            print("   - Extrayendo características HOG...")
            df_hog = construir_dataset_features(
                paths_cont, labels_cont, clases_cont, datasets_cont,
                features_hog
            )
            
            hog_path = os.path.join(dataset_out_dir, "hog_features.csv")
            df_hog.to_csv(hog_path, index=False)
            print(f"     ✅ Guardado: {hog_path} | Tamaño: {df_hog.shape}")
        
        else:
            print(f"   ⚠ No se encontraron imágenes de contraste para dataset {dataset_num}")
    
    print("\n" + "="*60)
    print("EXTRACCIÓN COMPLETADA")
    print("="*60)
    print(f"Los archivos CSV se han guardado organizados por dataset en: {out_dir}")
    print(f"Estructura:")
    print(f"  {out_dir}/dataset_1/")
    print(f"  {out_dir}/dataset_2/")
    print(f"\nMapeo de clases:")
    for dataset_num, clases in CLASE_LABELS.items():
        print(f"Dataset {dataset_num}:")
        for clase, label in sorted(clases.items(), key=lambda x: x[1]):
            print(f"  {label}: Clase {clase}")

if __name__ == "__main__":
    extraer_todas_las_caracteristicas()