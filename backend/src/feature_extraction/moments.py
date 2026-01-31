import os
import math
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog


# ============================================================
# MAPEO DE CLASES A ETIQUETAS NUMÉRICAS
# ============================================================
CLASE_LABELS = {
    1: {   # Dataset 1 - Vehículos
        "0": 0,
        "1": 1,
        "2": 2
    },
    2: {   # Dataset 2 - Animales
        "0": 0,
        "1": 1,
        "2": 2
    }
}



# ============================================================
# 1) UTILIDADES DE CARGA
# ============================================================
def listar_imagenes_binarias(data_root):
    """
    Busca en estructura:
      data/dataset_X/imagenes_procesadas/[clase]/binaria/
    Retorna: paths(list), labels(list), clase_names(list)
    """
    paths, labels, clase_names = [], [], []
    
    for dataset_num in [1, 2]:
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f"⚠ No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            binaria_path = os.path.join(clase_path, "binaria")
            if not os.path.isdir(binaria_path):
                print(f"⚠ No existe carpeta binaria: {binaria_path}")
                continue
            
            if dataset_num not in CLASE_LABELS or clase_folder not in CLASE_LABELS[dataset_num]:
                print(f"⚠ Clase no reconocida: {clase_folder} en dataset {dataset_num}")
                continue
            
            label = CLASE_LABELS[dataset_num][clase_folder]
            
            for fn in os.listdir(binaria_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(binaria_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
    
    return paths, labels, clase_names


def leer_gris(path, size=(256, 256)):
    """Lee imagen en escala de grises"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


# ============================================================
# 2) EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================
def features_momentos(img_gray):
    """
    Extrae 24 momentos: 10 regulares + 7 centrales + 7 normalizados
    Orden: m00,m10,m01,m20,m11,m02,m30,m21,m12,m03,
            mu20,mu11,mu02,mu30,mu21,mu12,mu03,
            nu20,nu11,nu02,nu30,nu21,nu12,nu03
    """
    M = cv2.moments(img_gray)
    
    # Momentos regulares
    m00 = M['m00']
    m10 = M['m10']
    m01 = M['m01']
    m20 = M['m20']
    m11 = M['m11']
    m02 = M['m02']
    m30 = M['m30']
    m21 = M['m21']
    m12 = M['m12']
    m03 = M['m03']
    
    # Momentos centrales
    mu20 = M['mu20']
    mu11 = M['mu11']
    mu02 = M['mu02']
    mu30 = M['mu30']
    mu21 = M['mu21']
    mu12 = M['mu12']
    mu03 = M['mu03']
    
    # Momentos centrales normalizados
    eps = 1e-10
    nu20 = mu20 / (m00 ** 2 + eps)
    nu11 = mu11 / (m00 ** 2 + eps)
    nu02 = mu02 / (m00 ** 2 + eps)
    nu30 = mu30 / (m00 ** 2.5 + eps)
    nu21 = mu21 / (m00 ** 2.5 + eps)
    nu12 = mu12 / (m00 ** 2.5 + eps)
    nu03 = mu03 / (m00 ** 2.5 + eps)
    
    feats = [
        m00, m10, m01, m20, m11, m02, m30, m21, m12, m03,
        mu20, mu11, mu02, mu30, mu21, mu12, mu03,
        nu20, nu11, nu02, nu30, nu21, nu12, nu03
    ]
    return np.array(feats, dtype=np.float64)


def features_hu(img_gray, log_scale=True, eps=1e-30):
    """
    Extrae 7 Momentos de Hu
    """
    M = cv2.moments(img_gray)
    hu = cv2.HuMoments(M).flatten().astype(np.float64)
    if log_scale:
        hu = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu


def features_zernike(img_gray, radius=21, degree=8):
    """
    Calcula Momentos de Zernike sin dependencias externas
    """
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    
    # Crear malla de coordenadas
    y, x = np.ogrid[:h, :w]
    x = x - cx
    y = y - cy
    rho = np.sqrt(x**2 + y**2) / radius
    rho = np.clip(rho, 0, 1)
    theta = np.arctan2(y, x)
    
    # Normalizar imagen
    img_norm = img_gray.astype(np.float64)
    img_norm = img_norm / (np.max(img_norm) + 1e-10)
    
    zernike_feats = []
    
    # Calcular momentos de Zernike
    for n in range(degree + 1):
        for m in range(-n, n + 1, 2):
            # Polinomio radial Zernike
            vnm = 0
            for s in range((n - abs(m)) // 2 + 1):
                coeff = ((-1) ** s * math.factorial(n - s)) / (
                    math.factorial(s) * 
                    math.factorial((n - 2*s + abs(m)) // 2) * 
                    math.factorial((n - 2*s - abs(m)) // 2)
                )
                vnm += coeff * (rho ** (n - 2*s))
            
            # Función angular
            real_part = vnm * np.cos(m * theta)
            imag_part = vnm * np.sin(m * theta)
            
            # Calcular el momento
            zmn_real = np.sum(img_norm * real_part)
            zmn_imag = np.sum(img_norm * imag_part)
            zmn_mag = np.sqrt(zmn_real**2 + zmn_imag**2)
            
            zernike_feats.append(zmn_mag)
    
    return np.array(zernike_feats, dtype=np.float64)



# ============================================================
# 3) CONSTRUCCIÓN DE DATASETS
# ============================================================
def construir_dataset(paths, labels, extractor_fn, nombres_clases=None, feature_names=None):
    """
    Extrae características de todas las imágenes
    """
    X = []
    valid_labels = []
    valid_nombres = []
    
    for i, p in enumerate(paths):
        try:
            img = leer_gris(p)
            feats = extractor_fn(img)
            X.append(feats)
            valid_labels.append(labels[i])
            if nombres_clases is not None:
                valid_nombres.append(nombres_clases[i])
        except Exception as e:
            print(f" Error procesando {p}: {e}")
            continue
    
    X = np.vstack(X)
    
    # Crear DataFrame con nombres de columnas si se proporcionan
    if feature_names is not None:
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = pd.DataFrame(X)
    
    df.insert(0, "Clase", np.array(valid_labels, dtype=int))

    
    return df


def generar_datasets_momentos(
    data_root="data",
    out_dir="data/features",
    z_radius=21,
    z_degree=8
):
    """
    Genera 3 datasets completos con características diferentes para cada dataset (1 y 2):
    1. Momentos clásicos
    2. Momentos de Hu
    3. Momentos de Zernike
    
    Estructura: data/features/dataset_1/, data/features/dataset_2/
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_num in [1, 2]:
        print("\n" + "=" * 60)
        print(f"PROCESANDO DATASET {dataset_num}")
        print("=" * 60)
        
        # Crear carpeta específica del dataset
        dataset_out_dir = os.path.join(out_dir, f"dataset_{dataset_num}")
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        # Leer imágenes del dataset específico
        paths, labels, clase_names = [], [], []
        dataset_path = os.path.join(data_root, f"dataset_{dataset_num}", "imagenes_procesadas")
        
        if not os.path.isdir(dataset_path):
            print(f" No existe: {dataset_path}")
            continue
        
        for clase_folder in os.listdir(dataset_path):
            clase_path = os.path.join(dataset_path, clase_folder)
            if not os.path.isdir(clase_path):
                continue
            
            contraste_path = os.path.join(clase_path, "contraste")
            if not os.path.isdir(contraste_path):
                continue
            
            if clase_folder not in CLASE_LABELS:
                continue
            
            label = CLASE_LABELS[clase_folder]
            
            for fn in os.listdir(binaria_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(binaria_path, fn))
                    labels.append(label)
                    clase_names.append(clase_folder)
        
        if len(paths) == 0:
            print(f" No se encontraron imágenes en dataset {dataset_num}")
            continue
        
        print(f" Total de imágenes: {len(paths)}")
        print(f"Clases encontradas: {set(clase_names)}")
        
        # --- Dataset: Momentos ---
        print(f"\n⏳ Extrayendo Momentos clásicos...")
        feature_names_momentos = [
            "m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
            "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
            "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"
        ]
        df_momentos = construir_dataset(paths, labels, features_momentos, clase_names, feature_names_momentos)
        
        momentos_path = os.path.join(dataset_out_dir, "dataset_momentos.csv")
        df_momentos.to_csv(momentos_path, index=False)
        print(f" Guardado: {momentos_path}")
        print(f"   Tamaño: {df_momentos.shape}")
        
        # --- Dataset: Hu ---
        print(f"\n Extrayendo Momentos de Hu...")
        df_hu = construir_dataset(paths, labels, lambda im: features_hu(im, log_scale=True), clase_names)
        
        hu_path = os.path.join(dataset_out_dir, "dataset_hu.csv")
        df_hu.to_csv(hu_path, index=False)
        print(f" Guardado: {hu_path}")
        print(f"   Tamaño: {df_hu.shape}")
        
        # --- Dataset: Zernike ---
        print(f"\n Extrayendo Momentos de Zernike...")
        df_zernike = construir_dataset(
            paths, labels, 
            lambda im: features_zernike(im, radius=z_radius, degree=z_degree), 
            clase_names
        )
        
        zernike_path = os.path.join(dataset_out_dir, "dataset_zernike.csv")
        df_zernike.to_csv(zernike_path, index=False)
        print(f" Guardado: {zernike_path}")
        print(f"   Tamaño: {df_zernike.shape}")
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN COMPLETADA")
    print("=" * 60)
    print(f"\nMapeo de clases:")
    for dataset_num, clases in CLASE_LABELS.items():
        print(f"Dataset {dataset_num}:")
        for clase, label in sorted(clases.items(), key=lambda x: x[1]):
            print(f"  {label}: {clase}")


if __name__ == "__main__":
    generar_datasets_momentos()