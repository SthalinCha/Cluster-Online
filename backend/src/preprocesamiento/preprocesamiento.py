import os
import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO("yolov8n-seg.pt")

# ====================================================
#        FUNCIÓN PARA CONVERTIR A ESCALA DE GRISES
# ====================================================
def escala_grises(img):
    # Convierte una imagen BGR (OpenCV) a gris
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================
#   1. ESTADÍSTICAS DE IMAGEN
# ============================
def miStat(P):
    media = np.mean(P)
    varianza = np.var(P)
    desviacion = np.std(P)
    return media, varianza, desviacion


# ============================
#   2. AMPLIACIÓN DE HISTOGRAMA
# ============================
def miAmpliH(P, a, b, L):
    P = np.array(P, dtype=float)
    X = np.zeros_like(P)
    X = L * ((P - a) / (b - a))
    X[X < 0] = 0
    X[X > L] = L
    return X


# ============================
#   3. TRANSFORMACIÓN CUADRÁTICA
# ============================
def micuadrada(P, L, O):
    P = np.array(P, dtype=float)
    X = np.zeros_like(P)

    if O == 0:
        X = P**2 / L
    elif O == 1:
        X = np.sqrt(L * P)

    X[X < 0] = 0
    X[X > L] = L
    return X


# ============================
#   4. ECUALIZACIÓN DE HISTOGRAMA
# ============================
def miEcualizador(P):
    L_max = 255
    N, M = P.shape
    total_pixeles = N * M

    hist, _ = np.histogram(P.flatten(), 256, [0, 256])
    cdf = hist.cumsum()

    cdf_normalized = np.round((L_max / total_pixeles) * cdf).astype(np.uint8)

    X = cdf_normalized[P]
    return X


# ============================
#   5. SEGMENTACIÓN CON YOLO
# ============================
def segmentar_yolo(img, umbral=0.7):
    """
    Segmenta la imagen usando YOLO (detecta el objeto principal)
    
    Parameters
    ----------
    img : ndarray
        Imagen en BGR (OpenCV)
    umbral : float
        Umbral de confianza para la máscara (default: 0.7)
    
    Returns
    -------
    ndarray or None
        Máscara binaria del objeto detectado o None si no detecta
    """
    h, w = img.shape[:2]
    
    try:
        res = model(img, verbose=False, conf=0.3)[0]
        
        if res.masks is None:
            return None
        
        masks = res.masks.data.cpu().numpy()
        
        # Seleccionar la máscara más grande (objeto principal)
        if len(masks) == 0:
            return None
        
        best_idx = np.argmax([masks[i].sum() for i in range(len(masks))])
        mask = (masks[best_idx] >= umbral).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return mask * 255
    
    except Exception as e:
        return None


# ============================
#   6. BINARIZACIÓN CON OTSU
# ============================
def otsu(img):
    """
    Aplica binarización usando el método de Otsu
    """
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


# ================================================
#     PROCESAMIENTO BASADO EN PERCENTILES
#     Clasificación automática de imágenes
# ================================================
def procesar_prioridad_unica(img, percentiles):
    Lmax = 255
    
    # Desempacar percentiles según notación del usuario
    p10 = percentiles[1]
    p25 = percentiles[2]
    p50 = percentiles[3]
    p75 = percentiles[4]
    p90 = percentiles[5]

    # ------------------------------------------------
    # 1. SUBEXPOSICIÓN
    # Imagen oscura: mediana y altas luces bajas.
    # ------------------------------------------------
    if p50 < 85 and p90 < 150:
        salida = micuadrada(img, Lmax, 1)   # √ intensidades
        return salida, "subexpo_raiz"

    # ------------------------------------------------
    # 2. SOBREEXPOSICIÓN
    # Imagen clara: sombras altas + mediana alta.
    # ------------------------------------------------
    if p50 > 170 and p10 > 100:
        salida = micuadrada(img, Lmax, 0)   # intensidades² / L
        return salida, "sobreexpo_cuadratica"

    # ------------------------------------------------
    # 3. BAJO CONTRASTE
    # Distribución de intensidades muy comprimida.
    # ------------------------------------------------
    if (p90 - p10) < 40:
        a = np.min(img)
        b = np.max(img)
        salida = miAmpliH(img, a, b, Lmax)
        return salida, "bajo_contraste_stretching"

    # ------------------------------------------------
    # 4. IMAGEN CORRECTA
    # No requiere mejora.
    # ------------------------------------------------
    return img, "original"


# ====================================================
#         PROCESAR IMÁGENES DE UNA CARPETA
# ====================================================
def procesar_imagenes(carpeta_entrada, carpeta_salida_gris, carpeta_salida_contraste, carpeta_salida_binaria):
    # ============================================
    #       0. CREAR CARPETAS DE SALIDA
    # ============================================
    os.makedirs(carpeta_salida_gris, exist_ok=True)
    os.makedirs(carpeta_salida_contraste, exist_ok=True)
    os.makedirs(carpeta_salida_binaria, exist_ok=True)

    # Filtrar imágenes válidas de la carpeta
    imagenes = [f for f in os.listdir(carpeta_entrada)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    total_convertidas = 0  # contador global

    # ============================================
    #     1. PROCESAR CADA IMAGEN DEL DIRECTORIO
    # ============================================
    for i, archivo in enumerate(imagenes):

        ruta_completa = os.path.join(carpeta_entrada, archivo)

        # Leer imagen
        img = cv2.imread(ruta_completa)
        if img is None:
            print(f"[ERROR] No se pudo leer: {ruta_completa}")
            continue

        # Rescalar
        img = cv2.resize(img, (256, 256))

        # --------------------------------------------
        #     1.1 CONVERTIR IMAGEN A ESCALA DE GRISES
        # --------------------------------------------
        gris = escala_grises(img)

        # Guardar imagen en gris (256x256)
        nombre_base = os.path.splitext(archivo)[0]
        archivo_gris = os.path.join(carpeta_salida_gris, f"{nombre_base}_gris.png")
        gris_rescaled = cv2.resize(gris, (256, 256))
        cv2.imwrite(archivo_gris, gris_rescaled)

        # --------------------------------------------
        #     1.2 MEJORA DE CONTRASTE BASADA EN PERCENTILES
        # --------------------------------------------
        p = np.percentile(gris, [10, 25, 50, 75, 90])
        percentiles = [None] + list(p)
        resultado_contraste, nombre_op = procesar_prioridad_unica(gris, percentiles)

        # Guardar imagen con contraste mejorado (256x256)
        archivo_contraste = os.path.join(carpeta_salida_contraste, f"{nombre_base}_contraste.png")
        contraste_rescaled = cv2.resize(resultado_contraste.astype(np.uint8), (256, 256))
        cv2.imwrite(archivo_contraste, contraste_rescaled)

        # --------------------------------------------
        #     1.3 SEGMENTACIÓN CON YOLO + BINARIZACIÓN
        # --------------------------------------------
        # Convertir a BGR para YOLO
        contraste_bgr = cv2.cvtColor(contraste_rescaled, cv2.COLOR_GRAY2BGR)
        mascara_yolo = segmentar_yolo(contraste_bgr)

        # Si YOLO detecta algo
        if mascara_yolo is not None:
            objeto_segmentado = cv2.bitwise_and(contraste_rescaled, contraste_rescaled, mask=mascara_yolo.astype(np.uint8))
            binaria = otsu(objeto_segmentado)
        else:
            # Si YOLO no detecta, usar imagen de contraste completa
            binaria = otsu(contraste_rescaled)

        # Guardar imagen binarizada (256x256)
        archivo_binario = os.path.join(carpeta_salida_binaria, f"{nombre_base}_bin.png")
        binaria_rescaled = cv2.resize(binaria, (256, 256))
        cv2.imwrite(archivo_binario, binaria_rescaled)

        total_convertidas += 1

    return total_convertidas
