import os
import random
import cv2

SEED = 42


def cargar_imagenes(
    directorio: str,
    max_imgs: int = 200,
    size: tuple = (224, 224),
    recursive: bool = True
):
    """
    Carga imágenes desde un directorio (con o sin subcarpetas),
    sin etiquetas.
    
    Retorna:
    --------
    X : list[np.ndarray]
    """

    random.seed(SEED)
    X = []

    archivos = []

    if recursive:
        for root, _, files in os.walk(directorio):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    archivos.append(os.path.join(root, f))
    else:
        archivos = [
            os.path.join(directorio, f)
            for f in os.listdir(directorio)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    seleccion = random.sample(
        archivos, min(max_imgs, len(archivos))
    )

    for ruta_img in seleccion:
        img = cv2.imread(ruta_img)
        if img is None:
            continue

        img = cv2.resize(img, size)
        X.append(img)

    return X

#X = cargar_imagenes("imagenes", max_imgs=200)
#print(f"Cargadas {len(X)} imágenes desde 'imagenes/'")