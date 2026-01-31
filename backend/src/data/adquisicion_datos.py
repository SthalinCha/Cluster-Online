import os
import shutil
import random
import kagglehub
import cv2

SEED = 42
MAX_IMGS = 200
IMG_SIZE = (256, 256)


def descargar_dataset_generico(
    dataset_id: str,
    clases: dict,
    dataset_num: int,
    max_imgs: int = MAX_IMGS,
    seed: int = SEED,
    img_size: tuple = IMG_SIZE
):
    """
    Descarga un dataset desde Kaggle y copia un número limitado de imágenes
    redimensionadas a un tamaño fijo en carpetas numéricas (0,1,2,...)
    """

    random.seed(seed)

    base_output = f"datasets/dataset_{dataset_num}/imagenes_crudas"
    os.makedirs(base_output, exist_ok=True)

    print(f"\n⬇ Descargando dataset: {dataset_id}")
    path = kagglehub.dataset_download(dataset_id)

    rutas = {}

    for root, dirs, _ in os.walk(path):
        for etiqueta, carpeta_origen in clases.items():
            if carpeta_origen in dirs and etiqueta not in rutas:

                origen = os.path.join(root, carpeta_origen)
                destino = os.path.join(base_output, str(etiqueta))
                os.makedirs(destino, exist_ok=True)

                imgs = [
                    f for f in os.listdir(origen)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if not imgs:
                    raise RuntimeError(f"No hay imágenes en {carpeta_origen}")

                seleccion = random.sample(
                    imgs, min(max_imgs, len(imgs))
                )

                for i, img_name in enumerate(seleccion):
                    src = os.path.join(origen, img_name)

                    img = cv2.imread(src)
                    if img is None:
                        continue

                    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

                    dst_name = f"{etiqueta}_{i}.jpg"
                    dst = os.path.join(destino, dst_name)

                    cv2.imwrite(dst, img)

                print(f" Clase {etiqueta}: {len(seleccion)} imágenes guardadas")
                rutas[etiqueta] = destino

        if len(rutas) == len(clases):
            break

    if len(rutas) != len(clases):
        raise RuntimeError("No se encontraron todas las clases")

    return rutas


def descargar_dataset_1():
    clases = {
        0: "Motorcycles",
        1: "Planes",
        2: "Ships"
    }

    return descargar_dataset_generico(
        dataset_id="mohamedmaher5/vehicle-classification",
        clases=clases,
        dataset_num=1
    )

def descargar_dataset_2():
    clases = {
        0: "gatto",
        1: "cavallo",
        2: "elefante"
    }

    return descargar_dataset_generico(
        dataset_id="alessiocorrado99/animals10",
        clases=clases,
        dataset_num=2
    )
