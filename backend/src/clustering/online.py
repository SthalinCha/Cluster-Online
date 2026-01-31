import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter
from dataclasses import dataclass

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, normalize


# Cosine similarity

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine(a, b)

# LINKS-like online 

class Subcluster:
    def __init__(self, initial_vector: np.ndarray):
        self.centroid = initial_vector.copy()
        self.n_vectors = 1

    def add(self, vector: np.ndarray):
        self.n_vectors += 1
        self.centroid = (self.n_vectors - 1) / self.n_vectors * self.centroid + vector / self.n_vectors


class LinksClusterReusable:
    def __init__(
        self,
        cluster_similarity_threshold: float = 0.75,
        subcluster_similarity_threshold: float = 0.85,
        pair_similarity_maximum: float = 0.95,
    ):
        self.clusters = []  # list[list[Subcluster]]
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.subcluster_similarity_threshold = subcluster_similarity_threshold
        self.pair_similarity_maximum = pair_similarity_maximum

    def sim_threshold(self, k: int, kp: int) -> float:
        s = (1.0 + 1.0 / k * (1.0 / self.cluster_similarity_threshold**2 - 1.0))
        s *= (1.0 + 1.0 / kp * (1.0 / self.cluster_similarity_threshold**2 - 1.0))
        s = 1.0 / np.sqrt(s)
        s = (
            self.cluster_similarity_threshold**2
            + (self.pair_similarity_maximum - self.cluster_similarity_threshold**2)
            / (1.0 - self.cluster_similarity_threshold**2)
            * (s - self.cluster_similarity_threshold**2)
        )
        return s


# -------------------------
# LINKS-like 100% online con restriccion de tamanio (capacidad)
# -------------------------
class LinksClusterCapacityOnline(LinksClusterReusable):
    """Asignacion online con restriccion dura: cada cluster puede recibir como maximo m puntos.

    Importante:
    - No hace rebalance offline.
    - Si n == k*m y nunca se excede la capacidad, al final todos los clusters quedan en m.
    """

    def __init__(
        self,
        k: int,
        m: int,
        cluster_similarity_threshold: float = 0.75,
        subcluster_similarity_threshold: float = 0.85,
        pair_similarity_maximum: float = 0.95,
    ):
        super().__init__(
            cluster_similarity_threshold=cluster_similarity_threshold,
            subcluster_similarity_threshold=subcluster_similarity_threshold,
            pair_similarity_maximum=pair_similarity_maximum,
        )
        if k <= 0:
            raise ValueError("k debe ser > 0")
        if m <= 0:
            raise ValueError("m debe ser > 0")
        self.k = int(k)
        self.m = int(m)
        self.cluster_counts: list[int] = []

    def _has_capacity(self, cid: int) -> bool:
        return self.cluster_counts[cid] < self.m

    def _append_new_cluster(self, x: np.ndarray) -> int:
        if len(self.clusters) >= self.k:
            raise RuntimeError("No se pueden crear mas clusters (k alcanzado)")
        self.clusters.append([Subcluster(x)])
        self.cluster_counts.append(1)
        return len(self.clusters) - 1

    def predict(self, x: np.ndarray) -> int:
        # primer punto
        if len(self.clusters) == 0:
            return self._append_new_cluster(x)

        # si todos los clusters existentes estan llenos y aun puedo crear mas, creo uno nuevo
        if all(c >= self.m for c in self.cluster_counts):
            if len(self.clusters) < self.k:
                return self._append_new_cluster(x)
            raise RuntimeError("Capacidad excedida: todos los clusters estan llenos")

        # buscar el mejor subcluster dentro de clusters con capacidad
        best_cid, best_sc, best_sim = None, None, -np.inf
        for cid, cl in enumerate(self.clusters):
            if not self._has_capacity(cid):
                continue
            for sc in cl:
                s = cos_sim(x, sc.centroid)
                if s > best_sim:
                    best_sim = s
                    best_cid = cid
                    best_sc = sc

        if best_cid is None or best_sc is None:
            # no hubo candidato con capacidad (deberia estar cubierto arriba)
            if len(self.clusters) < self.k:
                return self._append_new_cluster(x)
            raise RuntimeError("No hay clusters con capacidad disponible")

        # caso: asignar al subcluster mas parecido
        if best_sim >= self.subcluster_similarity_threshold:
            best_sc.add(x)
            self.cluster_counts[best_cid] += 1
            return best_cid

        # caso: crear nuevo subcluster dentro del mejor cluster, si cumple umbral
        new_sc = Subcluster(x)
        s_link = cos_sim(new_sc.centroid, best_sc.centroid)
        if s_link >= self.sim_threshold(best_sc.n_vectors, 1):
            self.clusters[best_cid].append(new_sc)
            self.cluster_counts[best_cid] += 1
            return best_cid

        # caso: crear nuevo cluster si se permite
        if len(self.clusters) < self.k:
            return self._append_new_cluster(x)
        # caso: no puedo crear mas clusters, asigno al mejor disponible 
        best_sc.add(x)
        self.cluster_counts[best_cid] += 1
        return best_cid


# -------------------------
# LINKS-like online con capacidades personalizadas por cluster
# -------------------------
class LinksClusterCustomCapacityOnline(LinksClusterReusable):
    """Asignacion online con restriccion personalizada: cada cluster puede tener su propia capacidad maxima.
    
    Ejemplo: cluster 0 max 40, cluster 1 max 30, cluster 2 max 30
    """

    def __init__(
        self,
        cluster_capacities: list[int],
        cluster_similarity_threshold: float = 0.75,
        subcluster_similarity_threshold: float = 0.85,
        pair_similarity_maximum: float = 0.95,
    ):
        super().__init__(
            cluster_similarity_threshold=cluster_similarity_threshold,
            subcluster_similarity_threshold=subcluster_similarity_threshold,
            pair_similarity_maximum=pair_similarity_maximum,
        )
        if not cluster_capacities or any(c <= 0 for c in cluster_capacities):
            raise ValueError("Todas las capacidades deben ser > 0")
        
        self.cluster_capacities = [int(c) for c in cluster_capacities]
        self.k = len(self.cluster_capacities)
        self.cluster_counts: list[int] = []

    def _has_capacity(self, cid: int) -> bool:
        return self.cluster_counts[cid] < self.cluster_capacities[cid]

    def _append_new_cluster(self, x: np.ndarray) -> int:
        if len(self.clusters) >= self.k:
            raise RuntimeError("No se pueden crear mas clusters (k alcanzado)")
        self.clusters.append([Subcluster(x)])
        self.cluster_counts.append(1)
        return len(self.clusters) - 1

    def predict(self, x: np.ndarray) -> int:
        # primer punto
        if len(self.clusters) == 0:
            return self._append_new_cluster(x)

        # si todos los clusters existentes estan llenos y aun puedo crear mas, creo uno nuevo
        if all(self.cluster_counts[i] >= self.cluster_capacities[i] for i in range(len(self.cluster_counts))):
            if len(self.clusters) < self.k:
                return self._append_new_cluster(x)
            raise RuntimeError("Capacidad excedida: todos los clusters estan llenos")

        # buscar el mejor subcluster dentro de clusters con capacidad
        best_cid, best_sc, best_sim = None, None, -np.inf
        for cid, cl in enumerate(self.clusters):
            if not self._has_capacity(cid):
                continue
            for sc in cl:
                s = cos_sim(x, sc.centroid)
                if s > best_sim:
                    best_sim = s
                    best_cid = cid
                    best_sc = sc

        if best_cid is None or best_sc is None:
            # no hubo candidato con capacidad (deberia estar cubierto arriba)
            if len(self.clusters) < self.k:
                return self._append_new_cluster(x)
            raise RuntimeError("No hay clusters con capacidad disponible")

        # caso: asignar al subcluster mas parecido
        if best_sim >= self.subcluster_similarity_threshold:
            best_sc.add(x)
            self.cluster_counts[best_cid] += 1
            return best_cid

        # caso: crear nuevo subcluster dentro del mejor cluster, si cumple umbral
        new_sc = Subcluster(x)
        s_link = cos_sim(new_sc.centroid, best_sc.centroid)
        if s_link >= self.sim_threshold(best_sc.n_vectors, 1):
            self.clusters[best_cid].append(new_sc)
            self.cluster_counts[best_cid] += 1
            return best_cid

        # caso: crear nuevo cluster si se permite
        if len(self.clusters) < self.k:
            return self._append_new_cluster(x)
        # caso: no puedo crear mas clusters, asigno al mejor disponible 
        best_sc.add(x)
        self.cluster_counts[best_cid] += 1
        return best_cid


@dataclass
class OnlineBalancedLinksResult:
    labels: np.ndarray
    counts: Counter
    nmi: float
    ami: float
    ari: float


def online_capacity_links_with_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    k: int,
    m: int,
    shuffle_data: bool = True,
    random_state: int = 42,
    cluster_similarity_threshold: float = 0.75,
    subcluster_similarity_threshold: float = 0.85,
    pair_similarity_maximum: float = 0.95,
) -> OnlineBalancedLinksResult:
    """Version 100% online: impone la restriccion de tamanio durante la asignacion.

    Restriccion implementada:
    - capacidad maxima m por cluster (dura).
    - si n == k*m, al final necesariamente quedan exactamente m por cluster.
    """
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true)

    n = len(X)
    if n != k * m:
        raise ValueError(
            f"Tu X tiene n={n}, pero para obtener tamanios exactamente iguales necesitas n=k*m={k*m}. "
            "(Si quieres solo capacidad maxima, puedo ajustarlo.)"
        )

    idx = np.arange(n)
    if shuffle_data:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
        Xw = X[idx]
        yw = y_true[idx]
    else:
        Xw = X
        yw = y_true

    model = LinksClusterCapacityOnline(
        k=k,
        m=m,
        cluster_similarity_threshold=cluster_similarity_threshold,
        subcluster_similarity_threshold=subcluster_similarity_threshold,
        pair_similarity_maximum=pair_similarity_maximum,
    )

    labels_w = np.array([model.predict(xi) for xi in Xw], dtype=int)

    # metricas externas (clusters no tienen orden -> NMI/AMI/ARI sirven)
    nmi = normalized_mutual_info_score(yw, labels_w)
    ami = adjusted_mutual_info_score(yw, labels_w)
    ari = adjusted_rand_score(yw, labels_w)

    # volver al orden original si se barajo
    if shuffle_data:
        inv = np.empty_like(idx)
        inv[idx] = np.arange(n)
        labels = labels_w[inv]
    else:
        labels = labels_w

    return OnlineBalancedLinksResult(
        labels=labels,
        counts=Counter(labels),
        nmi=nmi,
        ami=ami,
        ari=ari,
    )


def online_flexible_links_with_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    k: int,
    m: int,
    shuffle_data: bool = True,
    random_state: int = 42,
    cluster_similarity_threshold: float = 0.75,
    subcluster_similarity_threshold: float = 0.85,
    pair_similarity_maximum: float = 0.95,
) -> OnlineBalancedLinksResult:
    """Version online flexible: permite clases desbalanceadas con capacidad maxima m por cluster.
    
    A diferencia de online_capacity_links_with_metrics, esta funcion:
    - NO requiere que n == k*m exactamente
    - Permite clusters con diferentes tamaños (hasta m como maximo)
    - Maneja clases desbalanceadas naturalmente
    - Crea nuevos clusters dinamicamente hasta alcanzar k
    """
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true)
    n = len(X)
    
    if n == 0:
        raise ValueError("El dataset esta vacio")
    if k <= 0 or m <= 0:
        raise ValueError("k y m deben ser > 0")
    
    idx = np.arange(n)
    if shuffle_data:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
        Xw = X[idx]
        yw = y_true[idx]
    else:
        Xw = X
        yw = y_true

    model = LinksClusterCapacityOnline(
        k=k,
        m=m,
        cluster_similarity_threshold=cluster_similarity_threshold,
        subcluster_similarity_threshold=subcluster_similarity_threshold,
        pair_similarity_maximum=pair_similarity_maximum,
    )

    labels_w = []
    for i, xi in enumerate(Xw):
        try:
            label = model.predict(xi)
            labels_w.append(label)
        except RuntimeError as e:
            # Si todos los clusters estan llenos y no se puede crear mas
            # asignar al cluster con mayor similitud (sin restriccion de capacidad)
            if "Capacidad excedida" in str(e) or "todos los clusters estan llenos" in str(e):
                best_cid, best_sim = None, -np.inf
                for cid, cl in enumerate(model.clusters):
                    for sc in cl:
                        s = cos_sim(xi, sc.centroid)
                        if s > best_sim:
                            best_sim = s
                            best_cid = cid
                if best_cid is not None:
                    # Forzar asignacion al mejor cluster (excediendo capacidad)
                    best_subcluster = None
                    best_sim = -np.inf
                    for sc in model.clusters[best_cid]:
                        s = cos_sim(xi, sc.centroid)
                        if s > best_sim:
                            best_sim = s
                            best_subcluster = sc
                    best_subcluster.add(xi)
                    model.cluster_counts[best_cid] += 1
                    labels_w.append(best_cid)
                else:
                    # Caso extremo: crear cluster en posicion 0
                    labels_w.append(0)
            else:
                raise e

    labels_w = np.array(labels_w, dtype=int)
    
    # metricas externas
    nmi = normalized_mutual_info_score(yw, labels_w)
    ami = adjusted_mutual_info_score(yw, labels_w)
    ari = adjusted_rand_score(yw, labels_w)

    # volver al orden original si se barajo
    if shuffle_data:
        inv = np.empty_like(idx)
        inv[idx] = np.arange(n)
        labels = labels_w[inv]
    else:
        labels = labels_w

    return OnlineBalancedLinksResult(
        labels=labels,
        counts=Counter(labels),
        nmi=nmi,
        ami=ami,
        ari=ari,
    )


def online_custom_capacity_links_with_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    cluster_capacities: list[int],
    shuffle_data: bool = True,
    random_state: int = 42,
    cluster_similarity_threshold: float = 0.75,
    subcluster_similarity_threshold: float = 0.85,
    pair_similarity_maximum: float = 0.95,
) -> OnlineBalancedLinksResult:
    """Version online con capacidades personalizadas por cluster.
    
    Permite especificar una capacidad diferente para cada cluster.
    Ejemplo: cluster_capacities = [40, 30, 30] significa que:
    - Cluster 0 puede tener máximo 40 elementos
    - Cluster 1 puede tener máximo 30 elementos  
    - Cluster 2 puede tener máximo 30 elementos
    
    Args:
        cluster_capacities: Lista con la capacidad máxima para cada cluster
                          Por ejemplo [40, 30, 30] para 3 clusters con capacidades diferentes
    """
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true)
    n = len(X)
    
    if n == 0:
        raise ValueError("El dataset esta vacio")
    if not cluster_capacities or any(c <= 0 for c in cluster_capacities):
        raise ValueError("Todas las capacidades deben ser > 0")
    
    k = len(cluster_capacities)
    total_capacity = sum(cluster_capacities)
    
    if n > total_capacity:
        print(f"Advertencia: El dataset tiene {n} muestras pero la capacidad total es {total_capacity}. "
              f"Algunas muestras podrian exceder la capacidad.")
    
    idx = np.arange(n)
    if shuffle_data:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
        Xw = X[idx]
        yw = y_true[idx]
    else:
        Xw = X
        yw = y_true

    model = LinksClusterCustomCapacityOnline(
        cluster_capacities=cluster_capacities,
        cluster_similarity_threshold=cluster_similarity_threshold,
        subcluster_similarity_threshold=subcluster_similarity_threshold,
        pair_similarity_maximum=pair_similarity_maximum,
    )

    labels_w = []
    for i, xi in enumerate(Xw):
        try:
            label = model.predict(xi)
            labels_w.append(label)
        except RuntimeError as e:
            # Si todos los clusters estan llenos, asignar al cluster con mayor similitud
            if "Capacidad excedida" in str(e) or "todos los clusters estan llenos" in str(e):
                best_cid, best_sim = None, -np.inf
                for cid, cl in enumerate(model.clusters):
                    for sc in cl:
                        s = cos_sim(xi, sc.centroid)
                        if s > best_sim:
                            best_sim = s
                            best_cid = cid
                if best_cid is not None:
                    # Forzar asignacion al mejor cluster (excediendo capacidad)
                    best_subcluster = None
                    best_sim = -np.inf
                    for sc in model.clusters[best_cid]:
                        s = cos_sim(xi, sc.centroid)
                        if s > best_sim:
                            best_sim = s
                            best_subcluster = sc
                    best_subcluster.add(xi)
                    model.cluster_counts[best_cid] += 1
                    labels_w.append(best_cid)
                else:
                    # Caso extremo: asignar al primer cluster
                    labels_w.append(0)
            else:
                raise e

    labels_w = np.array(labels_w, dtype=int)
    
    # metricas externas
    nmi = normalized_mutual_info_score(yw, labels_w)
    ami = adjusted_mutual_info_score(yw, labels_w)
    ari = adjusted_rand_score(yw, labels_w)

    # volver al orden original si se barajo
    if shuffle_data:
        inv = np.empty_like(idx)
        inv[idx] = np.arange(n)
        labels = labels_w[inv]
    else:
        labels = labels_w

    return OnlineBalancedLinksResult(
        labels=labels,
        counts=Counter(labels),
        nmi=nmi,
        ami=ami,
        ari=ari,
    )


def online_custom_capacity_flexible_links_with_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    cluster_capacities: list[int],
    shuffle_data: bool = True,
    random_state: int = 42,
    cluster_similarity_threshold: float = 0.75,
    subcluster_similarity_threshold: float = 0.85,
    pair_similarity_maximum: float = 0.95,
    allow_overflow: bool = True,
) -> OnlineBalancedLinksResult:
    """Version completamente flexible con capacidades personalizadas.
    
    Similar a online_custom_capacity_links_with_metrics pero con manejo más flexible
    cuando se exceden las capacidades.
    
    Args:
        allow_overflow: Si True, permite exceder capacidades asignando al cluster más similar.
                       Si False, lanza excepción cuando se excede la capacidad total.
    """
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true)
    n = len(X)
    
    if n == 0:
        raise ValueError("El dataset esta vacio")
    if not cluster_capacities or any(c <= 0 for c in cluster_capacities):
        raise ValueError("Todas las capacidades deben ser > 0")
    
    total_capacity = sum(cluster_capacities)
    
    if n > total_capacity and not allow_overflow:
        raise ValueError(f"El dataset tiene {n} muestras pero la capacidad total es {total_capacity}. "
                        f"Establece allow_overflow=True para permitir exceder capacidades.")
    
    idx = np.arange(n)
    if shuffle_data:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
        Xw = X[idx]
        yw = y_true[idx]
    else:
        Xw = X
        yw = y_true

    model = LinksClusterCustomCapacityOnline(
        cluster_capacities=cluster_capacities,
        cluster_similarity_threshold=cluster_similarity_threshold,
        subcluster_similarity_threshold=subcluster_similarity_threshold,
        pair_similarity_maximum=pair_similarity_maximum,
    )

    labels_w = []
    overflow_count = 0
    
    for i, xi in enumerate(Xw):
        try:
            label = model.predict(xi)
            labels_w.append(label)
        except RuntimeError as e:
            if allow_overflow and ("Capacidad excedida" in str(e) or "todos los clusters estan llenos" in str(e)):
                # Estrategia de overflow: asignar al cluster con mayor similitud sin restriccion
                best_cid, best_sim = 0, -np.inf  # Default al cluster 0
                
                for cid, cl in enumerate(model.clusters):
                    cluster_sim = 0.0
                    cluster_size = 0
                    for sc in cl:
                        s = cos_sim(xi, sc.centroid)
                        cluster_sim += s * sc.n_vectors
                        cluster_size += sc.n_vectors
                    
                    # Similitud promedio ponderada del cluster
                    if cluster_size > 0:
                        avg_sim = cluster_sim / cluster_size
                        if avg_sim > best_sim:
                            best_sim = avg_sim
                            best_cid = cid
                
                # Agregar al mejor cluster encontrado (forzando overflow)
                if model.clusters[best_cid]:
                    best_subcluster = model.clusters[best_cid][0]  # Usar el primer subcluster
                    best_subcluster.add(xi)
                    model.cluster_counts[best_cid] += 1
                    labels_w.append(best_cid)
                    overflow_count += 1
                else:
                    labels_w.append(best_cid)
            else:
                raise e

    labels_w = np.array(labels_w, dtype=int)
    
    if overflow_count > 0:
        print(f"Advertencia: {overflow_count} muestras excedieron la capacidad de sus clusters asignados.")
    
    # metricas externas
    nmi = normalized_mutual_info_score(yw, labels_w)
    ami = adjusted_mutual_info_score(yw, labels_w)
    ari = adjusted_rand_score(yw, labels_w)

    # volver al orden original si se barajo
    if shuffle_data:
        inv = np.empty_like(idx)
        inv[idx] = np.arange(n)
        labels = labels_w[inv]
    else:
        labels = labels_w

    return OnlineBalancedLinksResult(
        labels=labels,
        counts=Counter(labels),
        nmi=nmi,
        ami=ami,
        ari=ari,
    )


def _safe_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Metricas internas tipicas; devuelve NaN si no son computables."""
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        return {
            "silhouette_cosine": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "n_clusters": int(n_clusters),
        }

    sil = silhouette_score(X, labels, metric="cosine")
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return {
        "silhouette_cosine": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "n_clusters": int(n_clusters),
    }


