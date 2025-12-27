from .whitening import lds_whitening_transform
import numpy as np
from ssm.variational import SLDSMeanFieldVariationalPosterior as Q
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple


def get_clustering_results(lds_params: Tuple[np.ndarray], q: Q) -> np.ndarray:
    # whiten transform lds outputs
    C = lds_params[0].squeeze()
    _, _, _, C_prime = lds_whitening_transform(q.mean[0], q.mean[1], C)

    # choose k with sillhouete scores
    n_clusters = choose_k(C_prime)

    # calculate final labels from kmeans
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=50).fit_predict(
        C_prime.squeeze()
    )
    return labels


# choose k using sillhouette score from whitened lds outputs
def choose_k(
    embedding: np.ndarray,
    k_range: Tuple[int] = (2, 3, 4, 5),
    min_cluster_size: int = 10,
    top_n: int = 2,
    random_state: int = 10,
) -> int:
    """
    Pick k using both silhouette and Davies–Bouldin (DB) index.

    Steps:
      1. For each k in k_range, run KMeans and compute:
         - silhouette score
         - DB index
         - minimum cluster size
      2. Discard ks with any cluster < min_cluster_size.
      3. Among the remaining ks, take the top_n by silhouette,
         then choose the one with the smallest DB.
    """
    X = embedding.squeeze()
    results = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)

        # cluster sizes
        counts = np.bincount(labels)
        min_size = counts.min()

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append({"k": k, "sil": sil, "db": db, "min_size": min_size})

    # keep only ks with clusters of decent size
    valid = [r for r in results if r["min_size"] >= min_cluster_size]
    if not valid:
        # fallback: just use best silhouette if everything is tiny
        best = max(results, key=lambda r: r["sil"])
        return best["k"]

    # sort by silhouette (desc), take top_n
    valid_sorted = sorted(valid, key=lambda r: r["sil"], reverse=True)
    top = valid_sorted[:top_n]

    # among top silhouette candidates, pick smallest DB
    best = min(top, key=lambda r: r["db"])
    return best["k"]
