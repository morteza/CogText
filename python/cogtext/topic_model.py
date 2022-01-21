import hdbscan
from hdbscan import HDBSCAN
import joblib
import numpy as np
from pathlib import Path


class TopicModel():
  def __init__(self, parametric_umap=True, verbose=False) -> None:

    self.verbose = verbose

    # importing UMAP takes a while, so we import only when retraining is needed
    if parametric_umap:
      from umap.parametric_umap import ParametricUMAP as UMAP
    else:
      from umap import UMAP

    self.reducer = UMAP(n_neighbors=15,
                        n_components=5,
                        metric='euclidean',
                        min_dist=0.0,
                        verbose=True,
                        n_jobs=-1)

    self.clusterer = HDBSCAN(min_cluster_size=100,
                             min_samples=1,
                             metric='euclidean',
                             core_dist_n_jobs=-1,
                             memory=joblib.Memory(location='tmp'),
                             prediction_data=True)

  def fit_transform(self, X, y=None, umap_embeddings=None):
    """First reduce dimensionality of the input and apply soft-clustering.
    
    Returns:
      clusters: array of cluster labels.
      weights: array of cluster weights (soft clusters where applying argmax equals
        `clusters`).
    """

    if umap_embeddings is None:
      reduced = self.reducer.fit_transform(X, y)
      reduced = np.nan_to_num(reduced)
    else:
      # load the embedding from cached file
      reduced = umap_embeddings

    self.verbose and print('[TopicModel] Reduced embedding dimension. Now clustering...')

    clusters = self.clusterer.fit_predict(reduced)

    self.verbose and print('[TopicModel] Clustered embedding. Now computing weights...')

    # weights = self.clusterer.probabilities_
    weights = hdbscan.all_points_membership_vectors(self.clusterer)

    # commented because of soft clustering; we don't need to zero out weights for the noise points
    # weights[clusters < 0] = 0.0

    self.verbose and print('[TopicModel] Done!')

    return clusters, weights
