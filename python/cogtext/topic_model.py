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

    # caches
    self.reducer_cache = Path('models/embeddings/abstracts_UMAP-5d.npz')
    self.clusterer_cache = joblib.Memory(location='tmp')

    self.reducer = UMAP(n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        verbose=True,
                        n_jobs=-1,
                        low_memory=False)

    self.clusterer = HDBSCAN(min_cluster_size=5,
                             # min_samples=1,
                             metric='euclidean',
                             core_dist_n_jobs=-1,
                             memory=self.clusterer_cache,
                             prediction_data=True)

  def fit_transform(self, X, y=None):
    """Fit and transform documents with UMAP"""

    if isinstance(self.reducer_cache, Path) and self.reducer_cache.exists():
      # load the embedding from cached file
      reduced = np.load(self.reducer_cache)['arr_0']
    else:
      reduced = self.reducer.fit_transform(X, y)
      reduced = np.nan_to_num(reduced)

    self.verbose and print('[TopicModel] reduced embedding dimension.')

    clusters = self.clusterer.fit_predict(X)
    # weights = self.clusterer.probabilities_
    weights = hdbscan.all_points_membership_vectors(self.clusterer)
    weights[clusters < 0] = 0.0

    return clusters, weights
