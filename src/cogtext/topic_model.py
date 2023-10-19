import hdbscan
from hdbscan import HDBSCAN
import joblib
import numpy as np


class TopicModel():
  def __init__(self, parametric_umap=True, verbose=False) -> None:

    self.verbose = verbose
    self.parametric_umap = parametric_umap
    self.umap_embeddings_ = None

    # importing UMAP takes a while, so we import only when retraining is needed
    if parametric_umap:
      from umap.parametric_umap import ParametricUMAP as UMAP
    else:
      from umap import UMAP

    self.reducer_ = UMAP(n_neighbors=15,
                         n_components=5,
                         metric='euclidean',
                         min_dist=0.0,
                         verbose=self.verbose,
                         n_jobs=-1)

    self.clusterer_ = HDBSCAN(min_cluster_size=100,
                              min_samples=1,
                              metric='euclidean',
                              core_dist_n_jobs=-1,
                              memory=joblib.Memory(location='tmp'),
                              prediction_data=True)

  def fit_transform(self, X, y=None, umap_embeddings=None):
    """Reduce the dimensionality of the input and then apply soft-clustering.

    Returns:
      clusters: array of assigned cluster labels to each data point.
      weights: array of cluster membership weights (soft clusters where applying argmax
               equals `clusters`).
    """

    self.umap_embeddings_ = umap_embeddings

    if self.umap_embeddings_ is None:
      self.umap_embeddings_ = self.reducer_.fit_transform(X, y)
      self.umap_embeddings_ = np.nan_to_num(self.umap_embeddings_)
      self.verbose and print('[TopicModel] Reduced embeddings dimension. Now clustering...')

    clusters = self.clusterer_.fit_predict(self.umap_embeddings_)
    self.verbose and print('[TopicModel] Clustered embeddings. Now computing membership weights...')

    # weights = self.clusterer.probabilities_
    weights = hdbscan.all_points_membership_vectors(self.clusterer_)

    # commented because of soft clustering; we don't want to zero out weights for noisy data
    # weights[clusters < 0] = 0.0

    self.verbose and print('[TopicModel] Done!')

    return clusters, weights
