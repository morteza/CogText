def fit_hdbscan(X, y=None):
  """Cluster documents with HDBSCAN (output: cluster labels, membership scores)
  """

  from hdbscan import HDBSCAN
  import joblib

  joblib_cache = joblib.Memory(location='tmp')

  clusterer = HDBSCAN(min_cluster_size=15,
                      # min_samples=1,
                      metric='euclidean',
                      core_dist_n_jobs=1,
                      memory=joblib_cache,
                      prediction_data=True)

  clusters = clusterer.fit_predict(X)
  scores = clusterer.probabilities_

  return clusters, scores
