
def fit_parametric_umap(X, y=None, name=None, parametric=True):
  """reduce dimensionality with UMAP"""

  import numpy as np
  from pathlib import Path

  cached_file = name or Path('models/umap/abstracts_embeddings_USE-v4-5d.npz')

  if cached_file.exists():
    # load the embedding from cached file
    embeddings = np.load(cached_file)['arr_0']
  else:
    # importing UMAP takes a while, so we import only when retraining is needed
    if parametric:
      from umap.parametric_umap import ParametricUMAP as UMAP
    else:
      from UMAP import UMAP

    reducer = UMAP(n_neighbors=15,
                   n_components=5,
                   min_dist=0.0,
                   metric='cosine',
                   verbose=True,
                   n_jobs=1,
                   low_memory=True)

    embeddings = reducer.fit_transform(X, y)
    embeddings = np.nan_to_num(embeddings)

  return embeddings
