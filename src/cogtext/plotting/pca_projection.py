
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def pca_projection(embedding: pd.DataFrame):
  """project the embedding to 3D space and visualize it.

  Example:
    ```
    plotting.pca_projection(result.Z)
    ```
  """

  pca = PCA(n_components=3)
  proj = pca.fit_transform(embedding).T

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(proj[0], proj[1], proj[2])

  ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')

  for i, label in enumerate(embedding.index):
    if np.max(proj[:, i]) > .2:
      ax.text(proj[0, i], proj[1, i], proj[2, i], label)

  # plt.show(block=False)
