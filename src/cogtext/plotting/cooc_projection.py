import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

import hdbscan

from .. import svd_embedding


def plot_cooc_projection(
    cooc: pd.DataFrame,
    embedding_dim=-1,
    kind='pca',
    normalize=True,
    title='',
    ax=None
):
    """Transform the co-occurrence matrix to an embedding space and plot its 2D projection.

    Parameters
    ----------
    cooc (`pd.DataFrame`): co-occurrence matrix.
    embedding_dim (`int`, optional): Number of embedding dimensions.
        Defaults to -1 which skips embedding.
    kind (str, optional): Either `pca`, `umap`, or 'tsne'.
        Defaults to `pca`.
    normalize (bool, optional): If set, first normalizes the input using
        `StandardScaler`. Defaults to True.
    title (str, optional): Subplot title.
        Defaults to ''.
     ax ([type], optional): Matplotlib subplot Axes to draw on.
        Defaults to `None`.
    """

    if ax is None:
      ax = plt.gca()

    cooc_transformed = cooc

    # 1. normalized and compress using svd; only compress if embedding_dim > 0
    if normalize:
        cooc_transformed = StandardScaler().fit_transform(cooc)

    # 2. compress and reconstruct the matrix using SVD
    if embedding_dim > 0:
        _, _, _, cooc_transformed = svd_embedding(cooc_transformed, embedding_dim)

    # 3. now project to 2D using tsne, UMAP, or PCA. Defaults to PCA.
    if kind.lower() == 'tsne':
        cooc_transformed = TSNE(n_components=2).fit_transform(cooc_transformed)
    elif kind.lower() == 'umap':
        cooc_transformed = umap.UMAP(n_components=2, random_state=0).fit_transform(cooc_transformed)
    else:
        cooc_transformed = PCA(n_components=2).fit_transform(cooc_transformed)

    # 4. cluster points in the transformed space (2D)
    cluster = hdbscan.HDBSCAN(
        min_samples=3,
        min_cluster_size=3,
    ).fit_predict(cooc_transformed).reshape(-1, 1)
    cooc_transformed = np.hstack((cooc_transformed, cluster))

    # 5. plot points and cluster colors
    sns.scatterplot(
        data=pd.DataFrame(cooc_transformed, columns=['comp1', 'comp2', 'cluster']),
        x='comp1', y='comp2', hue='cluster',
        color='r', ax=ax)

    # 6. plot labels
    for lbl, (x, y, c) in zip(cooc.index.to_list(), cooc_transformed):
        ax.text(x + 0.005, y + 0.0005, lbl, alpha=0.1)

    # 7. title
    ax.set(title=f'{title}: {kind.upper()} projection of the co-occurrences\n(embedding_dim={embedding_dim})')
