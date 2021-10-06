import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib as plt
import seaborn as sns


def similarity_map(embedding, y, title):
  """[summary]

  Args:
      embedding ([type]): [description]
      y ([type]): [description]
      title ([type]): [description]

  Example:
    ```
    plotting.similarity_map(
      embedding=result.Z,
      y=result.y,
      title='Similarities in the topic embedding space')
    ```
  """

  Z_sim = pd.DataFrame(cosine_similarity(embedding), columns=embedding.index, index=embedding.index)

  cats = y.groupby(['label'])['category'].apply(lambda x: x.unique()[0] if x.nunique() > 0 else '-')
  _palette = dict(zip(cats.unique(), sns.color_palette('Accent', cats.nunique())))
  cat_colors = cats.apply(lambda x: _palette[x])
  w_fig = len(Z_sim) / 2

  g = sns.clustermap(
      Z_sim, metric='cosine', lw=1, cmap='RdBu', figsize=(w_fig, w_fig),
      col_colors=cat_colors, row_colors=cat_colors)
  g.ax_row_dendrogram.set_visible(False)
  g.ax_col_dendrogram.set_visible(False)
  plt.suptitle(title)
  # plt.show(block=False)
