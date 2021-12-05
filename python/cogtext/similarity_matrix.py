import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity_matrix(H, metric='cosine', pivot_by_category=True):
  assert metric in ['cosine'], 'Invalid similarity metric'

  H_sim = pd.DataFrame(cosine_similarity(H),
                       index=H.index.get_level_values('label'),
                       columns=H.index.get_level_values('label'))

  if pivot_by_category:
    tasks = H.query('index.get_level_values("category").str.contains("Task")'
                    ).index.get_level_values('label').unique()
    constructs = H.query('index.get_level_values("category").str.contains("Construct")'
                         ).index.get_level_values('label').unique()
    categorized_df = H_sim.drop(index=tasks, columns=constructs)
    return categorized_df

  return H_sim
