import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
  import tensorflow as tf  # noqa
  import tensorflow_probability as tfp
  tfd = tfp.distributions
except ImportError:
  print('WARNING: `tensorflow_probability` not installed. Cannot compute multivariate_normal_* metrics')


def multivariate_normal_js(p, q):
  P = tfd.MultivariateNormalDiag(p.mean(axis=0), scale=np.cov(p))
  Q = tfd.MultivariateNormalDiag(q.mean(axis=0), scale=np.cov(q))
  M = 0.5 * (P + Q)
  return (0.5 * P.kl_divergence(M).numpy() + 0.5 * Q.multivariate_normal_kl(M)).numpy()


def multivariate_normal_kl(p, q):
  P = tfd.MultivariateNormalDiag(p.mean(axis=0), scale=np.cov(p))
  Q = tfd.MultivariateNormalDiag(q.mean(axis=0), scale=np.cov(q))

  # alternatively use independent normals for each dimension
  # P = tfd.Normal(p.mean(axis=0), p.std(axis=0))
  # Q = tfd.Normal(q.mean(axis=0), q.std(axis=0))
  # return tfd.kl_divergence(P, Q)

  return P.kl_divergence(Q).numpy()
  # return tfp.vi.jensen_shannon(P, Q).numpy()



def categorical_kl(p, q):
  P = tfd.Categorical(probs=p)
  Q = tfd.Categorical(probs=q)
  return tfd.kl_divergence(P, Q)


def get_similarity_matrix(H, metric='cosine', pivot_by_category=True):
  assert metric in ['cosine', 'kl'], 'Invalid similarity metric'

  if metric.lower() == 'kl':
    H_sim = H.T.corr(method=categorical_kl)
    # reset KL of equal distributions to zero
    # np.fill_diagonal(H_sim.values, 0.)
    H_sim.index = H_sim.index.droplevel(level=1)
    H_sim.columns = H_sim.columns.droplevel(level=1)

  elif metric.lower() == 'cosine':
    H_sim = pd.DataFrame(
        cosine_similarity(H),
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
