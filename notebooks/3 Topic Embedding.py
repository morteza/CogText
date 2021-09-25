# %% 1. Load the data and define the analysis
import os
from datetime import datetime
from collections import namedtuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# ====================
# parameters
# ====================
DATA_SAMPLE_FRACTION = .01

# Fix transformers bug when nprocess > 1
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# load data
PUBMED = pd.read_csv('data/pubmed_abstracts_preprocessed.csv.gz').dropna(subset=['abstract'])

# select a fraction of data to speed up development
PUBMED = PUBMED.groupby('subcategory').sample(frac=DATA_SAMPLE_FRACTION)

# discard low-appeared tasks/constructs
valid_subcats = PUBMED['subcategory'].value_counts()[lambda cnt: cnt > 3].index.to_list() # noqa
PUBMED = PUBMED.query('subcategory in @valid_subcats')

print('# of tasks and constructs:\n', PUBMED.groupby('category')['subcategory'].nunique())


TopicModelResult = namedtuple('TopicModelResult', [
    'model', 'X_train', 'X_test', 'y_train', 'y_test', 'H_train', 'H_test',
    'H_train_topics', 'H_train_probs', 'H_test_topics', 'H_test_probs'
])
"""This is a handy container to store model fitting results"""


def fit_topic_embedding(df: pd.DataFrame) -> TopicModelResult:

  # prep input (X) and output (y)
  X = df['abstract'].values
  y = df[['category', 'subcategory']].astype('category')

  # train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, stratify=y['subcategory'])
  assert y_train['subcategory'].nunique() == y_test['subcategory'].nunique()

  # custom sentence embedding, defaults to paraphrase-MiniLM-L6-v2
  # from sentence_transformers import SentenceTransformer
  # sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
  # embeddings = sentence_model.encode(X_train, show_progress_bar=True)

  topic_model = BERTopic(verbose=True, calculate_probabilities=True)

  H_train_topics, H_train_probs = topic_model.fit_transform(
      X_train, y=y_train['subcategory'].cat.codes,)  # embeddings=embeddings,)
  H_train = pd.DataFrame(H_train_probs)
  H_train['subcategory'] = y_train['subcategory'].values
  H_train = H_train.groupby('subcategory').mean()

  H_test_topics, H_test_probs = topic_model.transform(X_test)
  H_test = pd.DataFrame(H_test_probs)
  H_test['subcategory'] = y_test['subcategory'].values
  H_test = H_test.groupby('subcategory').mean()

  return TopicModelResult(
      topic_model, X_train, X_test, y_train, y_test, H_train, H_test,
      H_train_topics, H_train_probs, H_test_topics, H_test_probs)


# ====================
# visualization
# ====================
def plot_pca_projection(embedding):
  """project the embedding to 3D space and visualize it."""
  pca = PCA(n_components=3)
  X_proj = pca.fit_transform(embedding).T

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(X_proj[0], X_proj[1], X_proj[2])

  ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')

  for i, label in enumerate(embedding.index):
    if np.max(X_proj[:, i]) > .2:
      ax.text(X_proj[0, i], X_proj[1, i], X_proj[2, i], label)

  plt.show()


def plot_joint_similarity_map(embedding, y, title):
  H_sim = pd.DataFrame(cosine_similarity(embedding), columns=embedding.index, index=embedding.index)

  cats = y.groupby(['subcategory'])['category'].apply(lambda x: x.unique()[0] if x.nunique() > 0 else '-')
  _palette = dict(zip(cats.unique(), sns.color_palette('Accent', cats.nunique())))
  cat_colors = cats.apply(lambda x: _palette[x])

  g = sns.clustermap(
      H_sim, metric='cosine', lw=1, cmap='RdBu', figsize=(50, 50),
      col_colors=cat_colors, row_colors=cat_colors)
  g.ax_row_dendrogram.set_visible(False)
  g.ax_col_dendrogram.set_visible(False)
  plt.suptitle(title)
  plt.show()


def save_result(result: TopicModelResult, name='pubmed_bertopic'):
  """Save topic modeling results and weights

  models naming: <dataset>_<model>_v<version>.model

  Args:
      model (BERTopic): [description]
  """
  version = datetime.now().strftime('%Y%m%d%H')

  result.model.save(f'outputs/models/{name}_v{version}.model')
  np.savez(f'outputs/models/{name}_v{version}.train_probs', result.H_train_probs)
  # TODO store test probs and test/train splits


# %% 2. Now run the model fitting


result = fit_topic_embedding(PUBMED)

# visualize similarity map on the trains set
plot_pca_projection(result.H_train)
plot_joint_similarity_map(
    embedding=result.H_train,
    y=result.y_train,
    title='Topics embedding similarity (train set)')

# visualize similarity map on the test set
plot_pca_projection(result.H_test)
plot_joint_similarity_map(
    embedding=result.H_test,
    y=result.y_test,
    title='Topics embedding similarity (train set)')

# now store the model and probabilities
save_result(result, f'pubmed{int(100*DATA_SAMPLE_FRACTION)}pct_bertopic')

# TODO fit n_topics given train/test split
# TODO use train/test cosine similarity as a measure of performance
# TODO discard abstracts with plasmid and genetic-related topics
# TODO: annotate topics as relevant/irrelevant
# TODO: automatically mark documents by using their assigned topics

# %% test/train RSA
from scipy.stats import spearmanr

sim_train = cosine_similarity(result.H_train)
sim_test = cosine_similarity(result.H_test)
rho = spearmanr(sim_train, sim_test)
print(f'[RSA] mean test/train correlation: {rho[0].mean():.2f}')
