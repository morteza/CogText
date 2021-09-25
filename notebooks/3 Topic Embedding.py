# %%
import os
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# ====================
# parameters
# ====================
DATA_SAMPLE_FRACTION = .05

# Fix transformers bug when nprocess>1
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# load data
PUBMED = pd.read_csv('data/pubmed_abstracts_preprocessed.csv.gz').dropna(subset=['abstract'])


# select a fraction of data to speed up development
if DATA_SAMPLE_FRACTION < 1.:
  PUBMED = PUBMED.groupby('subcategory').apply(lambda sc: sc.sample(frac=DATA_SAMPLE_FRACTION)).reset_index(drop=True)

print('# of tasks and constructs:\n', PUBMED.groupby('category').apply(lambda x: x['subcategory'].nunique()))


# discard low-appeared tasks/constructs
valid_subcats = PUBMED['subcategory'].value_counts()[lambda cnt: cnt > 3].index.to_list()
PUBMED = PUBMED.query('subcategory in @valid_subcats')


# prep input (X) and output (y)
X = PUBMED['abstract'].values
y = PUBMED[['category', 'subcategory']].astype('category')


# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, stratify=y['subcategory'])

assert y_train['subcategory'].nunique() == y_test['subcategory'].nunique()


# ====================
# training
# ====================
# custom sentence embedding, defaults to paraphrase-MiniLM-L6-v2
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = sentence_model.encode(X_train, show_progress_bar=True)


topic_model = BERTopic(verbose=True, calculate_probabilities=True)
H_train_topics, H_train_probs = topic_model.fit_transform(
    X_train, y=y_train['subcategory'].cat.codes,)  # embeddings=embeddings,)

# topics_per_class = topic_model.topics_per_class(
#     X_train,
#     topics,
#     classes=y_train['subcategory'])

# topic_model.visualize_topics_per_class(topics_per_class)

H_train = pd.DataFrame(H_train_probs)
H_train['subcategory'] = y_train['subcategory'].values
H_train = H_train.groupby('subcategory').mean()


# ====================
# evaluating the test split
# ====================
H_test_topics, H_test_probs = topic_model.transform(X_test)

H_test = pd.DataFrame(H_test_probs)
H_test['subcategory'] = y_test['subcategory'].values
H_test = H_test.groupby('subcategory').mean()


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


# visualize similarity map on the trains set
plot_pca_projection(H_train)
plot_joint_similarity_map(
    embedding=H_train,
    y=y_train,
    title='Topics embedding similarity (train set)')


# visualize similarity map on the test set
plot_pca_projection(H_test)
plot_joint_similarity_map(
    embedding=H_test,
    y=y_test,
    title='Topics embedding similarity (train set)')


# now store the model and probabilities

def save_topic_model(model, probs, name='pubmed_bertopic'):
  """[Summary]
  
  models naming: <dataset>_<model>_v<version>.model

  Args:
      model (BERTopic): [description]
  """
  version = datetime.now().strftime('%Y%m%d%H')

  model.save(f'outputs/models/{name}_v{version}.model')
  np.savez(f'outputs/models/{name}_v{version}.probs', probs)


save_topic_model(
    topic_model,
    H_train_probs,
    f'pubmed{int(100*DATA_SAMPLE_FRACTION)}pct_bertopic'
)

# TODO fit n_topics given train/test split
# TODO use train/test cosine similarity as a measure of performance
# TODO discard abstracts with plasmid and genetic-related topics
# TODO: annotate topics as relevant/irrelevant
# TODO: automatically mark documents by using their assigned topics
