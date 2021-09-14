# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# !python -m spacy download en_core_web_trf
# !python -m spacy download en_core_web_sm
import en_core_web_trf
import en_core_web_sm

from python.cogtext.preprocess_abstracts import preprocess_abstracts

sns.set()
DEV_MODE = True
DEV_MODE_MAX_CORPUS_SIZE = 20

nlp = en_core_web_sm.load()

# add additional stop words to the language model
CUSTOM_STOP_WORDS = ['study', 'task', 'test', 'performance']
for stop_word in CUSTOM_STOP_WORDS:
  lexeme = nlp.vocab[stop_word]
  lexeme.is_stop = True


df = pd.read_csv('data/pubmed_abstracts.csv.gz')

df.dropna(subset=['abstract'], inplace=True)

valid_subcats = df['subcategory'].value_counts()[lambda cnt: cnt > 3].index.to_list()
df = df.query('subcategory in @valid_subcats')


if DEV_MODE:
  small_subcats = df['subcategory'].value_counts()[lambda cnt: cnt < DEV_MODE_MAX_CORPUS_SIZE].index.to_list()
  df = df.query('subcategory in @small_subcats').copy()

print('# of tasks and constructs: ', df.groupby('category').apply(lambda x: x['subcategory'].nunique()))

# preprocess abstracts
df['abstract'] = preprocess_abstracts(df['abstract'].to_list(), nlp_model=nlp, extract_phrases=True)

X = df['abstract'].values
y = df[['category', 'subcategory']].astype('category')

# y = np.array([y[col].cat.codes.values for col in y.columns]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, stratify=y['subcategory'])

# custom sentence embedding (use SpaCy)
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = sentence_model.encode(X_train, show_progress_bar=True)

topic_model = BERTopic(verbose=True, calculate_probabilities=True)
H_train_topics, H_train_probs = topic_model.fit_transform(
    X_train,
    # embeddings=embeddings,
    y=y_train['subcategory'].cat.codes)

# topics_per_class = topic_model.topics_per_class(
#     X_train,
#     topics,
#     classes=y_train['subcategory'])

# topics_per_class
# topic_model.visualize_topics_per_class(topics_per_class)

H_train = pd.DataFrame(H_train_probs)
H_train['subcategory'] = y_train['subcategory'].values
H_train = H_train.groupby('subcategory').mean()

# project to 3D for visualization

def plot_pca_projection(embedding):
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


H_test_topics, H_test_probs = topic_model.transform(X_test)

H_test = pd.DataFrame(H_test_probs)
H_test['subcategory'] = y_test['subcategory'].values
H_test = H_test.groupby('subcategory').mean()

plot_pca_projection(H_train)
plot_pca_projection(H_test)

# TODO fit n_topics given train/test split
# TODO use train/test cosine similarity as a measure of performance
# TODO discard abstracts with plasmid and genetic-related topics

# plot joint similarity matrix (train set)
def plot_joint_similarity_map(embedding, y, title):
  H_sim = pd.DataFrame(cosine_similarity(embedding), columns=embedding.index, index=embedding.index)

  cats = y.groupby(['subcategory'])['category'].apply(lambda x: x.unique()[0])
  _palette = dict(zip(cats.unique(), sns.color_palette('Accent', cats.nunique())))
  cat_colors = cats.apply(lambda x: _palette[x])

  g = sns.clustermap(
      H_sim, metric='cosine', lw=1, cmap='RdBu', figsize=(15, 15),
      col_colors=cat_colors, row_colors=cat_colors)
  g.ax_row_dendrogram.set_visible(False)
  g.ax_col_dendrogram.set_visible(False)
  plt.suptitle(title)
  plt.show()


plot_joint_similarity_map(
    embedding=H_train,
    y=y_train,
    title='Topics embedding similarity (train set)')

plot_joint_similarity_map(
    embedding=H_test,
    y=y_test,
    title='Topics embedding similarity (test set)')
