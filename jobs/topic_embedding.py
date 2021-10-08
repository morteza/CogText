# %% 1. Load the data and define the analysis
import os
from datetime import datetime
from collections import namedtuple
from pathlib import Path

import pandas as pd
import numpy as np

from bertopic import BERTopic
from top2vec import Top2Vec
from sentence_transformers import SentenceTransformer


# PARAMETERS
# set the following env var to fit only a fraction of the dataset: COGTEXT_DATA_FRACTION
DATA_FRACTION = float(os.getenv('COGTEXT_DATA_FRACTION', '0.01'))
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # or a faster model: 'paraphrase-MiniLM-L3-v2'
CACHE_DIR = 'data/.cache/'

# load data
PUBMED = pd.read_csv('data/pubmed_abstracts_preprocessed.csv.gz').dropna(subset=['abstract'])

# init folders if they do not exist yet.
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path('outputs/models').mkdir(parents=True, exist_ok=True)

print(f'Fitting {int(DATA_FRACTION*100)}% of the PUBMED dataset...')


if 'label' not in PUBMED.columns:
  PUBMED['label'] = PUBMED['subcategory']

# select a fraction of data to speed up development
if DATA_FRACTION < 1.0:
  PUBMED = PUBMED.groupby('label').sample(frac=DATA_FRACTION)

# discard low-appeared tasks/constructs
valid_subcats = PUBMED['label'].value_counts()[lambda cnt: cnt > 3].index.to_list() # noqa
PUBMED = PUBMED.query('label in @valid_subcats')

print('# of tasks and constructs:\n', PUBMED.groupby('category')['label'].nunique())


BERTopicResult = namedtuple('BERTopicResult', ['model', 'data', 'topics', 'probs'])
"""This is a handy container to store fitted BERTopic results."""

Top2VecResult = namedtuple('Top2VecResult', ['model', 'data', 'scores'])
"""Handy container to store fitted Top2Vec results (with doc2vec embedding)."""


def fit_bertopic(
    df: pd.DataFrame,
    embedding_model=EMBEDDING_MODEL,
    cache_dir: str = CACHE_DIR
) -> BERTopicResult:

  # prep input and output (X and y)
  X = df['abstract'].values
  y = df[['category', 'label']].astype('category')

  # custom sentence embedding
  sentence_model = SentenceTransformer(embedding_model)

  embeddings_file = Path(cache_dir) / 'pubmed_abstracts_embeddings.npz'

  # cache embeddings to speed things up to the UMAP step
  if embeddings_file.exists():
    print('Loading sentence embeddings from cache...')
    with np.load(embeddings_file) as fp:
      embeddings = fp['arr_0']
  else:
    embeddings = sentence_model.encode(X, show_progress_bar=True)
    np.savez(embeddings_file, embeddings)

  # define the model
  topic_model = BERTopic(
      calculate_probabilities=True,
      n_gram_range=(1, 3),
      embedding_model=sentence_model,
      verbose=True)

  # fit the model
  topics, probs = topic_model.fit_transform(
      documents=X,
      y=y['label'].cat.codes,
      embeddings=embeddings)

  return BERTopicResult(topic_model, df, topics, probs)


def fit_top2vec(df: pd.DataFrame):

  docs = df['abstract'].to_list()
  # doc_ids = df['abstract'].to_list()
  # labels = df[['category', 'label']].astype('category')

  model = Top2Vec(docs, workers=os.cpu_count() - 1, speed='deep-learn', verbose=True)
  scores = model.get_documents_topics(model.document_ids, num_topics=model.get_num_topics())[1].shape

  return Top2VecResult(model, df, scores)


def save_top2vec(result: Top2VecResult, name='pubmed_top2vec', root=Path('outputs/models/')):

  version = datetime.now().strftime('%Y%m%d')
  version_iter = 1
  while (root / f'{name}_v{version}{version_iter}.model').exists():
    version_iter += 1

  result.model.save(root / f'{name}_v{version}{version_iter}.model')
  np.savez(root / f'{name}_v{version}{version_iter}.idx', result.data.index.values)
  np.savez(root / f'{name}_v{version}{version_iter}.scores', result.scores)


def save_bertopic(result: BERTopicResult, name='pubmed_bertopic', root=Path('outputs/models/')):

  version = datetime.now().strftime('%Y%m%d')
  version_iter = 1
  while (root / f'{name}_v{version}{version_iter}.model').exists():
    version_iter += 1

  result.model.save(root / f'{name}_v{version}{version_iter}.model')
  np.savez(root / f'{name}_v{version}{version_iter}.topics', result.topics)
  np.savez(root / f'{name}_v{version}{version_iter}.probs', result.probs)
  np.savez(root / f'{name}_v{version}{version_iter}.idx', result.data.index.values)


# Now run the model fitting, and then store the model, embedding, and probabilities.
t2v_result = fit_top2vec(PUBMED, )
save_top2vec(t2v_result, name=f'pubmed{int(100*DATA_FRACTION)}pct_top2vec')

brt_result = fit_bertopic(PUBMED)
save_bertopic(brt_result, f'pubmed{int(100*DATA_FRACTION)}pct_bertopic')

print('Finished!')
