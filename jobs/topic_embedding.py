# %% 1. Load the data and define the analysis
import os
from datetime import datetime
from collections import namedtuple
from pathlib import Path

import pandas as pd
import numpy as np

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# PARAMETERS
# set the following env var to fit only a fraction of the dataset: COGTEXT_SAMPLE_FRACTION
DATA_SAMPLE_FRACTION = float(os.getenv('COGTEXT_SAMPLE_FRACTION', '.01'))
EMBEDDINGS_CACHE_PATH = Path('data/tmp/pubmed_abstract_embedding.npz')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

print(f'Fitting {int(DATA_SAMPLE_FRACTION*100)}% of the PUBMED dataset...')

# Fix transformers bug when nprocess > 1 (commented because it's automatically set by the .env in Morty's local env)
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# load data
PUBMED = pd.read_csv('data/pubmed_abstracts.csv.gz').dropna(subset=['abstract'])

# select a fraction of data to speed up development
PUBMED = PUBMED.groupby('subcategory').sample(frac=DATA_SAMPLE_FRACTION)

# discard low-appeared tasks/constructs
valid_subcats = PUBMED['subcategory'].value_counts()[lambda cnt: cnt > 3].index.to_list() # noqa
PUBMED = PUBMED.query('subcategory in @valid_subcats')

print('# of tasks and constructs:\n', PUBMED.groupby('category')['subcategory'].nunique())


TopicModelResult = namedtuple('TopicModelResult', [
    'model', 'X', 'y', 'embedding',
    'Z', 'Z_topics', 'Z_probs'
])
"""This is a handy container to store model fitting results"""


def fit_topic_embedding(
    df: pd.DataFrame,
    embedding_model=EMBEDDING_MODEL,
    embedding_cache_path=EMBEDDINGS_CACHE_PATH
) -> TopicModelResult:

  # prep input and output (X and y)
  X = df['abstract'].values
  y = df[['category', 'subcategory']].astype('category')

  # TODO keep track of pmids in the `y` for future references

  # custom sentence embedding, defaults to all-MiniLM-L6-v2
  sentence_model = SentenceTransformer(embedding_model)

  # DEBUG: cache embeddings to speed things up by 4h
  if (embedding_cache_path is not None) and embedding_cache_path.exists():
    with np.load(embedding_cache_path) as _fp:
      embeddings = _fp['arr_0']
  else:
    embeddings = sentence_model.encode(X, show_progress_bar=True)
    np.savez(embedding_cache_path, embeddings)

  # define the model
  topic_model = BERTopic(
      low_memory=True,
      calculate_probabilities=True,
      n_gram_range=(1, 3),
      embedding_model=sentence_model,
      verbose=True)

  # fit the model
  Z_topics, Z_probs = topic_model.fit_transform(
      documents=X,
      y=y['subcategory'].cat.codes,
      embeddings=embeddings)

  # topic embeddings by subcategory
  Z = pd.DataFrame(Z_probs)
  Z['subcategory'] = y['subcategory'].values
  Z = Z.groupby('subcategory').mean()

  topic_model.find_topics('task')

  return TopicModelResult(
      topic_model, X, y, embeddings, Z, Z_topics, Z_probs
  )


def save_result(result: TopicModelResult, name='pubmed_bertopic'):
  """Save topic modeling results and weights

  models naming: <dataset>_<model>_v<version>.model

  Args:
      model (BERTopic): [description]
  """
  root = Path('outputs/models/')

  version = datetime.now().strftime('%Y%m%d')
  version_iter = 1
  while (root / f'{name}_v{version}{iter}.model').exists():
    version_iter += 1

  result.model.save(root / f'{name}_v{version}{version_iter}.model')
  np.savez(root / f'{name}_v{version}{version_iter}.probs', result.Z_probs)
  np.savez(root / f'{name}_v{version}{version_iter}.embeddings', result.embedding)


# Now run the model fitting, and then store the model, embedding, and probabilities.
result = fit_topic_embedding(PUBMED)
save_result(result, f'pubmed{int(100*DATA_SAMPLE_FRACTION)}pct_bertopic')
