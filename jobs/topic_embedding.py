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
DATA_FRACTION = float(os.getenv('COGTEXT_DATA_FRACTION', '0.1'))
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # or a faster model: 'paraphrase-MiniLM-L3-v2'
CACHE_DIR = 'data/.cache/'

# init folders if they do not exist yet.
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path('outputs/models').mkdir(parents=True, exist_ok=True)

print(f'Fitting {int(DATA_FRACTION*100)}% of the PUBMED dataset...')

# load data
PUBMED = pd.read_csv('data/pubmed_abstracts_preprocessed.csv.gz').dropna(subset=['abstract'])

# select a fraction of data to speed up development
PUBMED = PUBMED.groupby('label').sample(frac=DATA_FRACTION)

# discard low-appeared tasks/constructs
valid_subcats = PUBMED['label'].value_counts()[lambda cnt: cnt > 3].index.to_list() # noqa
PUBMED = PUBMED.query('label in @valid_subcats')

print('# of tasks and constructs:\n', PUBMED.groupby('category')['label'].nunique())


TopicModelResult = namedtuple('TopicModelResult', ['model', 'data', 'topics', 'probs'])
"""This is a handy container to store fitted results."""


def fit_topic_embedding(
    df: pd.DataFrame,
    embedding_model=EMBEDDING_MODEL,
    cache_dir: str = CACHE_DIR
) -> TopicModelResult:

  # prep input and output (X and y)
  X = df['abstract'].values
  y = df[['category', 'label']].astype('category')

  # TODO keep track of pmids in the `y` for future references

  # custom sentence embedding
  sentence_model = SentenceTransformer(embedding_model)

  embeddings_file = Path(CACHE_DIR) / 'pubmed_abstracts_embeddings.npz'

  # cache embeddings to speed things up to the UMAP process
  if (embeddings_file is not None) and embeddings_file.exists():
    with np.load(embeddings_file) as fp:
      embeddings = fp['arr_0']
  else:
    embeddings = sentence_model.encode(X, show_progress_bar=True)
    np.savez(embeddings_file, embeddings)

  # define the model
  topic_model = BERTopic(
      # FIXME low_memory=True,
      calculate_probabilities=True,
      n_gram_range=(1, 3),
      embedding_model=sentence_model,
      verbose=True)

  # fit the model
  topics, probs = topic_model.fit_transform(
      documents=X,
      y=y['label'].cat.codes,
      embeddings=embeddings)

  return TopicModelResult(
      topic_model, df, topics, probs
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
  while (root / f'{name}_v{version}{version_iter}.model').exists():
    version_iter += 1

  result.model.save(root / f'{name}_v{version}{version_iter}.model')
  np.savez(root / f'{name}_v{version}{version_iter}.topics', result.topics)
  np.savez(root / f'{name}_v{version}{version_iter}.probs', result.probs)
  np.savez(root / f'{name}_v{version}{version_iter}.idx', result.data.index.values)


# Now run the model fitting, and then store the model, embedding, and probabilities.
result = fit_topic_embedding(PUBMED)
save_result(result, f'pubmed{int(100*DATA_FRACTION)}pct_bertopic')

print('Finished!')
