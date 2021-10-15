# %% 1. Load the data and define the analysis

#!pip install --upgrade git+https://github.com/scikit-learn-contrib/hdbscan

import os, sys
import argparse
from datetime import datetime
from collections import namedtuple
from pathlib import Path

import pandas as pd
import numpy as np

from bertopic import BERTopic

from umap.parametric_umap import ParametricUMAP
import hdbscan


# PARAMETERS
MODELS_DIR = Path('models/')
DATASET_NAME = 'pubmed_abstracts_preprocessed'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEVICE = 'cpu'


BERTopicResult = namedtuple('BERTopicResult', ['model', 'indices', 'topics', 'scores'])
"""This is a handy container to store fitted BERTopic results."""

Top2VecResult = namedtuple('Top2VecResult', ['model', 'data', 'scores'])
"""Handy container to store fitted Top2Vec results (with doc2vec embedding)."""


def fit_bertopic(
    df: pd.DataFrame,
    embedding_model_name: str,
    fraction: float = 1.0,
    device: str = 'cpu'
) -> BERTopicResult:

  embedding_file = MODELS_DIR / f'{DATASET_NAME}_{EMBEDDING_MODEL}_{DEVICE}.embeddings.npz'

  # sample dataset
  if fraction < 1.0:
    df = df.groupby('label', group_keys=False).apply(
        lambda grp: grp.sample(n=max(int(len(grp) * fraction), 1))
    )

  # load doc embedding
  with np.load(embedding_file) as fp:
    embeddings = fp['arr_0']
    embeddings = embeddings[df.index, :]

  print(f'Fitting {int(fraction*100)}% of the dataset ({embeddings.shape[0]} documents)...')

  # input and output (X and y)
  X = df['abstract'].values
  y = df['label'].astype('category').cat.codes

  # UMAP
  umap_model = ParametricUMAP(
      n_neighbors=15,
      n_components=5,
      min_dist=0.0,
      metric='cosine',
      low_memory=False)

  # define the model
  model = BERTopic(
      calculate_probabilities=False,
      n_gram_range=(1, 3),
      embedding_model=embedding_model_name,
      umap_model=umap_model,
      verbose=True)

  # fit the model
  topics, scores = model.fit_transform(
      documents=X, y=y,
      embeddings=embeddings
  )

  return BERTopicResult(model, df.index.values, topics, scores)


def fit_top2vec(df: pd.DataFrame):
  from top2vec import Top2Vec

  _df = df.drop_duplicates(subset=['pmid']).copy()
  abstracts = _df['abstract'].to_list()
  pmids = _df['pmid'].to_list()
  # DEBUG labels = df[['category', 'label']].astype('category')

  model = Top2Vec(
      abstracts,
      document_ids=pmids,
      embedding_model='doc2vec',
      speed='deep-learn',
      workers=os.cpu_count() - 1,
      verbose=True
  )

  scores = model.get_documents_topics(model.document_ids, num_topics=model.get_num_topics())[1]

  return Top2VecResult(model, df, scores)


def calc_bertopic_scores(model: BERTopic) -> np.array:
  scores = hdbscan.all_points_membership_vectors(model.hdbscan_model)
  return scores


def save_top2vec(result: Top2VecResult, name='pubmed_top2vec', root=Path('models/')):
  from top2vec import Top2Vec

  version = datetime.now().strftime('%Y%m%d')
  version_iter = 1
  while (root / f'{name}_v{version}{version_iter}.model').exists():
    version_iter += 1

  result.model.save(root / f'{name}_v{version}{version_iter}.model')
  np.savez(root / f'{name}_v{version}{version_iter}.idx', result.data.index.values)
  np.savez(root / f'{name}_v{version}{version_iter}.scores', result.scores)


def save_bertopic(result: BERTopicResult, name='pubmed_bertopic', root=Path('models/'), device='cpu'):

  version = datetime.now().strftime('%Y%m%d')
  version_iter = 1
  while (root / f'{name}_v{version}{version_iter}.model').exists():
    version_iter += 1

  result.model.save(root / f'{name}_v{version}{version_iter}_{device}.model')
  np.savez(root / f'{name}_v{version}{version_iter}_{device}.idx', result.indices)
  np.savez(root / f'{name}_v{version}{version_iter}_{device}.topics', result.topics)
  np.savez(root / f'{name}_v{version}{version_iter}_{device}.probs', result.scores)

  return f'{name}_v{version}{version_iter}'


def main():

  # CLI ARGUMENTS
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--fraction', type=float, default='0.01')
  parser.add_argument('--top2vec', dest='enable_top2vec', action='store_true')
  parser.add_argument('--bertopic', dest='enable_bertopic', action='store_true')
  args = vars(parser.parse_args())
  data_fraction = args['fraction']
  enable_top2vec = args['enable_top2vec']
  enable_bertopic = args['enable_bertopic'] # or a faster model: 'paraphrase-MiniLM-L3-v2'

  # DATA
  PUBMED = pd.read_csv(f'data/{DATASET_NAME}.csv.gz')
  PUBMED = PUBMED.dropna(subset=['abstract']).reset_index()
  PUBMED = PUBMED.rename(columns={'subcategory': 'label'}, errors='ignore')

  # DEBUG: discard low-appeared tasks/constructs
  # valid_subcats = PUBMED['label'].value_counts()[lambda cnt: cnt > 1].index.to_list() # noqa
  # PUBMED = PUBMED.query('label in @valid_subcats')

  # DEBUG
  print('# of tasks and constructs:\n', PUBMED.groupby('category')['label'].nunique())

  # Now run the model fitting, and then store the model, embedding, and probabilities.
  if enable_top2vec:
    t2v_result = fit_top2vec(PUBMED)
    save_top2vec(t2v_result, name=f'pubmed{int(100*data_fraction)}pct_top2vec', root=MODELS_DIR)

  if enable_bertopic:
    brt_result = fit_bertopic(PUBMED, EMBEDDING_MODEL, data_fraction, DEVICE)
    model_name = save_bertopic(
      brt_result,
      f'pubmed{int(100*data_fraction)}pct_bertopic',
      root=MODELS_DIR,
      device=DEVICE)

    print('BERTopic modeling completed. Now calculating doc2topic scores...')
    bertopic_scores = calc_bertopic_scores(brt_result.model)
    np.savez(MODELS_DIR / f'{model_name}_{DEVICE}.scores', bertopic_scores)

  print('Finished!')


if __name__ == '__main__':
  main()
