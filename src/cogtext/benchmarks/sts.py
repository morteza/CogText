# %%
import pandas as pd
import scipy.stats
import math
import csv
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import torch
import sys

sys.path.append('./python/')  # noqa
from cogtext.embeddings import BERTopicEmbedder  # noqa


def _encode_and_score(model, sts_data):
  """Encode the sentences in the STS dataset and return the similarity scores.
  """
  if isinstance(model, BERTopicEmbedder):
    sent1 = sts_data['sentence_1'].tolist()
    sent2 = sts_data['sentence_2'].tolist()
    sents = sent1 + sent2
    embeddings = model(sents)
    sts_encode1 = embeddings[:len(sent1)]
    sts_encode2 = embeddings[len(sent2):]
  else:
    sts_encode1 = model(sts_data['sentence_1'].values)
    sts_encode2 = model(sts_data['sentence_2'].values)

  if isinstance(model, torch.nn.Module):
    scores = torch.nn.functional.cosine_similarity(torch.tensor(sts_encode1),
                                                   torch.tensor(sts_encode2),
                                                   dim=1)
  else:
    sts_encode1_norm = tf.nn.l2_normalize(sts_encode1)
    sts_encode2_norm = tf.nn.l2_normalize(sts_encode2)

    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1_norm, sts_encode2_norm), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi

  return scores


def run_sts_benchmark(model, sts_subset='train'):
  """Perform STS benchmark on document embeddings.

  Args:
      model ([type]): Document embedding model with cosine-similarity support and an encode() function
      data_subset (str, optional): [description]. Defaults to 'test'.

  Returns:
      (float,float): Pearson's r coefficient and p-value as returned by scipy.stats.pearsonr(...).
  """

  sts_dataset = tf.keras.utils.get_file(
      fname='Stsbenchmark.tar.gz',
      origin='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz',
      extract=True)

  sts_data = pd.read_table(
      os.path.join(os.path.dirname(sts_dataset), 'Stsbenchmark', f'sts-{sts_subset}.csv'),
      quoting=csv.QUOTE_NONE,
      skip_blank_lines=True,
      usecols=[4, 5, 6], names=['similarity', 'sentence_1', 'sentence_2'])

  sts_data.dropna(subset=['sentence_2'], inplace=True)

  scores = []

  for batch in tqdm(np.array_split(sts_data, 10)):
    scores.extend(_encode_and_score(model, batch))

  r, p = scipy.stats.pearsonr(scores, sts_data['similarity'])

  return r, p


if __name__ == '__main__':

  models = [
      BERTopicEmbedder('all-mpnet-base-v2'),
      BERTopicEmbedder('all-distilroberta-v1')
      # SBertEmbedder('all-mpnet-base-v2'),
      # SBertEmbedder('all-distilroberta-v1'),
      # Top2VecEmbedder(),
      # TensorFlowHubEmbedder('universal-sentence-encoder-large/5', show_progress_bar=False),
      # Word2VecEmbedder(),
      # Doc2VecEmbedder(),
      # AverageDistilBert(),
  ]

  for model in models:
    results = run_sts_benchmark(model, 'test')
    print(f'{model}\n'
          f'Pearson coef={results[0]} (p-value={results[1]})')

    # clean up CPU/GPU memory
    model = None
    del model
