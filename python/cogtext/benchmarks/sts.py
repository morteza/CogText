# %%
import pandas
import scipy.stats
import math
import csv
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import torch


def _encode_and_score(model, sts_data):
  sts_encode1 = model.encode(sts_data['sentence_1'].values)
  sts_encode2 = model.encode(sts_data['sentence_2'].values)

  if isinstance(model, torch.nn.Module):
    scores = torch.nn.functional.cosine_similarity(torch.tensor(sts_encode1), torch.tensor(sts_encode2), dim=1)
  else:
    sts_encode1_norm = tf.nn.l2_normalize(sts_encode1)
    sts_encode2_norm = tf.nn.l2_normalize(sts_encode2)

    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1_norm, sts_encode2_norm), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi

  return scores


def run_sts_benchmark(model, data_subset='test'):
  """Perform STS benchmark on document embeddings.

  Args:
      model ([type]): Document embedding model with cosine-similarity support and an encode() function
      data_subset (str, optional): [description]. Defaults to 'test'.

  Returns:
      (float,float): Pearson's r coefficient and p-value as provided by scipy.stats.pearsonr(...).
  """

  sts_dataset = tf.keras.utils.get_file(
      fname='sts.tar.gz',
      origin='http://alt.qcri.org/semeval2016/task1/data/uploads/sts2016-english-v1.1.tar.gz',
      extract=True)

  if 'test' in data_subset:
    sts_data = pandas.read_table(
        os.path.join(os.path.dirname(sts_dataset), 'stsbenchmark', 'sts-test.csv'),
        quoting=csv.QUOTE_NONE,
        skip_blank_lines=True,
        usecols=[4, 5, 6], names=['similarity', 'sentence_1', 'sentence_2'])
  else:
    sts_data = pandas.read_table(
        os.path.join(os.path.dirname(sts_dataset), 'stsbenchmark', 'sts-dev.csv'),
        skip_blank_lines=True,
        usecols=[4, 5, 6], names=['similarity', 'sentence_1', 'sentence_2'])
    # cleanup missing sentences
    sts_data.dropna(subset=['sentence_2'], inplace=True)

  scores = []

  for batch in tqdm(np.array_split(sts_data, 10)):
    scores.extend(_encode_and_score(model, batch))

  r, p = scipy.stats.pearsonr(scores, sts_data['similarity'])

  return r, p


# if __name__ == '__main__':
#   import sys; sys.path.append('./python/')  # noqa
#   from cogtext.embeddings.universal_sentence_encoder import UniversalSentenceEncoder # noqa
#   from sentence_transformers import SentenceTransformer # noqa

#   models = [
#       # SentenceTransformer('all-mpnet-base-v2'),
#       # SentenceTransformer('all-distilroberta-v1'),
#       UniversalSentenceEncoder('universal-sentence-encoder-large/5', show_progress_bar=False),
#   ]

#   for model in models:
#     print(f'{model} (coef, p-value):', run_sts_benchmark(model, data_subset='test'))
#     model = None
#     del model

sts_dataset = tf.keras.utils.get_file(
    fname='sts.tar.gz',
    origin='http://alt.qcri.org/semeval2016/task1/data/uploads/sts2016-english-v1.1.tar.gz',
    extract=True)

sts_dataset_path = os.path.join(os.path.dirname(sts_dataset), 'stsbenchmark', 'sts-test.csv')

print(sts_dataset)
print(pandas.read_table(sts_dataset_path, quoting=csv.QUOTE_NONE, skip_blank_lines=True,))
