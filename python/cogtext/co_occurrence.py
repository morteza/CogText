import pandas as pd
from itertools import product


def co_occurrence_matrix(
    pubmed_abstracts: pd.DataFrame,
    probability=False,
    groupby_category=False
):

  """A new faster version of co-appearance matrix generator.

  TODO: currently only 'task' and 'construct' categories are supported, extend that to even
        unknown set of categories.

  TODO: try scikit-learn count-vectorizer to speed up?
  """

  pmids = pubmed_abstracts.pivot_table(
      values=['category', 'pmid'],
      index='label',
      aggfunc=lambda x: x.to_list()
  )

  pmids['category'] = pmids['category'].apply(lambda x: x[0])
  constructs = tasks = pmids.index
  columns = ['label_1', 'label_2']

  if groupby_category:
    tasks = pmids.reset_index().pivot(columns='category').iloc[:, 1].dropna()
    constructs = pmids.reset_index().pivot(columns='category').iloc[:, 0].dropna()
    columns = ['construct', 'task']

  cooc = pd.DataFrame.from_records(product(constructs, tasks), columns=columns)

  for i, col in enumerate(columns):
    cooc[f'{col}_corpus_size'] = cooc.apply(lambda t: len(set(pmids.loc[t[i], 'pmid'])), axis=1)

  cooc['intersection_corpus_size'] = cooc.apply(
      lambda t: len(set(pmids.loc[t[0], 'pmid']).intersection(set(pmids.loc[t[1], 'pmid']))), axis=1)

  cooc['union_corpus_size'] = cooc.apply(
      lambda t: len(set(pmids.loc[t[0], 'pmid']).union(set(pmids.loc[t[1], 'pmid']))), axis=1)

  if probability:
    cooc['probability'] = cooc['intersection_corpus_size'] / cooc['union_corpus_size']

  return cooc
