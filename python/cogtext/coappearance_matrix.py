import pandas as pd
from tqdm import tqdm
from itertools import product


def _select_relevant_articles(corpus):
  def is_relevant(article):
    _is_relevant = (
        pd.notna(article['title']) and pd.notna(article['abstract']) and (
            'cognit' in article['abstract'] or
            'psych' in article['abstract'] or
            'psych' in article['journal_title'] or
            'cognit' in article['journal_title'] or
            'cognit' in article['title'] or
            'psych' in article['title']
        )
    )
    return _is_relevant
  return corpus[corpus.apply(is_relevant, axis=1)]


def generate_coappearance_matrix(
    pubmed_abstracts: pd.DataFrame,
    group: str = None,
    sort=True,
    probabilities=False
) -> pd.DataFrame:

  """DEPRECATED: use generate_coappearance_matrix_fast() instead.

  This function will be deleted as soon as removeing its usages from the codebase.
  
  """


  tasks = pubmed_abstracts.query('category == "CognitiveTask"')['subcategory'].unique()
  constructs = pubmed_abstracts.query('category == "CognitiveConstruct"')['subcategory'].unique()

  freqs = []

  for task in tasks:
    for construct in constructs:
      task_df = pubmed_abstracts.query('subcategory == @task')
      construct_df = pubmed_abstracts.query('subcategory == @construct').pipe(_select_relevant_articles)
      task_pmids = set(task_df['pmid'].unique())
      construct_pmids = set(construct_df['pmid'].unique())
      union_pmids = task_pmids.union(construct_pmids)
      intersection_pmids = task_pmids.intersection(construct_pmids)
      freqs.append([
          task,
          construct,
          len(task_pmids),
          len(construct_pmids),
          len(union_pmids),
          len(intersection_pmids)]
      )

  freqs_df = pd.DataFrame(
      freqs,
      columns=['task', 'construct',
               'task_corpus_size', 'construct_corpus_size',
               'union_corpus_size', 'intersection_corpus_size'])

  if sort:
    freqs_df.sort_values('intersection_corpus_size', ascending=False, inplace=True)

  if probabilities:
    freqs_df['probability'] = freqs_df['intersection_corpus_size'] / freqs_df['union_corpus_size']
    freqs_df = freqs_df.pivot(index='task', columns='construct', values='p')

  # freqs_df.to_csv(OUTPUT_FILE, index=False, compression='gzip')
  return freqs_df


def generate_coappearance_matrix_fast(pubmed_abstracts: pd.DataFrame,
                                      probability=False,
                                      group_categories=False):

  """A new faster version of co-appearance matrix generator."""

  pmids = pubmed_abstracts.pivot_table(values=['category','pmid'], index='subcategory', aggfunc=lambda x: x.to_list())
  pmids['category'] = pmids['category'].apply(lambda x: x[0])

  constructs = tasks = pmids.index
  columns = ['subcategory_1', 'subcategory_2']

  if group_categories:
    tasks = pmids.reset_index().pivot(columns='category').iloc[:,1].dropna()
    constructs = pmids.reset_index().pivot(columns='category').iloc[:,0].dropna()
    columns = ['construct', 'task']

  coappearances = pd.DataFrame.from_records(product(constructs, tasks), columns=columns)

  coappearances['intersection_corpus_size'] = coappearances.apply(
      lambda t: len(set(pmids.loc[t[0], 'pmid']).intersection(set(pmids.loc[t[1], 'pmid']))), axis=1)

  coappearances['union_corpus_size'] = coappearances.apply(
      lambda t: len(set(pmids.loc[t[0], 'pmid']).union(set(pmids.loc[t[1], 'pmid']))), axis=1)

  if probability:
    coappearances['probability'] = coappearances['intersection_corpus_size'] / coappearances['union_corpus_size']

  return coappearances
