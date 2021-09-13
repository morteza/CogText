import pandas as pd
from tqdm import tqdm


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


def generate_coappearance_matrix(pubmed_abstracts: pd.DataFrame) -> pd.DataFrame:

  tasks = pubmed_abstracts.query('category == "CognitiveTask"')['subcategory'].unique()
  constructs = pubmed_abstracts.query('category == "CognitiveConstruct"')['subcategory'].unique()

  freqs = []

  for task in tqdm(tasks):
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

  freqs_df.sort_values('intersection_corpus_size', ascending=False, inplace=True)

  # freqs_df.to_csv(OUTPUT_FILE, index=False, compression='gzip')
  return freqs_df
