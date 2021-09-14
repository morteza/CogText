import pandas as pd
import re


def select_relevant_pubmed_articles(corpus: pd.DataFrame) -> pd.DataFrame:
  """Remove certain irrelevant articles from the corpus.

  Note:
    This function uses `journal_iso_abbreviation` and `journal_title` to find relevant articles.

  """

  def _is_relevant(journal):
    matched = re.search('', journal, flags=re.IGNORECASE)
    return bool(matched)

  journals = corpus[
      ['journal_title', 'journal_iso_abbreviation']
  ].value_counts().reset_index().rename(columns={0: 'n_articles'})

  # identify popular and relevant journals
  pop_journals_idx = journals[lambda x: x['n_articles'] > 1000]['n_articles'].index
  relevant_journals_idx = journals.query(
      'journal_title.str.contains("cognit|psyc|neur|brain|cortex|cog|intell|educ|behav|cereb", case=False)')[
          'n_articles'
  ].index

  relevant_journals = journals.loc[  # noqa: F841
      relevant_journals_idx.union(pop_journals_idx),
      'journal_iso_abbreviation'
  ]

  relevant_corpus = corpus.query('journal_iso_abbreviation.isin(@relevant_journals)').copy()

  return relevant_corpus
