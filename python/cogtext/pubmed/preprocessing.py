import pandas as pd


class PubMedPreprocessor():
  def __init__(self, vocabulary_size=20000):
    """[summary]

    Inputs:
    ---
    """
    pass

  def transform(self, pubmed_abstracts_df):
    pass

  @staticmethod
  def select_relevant_journals(cls, pubmed_abstracts_df: pd.DataFrame) -> pd.DataFrame:
    """Remove certain irrelevant articles from the corpus.

    Note:
      This function uses `journal_iso_abbreviation` and `journal_title` to find relevant articles.

    """

    journals = pubmed_df[
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

    relevant_corpus = pubmed_df.query('journal_iso_abbreviation.isin(@relevant_journals)').copy()

    return relevant_corpus
