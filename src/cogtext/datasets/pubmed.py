from pathlib import Path
import polars as pl


class PubMedDataset():

  def __init__(self,
               root_dir: str = 'data/pubmed/',
               year=2023,
               drop_low_occurred_labels=False):
    self.year = year
    self.drop_low_occurred_labels = drop_low_occurred_labels

    self.root_dir = Path(root_dir)

  def load(self) -> pl.DataFrame:

    data = pl.read_csv(self.root_dir / f'abstracts_{self.year}.csv.gz')
    data = data.rename({'subcategory': 'label'})

    if self.drop_low_occurred_labels:
      low_freq_labels = data.group_by('label').count()
      low_freq_labels = low_freq_labels.filter(pl.col('count') < 3)['label'].to_list()
      data = data.filter(~pl.col('label').is_in(low_freq_labels))

    data = data.filter(pl.col('abstract').is_not_null())
    self.data = data

    return self.data

  @classmethod
  def select_relevant_journals(cls, data: pl.DataFrame) -> pl.DataFrame:
    """Remove certain irrelevant articles from the corpus.

    Note:
      This function uses `journal_iso_abbreviation` and `journal_title` to find relevant articles.

    """
    journals = data.group_by('journal_title').agg(pl.count('pmid')).rename({'pmid': 'pmid_count'})

    popular_journals = journals.filter(pl.col('pmid_count').ge(1000))

    # identify popular and relevant journals
    relevant_keywords = ['cognit', 'psyc', 'neur', 'brain', 'cortex', 'cog', 'intell',
                         'educ', 'behav', 'cereb', 'health']

    relevant_journals = journals.filter(
        pl.col('journal_title').str.contains('(?i)' + '|'.join(relevant_keywords)))

    # union
    journals = relevant_journals.join(popular_journals, on='journal_title', how='outer')
    journals = journals.filter(~pl.col('journal_title').is_duplicated())

    relevant_data = data.filter(pl.col('journal_title').is_in(journals['journal_title']))

    return relevant_data

  @classmethod
  def remove_short_abstracts(cls, data: pl.DataFrame, min_words=10) -> pl.DataFrame:

    _data = data.with_columns(
        pl.col('abstract').str.split(' ').map_elements(lambda x: len(x)).alias('n_words_in_abstract')
    )
    _data = _data.filter(pl.col('n_words_in_abstract') > min_words)

    return _data

  def find_mesh(mesh_list):
    raise NotImplementedError()
