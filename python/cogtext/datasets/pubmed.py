import pandas as pd
from pandas._typing import FilePathOrBuffer

import os
import requests
from pathlib import Path
from collections import OrderedDict
from datetime import date
from xml.etree import ElementTree
import re

NCBI_EUTILS_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


class PubMedDataLoader():
  def __init__(self,
               root_dir: str = 'data/pubmed/',
               n_articles: int = None,
               preprocessed=True,
               drop_low_occurred_labels=False) -> pd.DataFrame:

    self.root_dir = Path(root_dir)

    if preprocessed:
      self.data = pd.read_csv(self.root_dir / 'abstracts_preprocessed.csv.gz', nrows=n_articles)
    else:
      self.data = pd.read_csv(self.root_dir / 'abstracts.csv.gz', nrows=n_articles)

    self.data.rename(columns={'subcategory': 'label'}, inplace=True)

    if drop_low_occurred_labels:
      low_lbls = self.data.groupby('label')['pmid'].count()
      low_lbls = low_lbls[low_lbls < 2].index
      low_lbls_idx = self.data.query('label in @low_lbls').index
      self.data.drop(index=low_lbls_idx, inplace=True)

    self.data = self.data.dropna(subset=['abstract', 'pmid']).reset_index(drop=True)

  def load(self) -> pd.DataFrame:
    return self.data

  def __call__(self) -> pd.DataFrame:
    return self.load()


def search_and_store(query, output_file: Path, db='pubmed', api_key=os.environ.get('NCBI_API_KEY', '')):
  """Search for a term and store abstracts in a file

  Args:
    term (str): search query to look for in titles and abstracts.
    output_file (Path): File path to store results.
    db (str, optional): NCBI database to search ('pubmed' or  'pmc'). Defaults to 'pubmed'.
    api_key (string, optional): NCBI api key. Defaults to `os.environ['NCBI_API_KEY']`.

  Returns:
    Does not return anything. Abstracts will be stored in the `output_file`.
  """
  import xmltodict  # noqa

  # step 1: create query and search

  print(f'[PubMed] query: {query}')

  url = f'{NCBI_EUTILS_BASE_URL}/esearch.fcgi'
  params = {
      'term': query,
      'usehistory': 'y',
      'db': db,
      'retmax': 0,
      'reldate': (date.today() - date(1900, 1, 1)).days,
      'api_key': api_key
  }

  response = requests.get(url, params=params)
  search_response = xmltodict.parse(response.text)
  results_count = int(search_response['eSearchResult']['Count'])

  if results_count == 0:
    print('[PubMed] no article found.')
    return

  print(f'[PubMed] stored {results_count} hits on NCBI history server.')

  # step 2: fetch abstracts
  url = f'{NCBI_EUTILS_BASE_URL}/efetch.fcgi'
  params = {
      'db': db,
      'api_key': api_key,
      'WebEnv': search_response['eSearchResult']['WebEnv'],
      'query_key': search_response['eSearchResult']['QueryKey'],
      'rettype': 'abstract',
      'retmode': 'xml'
  }

  retstart = 0
  retmax = 10000

  # step 3: store abstracts
  output_xml = None

  while retstart < results_count:
    params['retstart'] = retstart
    response = requests.post(url, params)
    retstart += retmax

    # get rid of invalid line terminator
    response_text = response.text.replace('\u2029', ' ')

    # combine XMLs
    response_xml = ElementTree.fromstring(response_text)

    if output_xml is None:
      output_xml = response_xml
    else:
      print('[PubMed] merging multiple responses...')
      output_xml.extend(response_xml.findall('PubmedArticle'))

  output_file.parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, 'w') as f:
    ElementTree.ElementTree(output_xml).write(f, encoding='unicode')

  print(f'[PubMed] stored hits in {output_file}.')


def cleanup_abstract(abstract_text):
    """PubMed returns abstract with semantic tags. This function cleans those tags and keep the text."""

    def _select_content(c):
      return c if isinstance(c, str) else c['#text'] if (c is not None and ('#text' in c)) else ''

    if isinstance(abstract_text, list):
        return ' '.join([_select_content(a) for a in abstract_text])
    elif isinstance(abstract_text, OrderedDict):
        return _select_content(abstract_text)
    return abstract_text    # when the abstract is string


def find_mesh(mesh_list):
    """Extracts MeSH names from a list of XML MedlineCitation.MeshHeadingList.MeshHeading tags."""
    if not isinstance(mesh_list, list):
        return []

    mesh_names = [h['DescriptorName']['#text'] for h in mesh_list]  # if d['DescriptorName']['@MajorTopicYN'] == 'Y']
    return mesh_names


def extract_doi(ids):
    """Helper function to extact DOI from PubMed `PubmedData.ArticleIdList.ArticleId`."""
    if isinstance(ids, list):
        all_dois = [_id['#text'] for _id in ids if _id['@IdType'] == 'doi' and '#text' in _id.keys()]
        if len(all_dois) == 0:
            return None
        return all_dois[0]
    else:
        return None


def parse_publication_year(x):
    if isinstance(x, str):
        year = re.findall('[0-9]+', x)[0]
        return int(year)
    return x if pd.notna(x) else 0.


def load_pubmed_abstacts_dataset(
    reader: FilePathOrBuffer = 'data/pubmed/abstracts.csv.gz',
    group_by: str = None,
    frac: int = None,
    min_count: int = None,
    max_count: int = None,
    preprocess_abstracts: bool = False,
    drop_empty_groups: bool = True,
    only_relevant_journals: bool = False,
    drop_invalid_abstracts: bool = False,
    reset_index: bool = False,
    return_Xy: bool = False,
) -> pd.DataFrame:
  """[summary]

  Inputs:
  ---
    pubmed_abstracts_df (pd.DataFrame): [description]
    groupby (str, optional): [description]. Defaults to 'label'.
    frac (int, optional): [description]. Defaults to None.

  Outputs:
  ---
    pd.DataFrame: [description]

  Examples:
  ---
    >>> sample_dataset(PUBMED)
    >>> PUBMED.pipe(sample_dataset)
  """
  raise NotImplementedError


class PubMedPreprocessor():
  def __init__(self, vocabulary_size=20000):
    """[summary]

    Inputs:
    ---
    """
    pass

  def transform(self, pubmed_abstracts_df):
    raise NotImplementedError

  @classmethod
  def select_relevant_journals(cls, pubmed_abstracts_df: pd.DataFrame) -> pd.DataFrame:
    """Remove certain irrelevant articles from the corpus.

    Note:
      This function uses `journal_iso_abbreviation` and `journal_title` to find relevant articles.

    """

    journals = pubmed_abstracts_df[
        ['journal_title', 'journal_iso_abbreviation']
    ].value_counts().reset_index().rename(columns={0: 'n_articles'})

    # identify popular and relevant journals
    pop_journals_idx = journals[lambda x: x['n_articles'] > 1000]['n_articles'].index
    relevant_journals_idx = journals.query(
        'journal_title.str.contains("cognit|psyc|neur|brain|cortex|cog|intell|educ|behav|cereb", case=False)')[
            'n_articles'
    ].index

    relevant_journals = journals.loc[  # noqa
        relevant_journals_idx.union(pop_journals_idx),
        'journal_iso_abbreviation'
    ]

    relevant_articles = pubmed_abstracts_df.query('journal_iso_abbreviation.isin(@relevant_journals)').copy()

    return relevant_articles

  @classmethod
  def remove_short_abstracts(cls, pubmed_df, min_words=10):
    df = pubmed_df.dropna(subset=['abstract'])

    # from sklearn import feature_extraction
    # vectorizer = feature_extraction.text.CountVectorizer()
    # counts = vectorizer.fit_transform(df['abstract']).toarray()
    # short_abstract_indices = (counts.sum(axis=1) < min_words).nonzero()[0]

    long_abstract_indices = df.query('abstract.str.count(" ") >= @min_words').index
    df = df.loc[long_abstract_indices]

    return df
