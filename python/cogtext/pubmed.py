import os
import requests
import xmltodict
from pathlib import Path
from collections import OrderedDict 


NCBI_EUTILS_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


def search_and_store(query, output_file, db='pubmed', api_key=os.environ['NCBI_API_KEY']):
  """Search for a term and store abstracts in a file

  Args:
    term (str): search query to look for in titles and abstracts.
    output_file (Path): File path to store results.
    db (str, optional): NCBI database to search ('pubmed' or  'pmc'). Defaults to 'pubmed'.
    api_key (string, optional): NCBI api key. Defaults to `os.environ['NCBI_API_KEY']`.

  Returns:
    Does not return anything. Abstracts will be stored in the `output_file`.
  """

  # step 1: create query and search

  print(f'[PubMed] {query}')

  url = f'{NCBI_EUTILS_BASE_URL}/esearch.fcgi'
  params = {
      'term': query,
      'usehistory': 'y',
      'db': db,
      'retmax': 0,
      'reldate': 21 * 365,
      'api_key': api_key
  }

  response = requests.get(url, params=params)
  search_response = xmltodict.parse(response.text)
  results_count = search_response['eSearchResult']['Count']

  print(f'[PubMed] Succesfully stored {results_count} hits on NCBI history server.')

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

  response = requests.post(url, params)

  # step 3: store abstracts
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, 'w') as f:
    f.write(response.text)
  print(f'[PubMed] Succesfully stored hits in {output_file}.')


def cleanup_abstract(abstract_text):
    """PubMed returns abstract with semantic tags. This function cleans those tags and keep the text."""
    select_content = lambda c: c if isinstance(c ,str) else c['#text'] if (c is not None and ('#text' in c)) else ''

    if isinstance(abstract_text, list):
        return ' '.join([select_content(a) for a in abstract_text])
    elif isinstance(abstract_text, OrderedDict):
        return select_content(abstract_text)
    return abstract_text    # when the abstract is string
