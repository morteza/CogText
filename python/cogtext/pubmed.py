import os
import requests
import xmltodict
from pathlib import Path


NCBI_EUTILS_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


def search_and_store(term, output_file, db='pubmed', api_key=os.environ['NCBI_API_KEY'], suffix=[]):
  """Search for a term and store abstracts in a file

  Args:
    term (str): search query to look for in titles and abstracts.
    output_file (Path): File path to store results.
    db (str, optional): NCBI database to search ('pubmed' or  'pmc'). Defaults to 'pubmed'.
    api_key (string, optional): NCBI api key. Defaults to `os.environ['NCBI_API_KEY']`.
    suffix (list, optional): add optional suffix to the query, e.g., 'task', test', or 'survey'. Defaults to `[]`.

  Returns:
    Does not return anything. Abstracts will be stored in the `output_file`.
  """

  # step 1: create query and search

  search_queries = []

  if len(suffix) == 0:
    search_queries.append(f'("{term}"[TIAB])')

  for sfx in suffix:
      search_queries.append(f'("{term} {sfx}"[TIAB])')

  # build final search query
  search_query = ' OR '.join(search_queries)  # .replace(' ', '+')

  print(f'{term}: query={search_query}')

  url = f'{NCBI_EUTILS_BASE_URL}/esearch.fcgi'
  params = {
      'term': search_query,
      'usehistory': 'y',
      'db': db,
      'retmax': 0,
      'reldate': 21 * 365,
      'api_key': api_key
  }

  response = requests.get(url, params=params)
  search_response = xmltodict.parse(response.text)
  results_count = search_response['eSearchResult']['Count']

  print(f'{term}: succesfully stored {results_count} search hits on NCBI history server. Now retriving them...')

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
  with open(output_file, 'a') as f:
    f.write(response.text)
  print(f'{term}: Succesfully stored search hits in {output_file}')
