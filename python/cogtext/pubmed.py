import os
import requests
import xmltodict
from pathlib import Path


NLM_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


def search_and_store(term, output_file, db='pubmed', api_key=os.environ['NCBI_API_KEY']):
  """Search for a term and store abstracts in a file

  Args:
      term (str): [description]
      output_file (Path): [description]
      db (str, optional): NCBI database to search ('pubmed' or  'pmc'). Defaults to 'pubmed'.
      api_key (string, optional): NCBI api key. Defaults to os.environ['NCBI_API_KEY'].

  Returns:
      Does not return anything. Abstracts will be stored in the `output_file`.
  """

  # step 1: create query and search

  search_query = f'({term}[TIAB])'
  url = f'{NLM_BASE_URL}/esearch.fcgi'
  params = {
    'term': search_query.replace(' ','+'),
    'usehistory': 'y',
    'db': db,
    'retmax': 0,
    'reldate': 21 * 365,
    'api_key': api_key
  }

  response = requests.get(url,params=params)
  search_response = xmltodict.parse(response.text)
  results_count = search_response['eSearchResult']['Count']


  print(f'Succesfully stored search results on history server. Now retriving {results_count} abstracts...')

  # step 2: fetch abstracts
  url = f'{NLM_BASE_URL}/efetch.fcgi'
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
  print(f'Succesfully stored results to {output_file}')
