import os
import requests
import xmltodict
from pathlib import Path
from collections import OrderedDict
from datetime import date
from xml.etree import ElementTree
import re

import pandas as pd

NCBI_EUTILS_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


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
