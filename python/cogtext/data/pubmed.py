import pandas as pd
from pandas._typing import FilePathOrBuffer


def load_pubmed_abstacts_dataset(
    reader: FilePathOrBuffer = 'data/pubmed_abstracts.csv.gz',
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
  pass
