# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Test-Construct frequency matrix
# 
# This notebook generates and stores a sparse matrix that contains test/construct co-occurrence frequencies. Each row is a test and each column is a cognitive construct. Values show number of articles that mention both the task and the cognitive construct.
# 
# Matrix is stored as sparse format in the following path: `data/pubmed/test_construct_frequencies.csv`.

# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm


test_files = list(Path('data/pubmed/tests/').glob('*.csv'))
construct_files = list(Path('data/pubmed/constructs/').glob('*.csv'))

freqs = []

for test_file in tqdm(test_files):
  for construct_file in construct_files:
    test_df = pd.read_csv(test_file)
    construct_df = pd.read_csv(construct_file)
    test_pmids = set(test_df['pmid'].unique())
    construct_pmids = set(construct_df['pmid'].unique())
    shared_pmids = test_pmids.intersection(construct_pmids)
    freqs.append([
        test_file.stem,
        construct_file.stem,
        len(test_pmids),
        len(construct_pmids),
        len(shared_pmids)]
    )

freqs_df = pd.DataFrame(
    freqs,
    columns=['test', 'construct', 'test_frequency', 'construct_frequency', 'joint_frequency'])

freqs_df.sort_values('joint_frequency', ascending=False).to_csv('data/pubmed/test_construct_frequency_matrix.csv', index=False)
