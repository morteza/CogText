import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


# PARAMETERS
MODEL_NAME = 'all-MiniLM-L6-v2'  # or a faster model: 'paraphrase-MiniLM-L3-v2'
EMBEDDINGS_FILE = Path(f'models/pubmed_abstracts_{MODEL_NAME}.embeddings')

# Initialize models/ folder if does not exist yet.
Path('models/').mkdir(parents=True, exist_ok=True)

# DATA
PUBMED = pd.read_csv(
    'data/pubmed_abstracts_preprocessed.csv.gz'
).dropna(subset=['abstract']).reset_index()

PUBMED.rename(columns={'subcategory': 'label'}, errors='ignore', inplace=True)

# MODEL FITTING
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    PUBMED['abstract'].to_list(),
    normalize_embeddings=True,
    show_progress_bar=True)

# STORE RESULTS
np.savez(EMBEDDINGS_FILE, embeddings)
print('Done!')
