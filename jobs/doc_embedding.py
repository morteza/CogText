from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch

# PARAMETERS
DATASET_NAMES = ['pubmed20pct_abstracts', 'pubmed20pct_abstracts_preprocessed']
MODEL_NAME = 'all-MiniLM-L6-v2'  # or a faster model: 'paraphrase-MiniLM-L3-v2'
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'


for dataset_name in DATASET_NAMES:
    print(f'Encoding {dataset_name}...')
    EMBEDDINGS_FILE = Path(f'models/{dataset_name}_{MODEL_NAME}_{DEVICE}.embeddings')

    # Initialize models/ folder if does not exist yet.
    Path('models/').mkdir(parents=True, exist_ok=True)

    # DATA
    data = pd.read_csv(f'data/{dataset_name}.csv.gz')

    # PREP
    data = data.dropna(subset=['abstract']).reset_index()
    data.rename(columns={'subcategory': 'label'}, errors='ignore', inplace=True)

    # MODEL FITTING
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        data['abstract'].to_list(),
        normalize_embeddings=True,
        show_progress_bar=True)

    # STORE RESULTS
    np.savez(EMBEDDINGS_FILE, embeddings)

    print('Done!')
