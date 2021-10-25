# %%
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

import sys; sys.path.append('./python/')  # noqa
from cogtext.embeddings.universal_sentence_encoder import UniversalSentenceEncoder # noqa

# PARAMETERS
DATASET_NAMES = ['pubmed/abstracts', 'pubmed/abstracts_preprocessed']
# MODEL_NAMES = ['universal-sentence-encoder', 'paraphrase-MiniLM-L3-v2', 'all-MiniLM-L6-v2']
MODEL_NAMES = ['universal-sentence-encoder-v4']
BATCH_SIZE = 100


for dataset_name in DATASET_NAMES:
    print(f'Encoding {dataset_name}...')

    # Initialize models/ folder if does not exist yet.
    Path('models/').mkdir(parents=True, exist_ok=True)

    # DATA
    data = pd.read_csv(f'data/{dataset_name}.csv.gz')

    # PREP
    data = data.dropna(subset=['abstract']).reset_index()
    data.rename(columns={'subcategory': 'label'}, errors='ignore', inplace=True)

    # MODEL FITTING
    for model_name in MODEL_NAMES:
        print(f'Fitting {model_name}...')

        if 'all-MiniLM' in model_name:
            # device = 'gpu' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer(model_name)
            embeddings = model.encode(
                data['abstract'].to_list(),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True)

        if 'universal-sentence-encoder' in model_name:
            model = UniversalSentenceEncoder(batch_size=BATCH_SIZE)
            embeddings = model.encode(data['abstract'].values)

        # STORE RESULTS
        embedding_fname = Path(f'models/{model_name}/{dataset_name.replace("/","_").replace("pubmed/", "")}_embeddings')
        np.savez(embedding_fname, embeddings)

    print('Done!')
