from ._embedder import Embedder
import numpy as np
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_documents


class Word2VecEmbedder(Embedder):

  def __init__(self, model_name='glove-wiki-gigaword-300') -> None:
    # self.model = gensim.downloader.load(model_name)
    pass

  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:

    # preprocess
    documents = [' '.join(d) for d in preprocess_documents(documents)]

    self.model = Word2Vec(documents, vector_size=10, window=5, sg=1, min_count=1, **kwargs)

    pooled_embeddings = []
    for doc in documents:
      weights = [self.model.wv[w] for w in doc]
      norm_weights = np.sum(weights, axis=0) / len(doc)
      pooled_embeddings.append(norm_weights.tolist())

    embeddings = np.array(pooled_embeddings)

    return embeddings
