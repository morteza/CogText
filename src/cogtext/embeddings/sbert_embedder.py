from ._embedder import Embedder
import numpy as np
from gensim.parsing.preprocessing import preprocess_documents
from sentence_transformers import SentenceTransformer


class SBertEmbedder(Embedder):

  def __init__(self, model_name='all-mpnet-base-v2', preprocess=False) -> None:
    self.model = SentenceTransformer(model_name)
    self.preprocess = preprocess

  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:

    # preprocess
    if self.preprocess:
      documents = [' '.join(d) for d in preprocess_documents(documents)]

    embeddings = self.model.encode(documents, **kwargs)

    return embeddings
