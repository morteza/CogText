from ._embedder import Embedder
import numpy as np
from transformers import pipeline


class HuggingFaceEmbedder(Embedder):

  def __init__(self, model_name='distilbert-base-cased') -> None:

    self.model = pipeline(task='feature-extraction',
                          model=model_name,
                          framework='tf', device=-1)

  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:

    if isinstance(documents, np.ndarray):
      documents = documents.tolist()

    embeddings = self.model(documents, **kwargs)
    pooled_embeddings = [np.array(e[0]).mean(axis=0) for e in embeddings]
    return np.array(pooled_embeddings)
