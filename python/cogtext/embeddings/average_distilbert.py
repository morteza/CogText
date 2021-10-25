import numpy as np
from transformers import pipeline
from typing import Union


class AverageDistilBert():
  def __init__(self, model_name='distilbert-base-cased') -> None:

    self.model = pipeline(task='feature-extraction',
                          model=model_name,
                          framework='tf', device=-1)

  def encode(self, documents: Union[list[str], str, np.ndarray], **kwargs) -> np.ndarray:

    if isinstance(documents, np.ndarray):
      documents = documents.tolist()

    embeddings = self.model(documents, **kwargs)
    pooled_embeddings = [np.array(e[0]).mean(axis=0) for e in embeddings]
    return np.array(pooled_embeddings)

  def __call__(self, documents: Union[list[str], str], **kwargs) -> np.ndarray:
    return self.encode(documents, **kwargs)
