import numpy as np
from typing import Union
from bertopic import BERTopic


class BERTopicScore():
  def __init__(self, embedding_model_name=None, **kwargs) -> None:
    self.embedding_model_name = embedding_model_name
    self.model = BERTopic(embedding_model=embedding_model_name,
                          calculate_probabilities=True,
                          n_gram_range=(1, 3),
                          # nr_topics='auto',
                          verbose=False)

  def encode(self, documents: list[str], **kwargs) -> np.ndarray:
    _, embeddings = self.model.fit_transform(documents=documents,)
    return embeddings

  def __call__(self, documents: Union[list[str], str], **kwargs) -> np.ndarray:
    return self.encode(documents, **kwargs)
