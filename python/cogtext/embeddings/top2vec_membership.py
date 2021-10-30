# %%

import numpy as np
from typing import Union
from top2vec import Top2Vec


class Top2VecMembership():
  def __init__(self,
               embedding_model_name='distiluse-base-multilingual-cased',
               speed='fast-learn') -> None:
    self.embedding_model_name = embedding_model_name
    self.speed = speed

  def encode(self, documents: list[str], **kwargs) -> np.ndarray:

    model = Top2Vec(documents,
                    min_count=5, speed=self.speed,
                    embedding_model=self.embedding_model_name)

    embeddings = model.get_documents_topics(model.document_ids,
                                            num_topics=model.get_num_topics())[1]

    return embeddings

  def __call__(self, documents: Union[list[str], str], **kwargs) -> np.ndarray:
    return self.encode(documents, **kwargs)
