#%%
import numpy as np
from typing import Union

import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_documents


class AverageWord2Vec():
  def __init__(self, model_name='glove-wiki-gigaword-300') -> None:
    # self.model = gensim.downloader.load(model_name)
    pass

  def encode(self, documents: Union[list[str], str, np.ndarray], **kwargs) -> np.ndarray:

    # TODO preprocess documents
    documents = [' '.join(d) for d in preprocess_documents(documents)]

    self.model = Word2Vec(documents, vector_size=10, window=5, sg=1, min_count=1)

    pooled_embeddings = []
    for doc in documents:
      weights = [self.model.wv[w] for w in doc]
      weights_normed = np.sum(weights, axis=0) / len(doc)
      pooled_embeddings.append(weights_normed.tolist())

    return pooled_embeddings

  def __call__(self, documents: Union[list[str], str], **kwargs) -> np.ndarray:
    return self.encode(documents, **kwargs)
