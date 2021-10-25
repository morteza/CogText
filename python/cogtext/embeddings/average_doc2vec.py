import numpy as np
from typing import Union

import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


class AverageDoc2Vec():
  def __init__(self) -> None:

    self.model = Doc2Vec()

  def encode(self, documents: Union[list[str], str, np.ndarray], **kwargs) -> np.ndarray:

    if isinstance(documents, np.ndarray):
      documents = documents.tolist()

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

    embeddings = Doc2Vec(documents).dv.get_normed_vectors()
    return np.array(embeddings)

  def __call__(self, documents: Union[list[str], str], **kwargs) -> np.ndarray:
    return self.encode(documents, **kwargs)
