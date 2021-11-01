from ._embedder import Embedder
import numpy as np
from top2vec import Top2Vec


class Top2VecEmbedder(Embedder):

  def __init__(self,
               embedding_model_name='distiluse-base-multilingual-cased'):
    self.embedding_model_name = embedding_model_name

  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:

    speed = kwargs.get('speed', 'deep-learn')

    model = Top2Vec(documents,
                    min_count=5, speed=speed,
                    embedding_model=self.embedding_model_name)

    embeddings = model.get_documents_topics(model.document_ids,
                                            num_topics=model.get_num_topics())[1]

    return embeddings
