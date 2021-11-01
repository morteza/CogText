from ._embedder import Embedder
import numpy as np
from bertopic import BERTopic


class BERTopicEmbedder(Embedder):

  def __init__(self,
               embedding_model='all-mpnet-base-v2',
               **kwargs):
    """[summary]

    Args:
        embedding_model_name ([type], optional): [description]. Defaults to None.
    """

    self.model = BERTopic(embedding_model=embedding_model,
                          calculate_probabilities=True,
                          n_gram_range=(1, 3),
                          # nr_topics='auto',
                          verbose=False,
                          **kwargs)

  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:

    _, embeddings = self.model.fit_transform(documents=documents, **kwargs)
    return embeddings
