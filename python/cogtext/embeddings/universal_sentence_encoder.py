
from typing import Union
import numpy as np
import tensorflow_hub as hub
from tqdm import tqdm


class UniversalSentenceEncoder():

  def __init__(
      self,
      model_name='universal-sentence-encoder-large/5',
      batch_size=1000,
      show_progress_bar: bool = True,
      **kwargs
  ):
    """Encodes raw texts using pretrained Universal Sentence Encoder.

      Note
      ----
        Pretrained model is fetched from Tensorflow Hub, so it requires `tensorflow`, `tensorflow_hub`,
        CUDA, and `cudnn`

      Args
      ----
        model_name (str, optional): Tensorflow Hub module name, e.g., 'universal-sentence-encoder/4'.
          Defaults to 'universal-sentence-encoder-large/5'.

      Example
      -------
        >>> print(UniversalSentenceEmbedding.example())
    """

    module_url = f'https://tfhub.dev/google/{model_name}'
    self.model = hub.load(module_url)
    self.batch_size = batch_size
    self.show_progress_bar = show_progress_bar

  def encode(self, documents: Union[list[str], str, np.array], **kwargs) -> np.array:
    """Maps a list of raw strings into a 512-dimensional embedding space.

    Args:
    ---
        documents (list[str]): Raw documents.

    Returns:
    ---
        numpy.array: Embeddings array of shape $n_{documents} \times 512$.
    """
    embeddings = []

    if isinstance(documents, str):
      documents = [documents]

    if isinstance(documents, list):
      documents = np.array(documents)

    if documents.shape[0] > self.batch_size:
      batches = np.array_split(documents, round(documents.shape[0] / self.batch_size))
    else:
      batches = documents.reshape((-1, 1))

    if self.show_progress_bar:
      batches = tqdm(batches)

    for batch in batches:
      batch_embeddings = self.model(batch).numpy()
      embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

  @staticmethod
  def example():
    return UniversalSentenceEncoder().encode(['this is a test'])
