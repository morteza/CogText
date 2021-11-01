from ._embedder import Embedder
import numpy as np
import tensorflow_hub as hub
from tqdm import tqdm


class TensorFlowHubEmbedder(Embedder):

  def __init__(self,
               model_name='universal-sentence-encoder-large/5',
               batch_size=1000,
               show_progress_bar: bool = True,
               **kwargs):
    """Embeds raw texts into a vector space using a pretrained Universal Sentence Encoder.

      Note
      ----
        Pretrained model is fetched from Tensorflow Hub and requires `tensorflow`, `tensorflow_hub`,
        CUDA, and `cudnn`.

      Args
      ----
        model_name (str, optional): Tensorflow Hub module name, e.g., 'universal-sentence-encoder/4'.
          Defaults to 'universal-sentence-encoder-large/5'.

      Example
      -------
        >>> print(GoogleUSEEmbedder().embed(['Hello world!', 'How are you?']))
    """

    module_url = f'https://tfhub.dev/google/{model_name}'
    self.model = hub.load(module_url)
    self.batch_size = batch_size
    self.show_progress_bar = show_progress_bar

  def __call__(self, documents: list[str], **kwargs) -> np.array:
    """Maps a list of raw strings into a 512-dimensional embedding space.

    Args:
    ---
        documents (list[str]): Raw documents.

    Returns:
    ---
        numpy.array: Embeddings array of shape $n_{documents} \times 512$.
    """
  
    n_batches = round(len(documents) / self.batch_size)

    if len(documents) > self.batch_size:
      batches = np.array_split(documents, n_batches)
    else:
      batches = np.array(documents).reshape((-1, 1))

    if self.show_progress_bar:
      batches = tqdm(batches)

    embeddings = []
    for batch in batches:
      batch_embeddings = self.model(batch).numpy()
      embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    return embeddings
