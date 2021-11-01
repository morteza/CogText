from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
  """Base abstract class for all the embedder classes."""

  @abstractmethod
  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:
    return self.embed(documents, **kwargs)
