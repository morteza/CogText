from sentence_transformers import SentenceTransformer


def universal_sentence_encoding(X, y=None, version='v1'):
  """
  This function returns a sentence encoder that can be used to encode sentences
  using the Universal Sentence Encoder.
  """
  pass


def distilroberta(X, y=None, version='v1'):
  model = SentenceTransformer('all-distilroberta-v1')
  embeddings = model.encode(X, show_progress_bar=True)
  return embeddings
