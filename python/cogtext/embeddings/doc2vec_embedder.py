from ._embedder import Embedder
import numpy as np
import collections
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import preprocess_documents
from sklearn.metrics.pairwise import cosine_similarity


class Doc2VecEmbedder(Embedder):

  def __init__(self):
    self.model = Doc2Vec(vector_size=768, min_count=1, epochs=50)

  def __call__(self, documents: list[str], **kwargs) -> np.ndarray:

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocess_documents(documents))]
    
    self.model.build_vocab(documents)
    self.model.train(documents,
                     total_examples=self.model.corpus_count,
                     epochs=self.model.epochs)

    embeddings = self.model.dv.get_normed_vectors()
    embeddings = np.array(embeddings)

    return embeddings

  def sanity_check(self, documents):
    
    """A sanity check for the doc2vce embedder;
    
    it make sure that each document is the most similar one to itself in the embedding space.

    Returns:
        [type]: [description]
    """
    ranks = []
    
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocess_documents(documents))]

    for doc_id in range(len(documents)):
      inferred_vector = self.model.infer_vector(documents[doc_id].words)
      sims = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))
      rank = [docid for docid, sim in sims].index(doc_id)
      ranks.append(rank)

    counter = collections.Counter(ranks)
    return counter

  @classmethod
  def example():
    docs = ["hello world iraq", "hello world iran", "hello world united states", "this is a text irrelvant to the rest about dogs"]

    embedder = Doc2VecEmbedder()
    e = embedder(docs)

    return cosine_similarity(e)
