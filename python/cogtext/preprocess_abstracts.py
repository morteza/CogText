import gensim
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

def preprocess_abstracts(abstracts: list[str], nlp_model, extract_phrases=False) -> list[str]:
  """Opinionated preprocessing pipeline.

  Note:
  
    run this to download the SpaCy model: `python -m spacy download en_core_web_sm`

  Args:
    texts (list[str]): list of texts, each item is one text document.
    corpus_name (str): Name of the corpus

  Returns:
      list[str]: preprocessed documents
  """
  # DEBUG standard preprocessing pipeline
  # docs = \
  #   texts['abstract'].progress_apply(lambda abstract: gensim.parsing.preprocess_string(abstract)).to_list()

  # flake8: noqa: W503
  def _clean(doc):
    cleaned = []
    for token in doc:
      if (not token.is_punct
          and token.is_alpha
          and not token.is_stop
          and not token.like_num
          and not token.is_space):
        cleaned.append(token.lemma_.lower().strip())
    return cleaned

  docs = [_clean(txt) for txt in nlp_model.pipe(abstracts)]

  if extract_phrases:
    # bigram
    ngram_phrases = gensim.models.Phrases(docs, connector_words=ENGLISH_CONNECTOR_WORDS) #, scoring='npmi')

    # find common phrases (1 to 5 words)
    for _ in range(1,6):
      ngram_phrases = gensim.models.Phrases(ngram_phrases[docs], connector_words=ENGLISH_CONNECTOR_WORDS) #, scoring='npmi')

    ngram = gensim.models.phrases.Phraser(ngram_phrases)

    # FIXME filter ngram stop words: docs = [[w for w in doc if w not in my_stop_words] for doc in docs]

    docs = [doc for doc in ngram[docs]]

  # concat tokens
  docs = [' '.join(doc) for doc in docs]

  return docs
