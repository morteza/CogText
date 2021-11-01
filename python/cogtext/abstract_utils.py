import gensim
from spacy import Language
from tqdm  import tqdm
import numpy

try:
  from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
except Exception as e:
  print('ERROR: Abstract preprocessing pipeline requires gensim v4.0 or later')


def preprocess_abstracts(abstracts: list[str], nlp_model: Language) -> list[str]:
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
  def _clean_doc(doc):
    cleaned = []
    for token in doc:
      if (not token.is_punct
          and token.is_alpha
          and not token.is_stop
          and not token.like_num
          and not token.is_space):
        cleaned.append(token.lemma_.strip())
    return cleaned

  # TODO do not discard sentence structure

  docs = nlp_model.pipe(abstracts, disable=['parser', 'ner'])
  docs = [_clean_doc(doc) for doc in tqdm(docs, total=len(abstracts))]

  # concat tokens
  docs = [' '.join(doc) for doc in docs]
  return docs


def concat_common_phrases(abstracts):
  # bigram
  ngram_phrases = gensim.models.Phrases(
    abstracts,
    connector_words=ENGLISH_CONNECTOR_WORDS) #, scoring='npmi')

  # find common phrases (1 to 5 words)
  for _ in range(1,6):
    ngram_phrases = gensim.models.Phrases(
      ngram_phrases[abstracts],
      connector_words=ENGLISH_CONNECTOR_WORDS) #, scoring='npmi')

  ngram = gensim.models.phrases.Phraser(ngram_phrases)

  # FIXME filter ngram stop words: docs = [[w for w in doc if w not in my_stop_words] for doc in docs]

  docs = [doc for doc in ngram[abstracts]]

  # concat tokens
  docs = [' '.join(doc) for doc in docs]
  return docs
