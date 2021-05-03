"""Here, topics will be extracted for each cognitive test/construct corpus and words in each topic will be
visualized as a word-cloud. It will produce one figure per corpus within that it contains one word-cloud per topic.

Note: All accesses to files assume current working directory is the project root folder.

Note: The fastest coherency scoring is `u_mass`, however, the default is `c_v`. I'm using `u_mass` here to
      speed things up on local machines.


TODO:
  - [ ] remove common terms (e.g., task, test).
  - [ ] apply bigram/trigram models to concat commonly co-occurred terms.
  - [ ] improve plot readabilities (x-label, y-label, description)
  - [ ] Some results are suspicious.
        (e.g., in tests_Reverse Categorization, some topics contain only zero coefficients).
  - [ ] Use c_v coherency instead of u_mass (it will be slower to fit).
  - [ ] allow a more exhaustive number of topics (min 1 and max 1000 topics?). Currently it's limited to 10 to 30 topics.
        Takes time to fit the model. Maybe I can run that on HPC?
  - [ ] cleaner file naming (remove tests/constructs prefix if there is not conflict).
  - [ ] Show progress bar for the corpora iterator.
  - [ ] model fitting progress bars should be a child of the main loop progress bar.
  - [ ] Ignore "MONITOR" and "STOP" tasks to speed things up.
"""

# %%
import sys
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm
import gensim
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
import spacy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('notebook')

# constraints and constants
N_TOPICS_MIN = 10
N_TOPICS_MAX = 30
N_TOPIC_WORDS = 10  # number of words per topic (for printing and plotting purposes)


def preprocess(texts: list[str], corpus_name: str):
  """Opinionated preprocessing pipeline.

  Note: run the following command first to download SpaCy corpus:
        ```bash
        python -m spacy download en_core_web_sm
        ```


  Args:
      texts (list[str]): list of texts, each item is one text document.
      corpus_name (str): Name of the corpus

  Returns:
      (list[list[str]], Dictionary): preprocessed documents, and gensim word2id dictionary
  """
  # DEBUG standard preprocessing pipeline
  # docs = \
  #   texts['abstract'].progress_apply(lambda abstract: gensim.parsing.preprocess_string(abstract)).to_list()

  print('Training NLP model...')
  nlp = spacy.load('en_core_web_sm')

  # additional stop words
  my_stop_words = ['study', 'task', 'test']
  for stop_word in my_stop_words:
    lexeme = nlp.vocab[stop_word]
    lexeme.is_stop = True

  # flake8: noqa: W503
  def _clean(doc):
    cleaned = []
    for w in doc:
      if (not w.is_punct
          and not w.is_stop
          and not w.like_num
          and not w.is_space):
        cleaned.append(w.lemma_)
    return cleaned

  docs = tqdm([_clean(d) for d in nlp.pipe(texts)], desc='Preprocessing')

  # bigram
  ngram_phrases = gensim.models.Phrases(docs, connector_words=ENGLISH_CONNECTOR_WORDS)

  # there are cases that a test or construct contains 4 terms; a heuristic is to count spaces in the corpus_name
  for n in range(max(1, 2 + corpus_name.count(' '))):
    ngram_phrases = gensim.models.Phrases(ngram_phrases[docs], connector_words=ENGLISH_CONNECTOR_WORDS)

  ngram = gensim.models.phrases.Phraser(ngram_phrases)
  docs = list(ngram[docs])
  # DEBUG filter ngram stop words: docs = [[w for w in doc if w not in my_stop_words] for doc in docs]

  words = gensim.corpora.Dictionary(docs)
  words.filter_extremes(no_below=2, no_above=.9)

  return docs, words


def get_topics(texts: pd.DataFrame, corpus_name: str):

  texts['abstract'].fillna(texts['title'], inplace=True)
  docs, words = preprocess(texts['abstract'].to_list(), corpus_name)
  corpus = [words.doc2bow(doc) for doc in docs]

  model_scores = {}
  models = {}
  for n_topics in tqdm(range(N_TOPICS_MIN, N_TOPICS_MAX), desc='Fitting n_topics'):
    models[n_topics] = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=words, num_topics=n_topics)
    model_scores[n_topics] = gensim.models.CoherenceModel(model=models[n_topics],
                                                          corpus=corpus,
                                                          # texts=docs,
                                                          coherence='u_mass',
                                                          # coherence='c_v',
                                                          dictionary=words,
                                                          processes=-1,
                                                          ).get_coherence()

  # DEBUG plot fitting trace
  # sns.lineplot(x=model_scores.keys(), y=model_scores.values())
  # plt.xlabel('Number of topics')
  # plt.ylabel('coherence score')
  # plt.suptitle(corpus_name)
  # plt.show()

  n_topics = max(model_scores, key=model_scores.get)
  model = models[n_topics]

  # get rid of all other fitted models and save memory
  models.clear()
  # print(model, file=sys.stderr)

  topics = model.top_topics(corpus, texts=docs, topn=N_TOPIC_WORDS, coherence='u_mass')

  # cleanup and reorder topics/scores/terms for plotting/reporting
  topics_df = pd.DataFrame(topics).rename(columns={1: 'topic_score', 0: 'term'})
  topics_df.sort_values('topic_score').reset_index(inplace=True)
  topics_df['topic_index'] = topics_df.index + 1
  topics_df = topics_df.explode('term')
  topics_df['term_coef'] = topics_df['term'].apply(lambda x: x[0])
  topics_df['term'] = topics_df['term'].apply(lambda x: x[1])
  topics_df.attrs['corpus_name'] = corpus_name

  return topics_df


def save_topic_barplots(topics: pd.DataFrame,
                        corpus_name: str,
                        savefig_fname: Path = None):
  """Generate a png plot in that topics are sorted by coherency score.

  Note: Each subplot shows a topic, in each terms are ordered by their
        contribution to the topic, i.e., coefficients. X axis shows
        coefficients and Y axis shows terms.
  """

  if savefig_fname is None:
    # default path
    savefig_fname = Path('outputs/topic_barplots') / (corpus_name.replace('/', '_') + '.png')

  grid = sns.FacetGrid(topics,
                       col='topic_index', hue='topic_index', palette='deep', col_wrap=4,
                       sharex='term_coef', sharey=False,
                       col_order=range(1, topics['topic_index'].nunique() + 1, 1))
  grid.map_dataframe(sns.barplot, x='term_coef', y='term', orient='h', dodge=False)
  plt.suptitle(f'{corpus_name} topics', weight='bold', x=.2)
  plt.tight_layout()
  # plt.show()
  plt.savefig(savefig_fname)
  # to reuse the same plt and prevent memory leak warning
  plt.clf()
  plt.close('all')


def save_topic_wordclouds(topics: pd.DataFrame,
                          corpus_name: str,
                          savefig_fname: Path = None):
  raise NotImplementedError()


# load and iterate corpora
corpora = Path('data/pubmed').glob('**/*.csv')

for fname in corpora:
  corpus_name = re.findall('.*/pubmed/(.*)\\.csv', str(fname))[0]
  print(f'>> {corpus_name}...', file=sys.stderr)
  df = pd.read_csv(fname)

  # store test/construct names in the dataframe attribute to save space
  df.attrs['corpus_name'] = corpus_name

  topics = get_topics(df, corpus_name=corpus_name)

  save_topic_barplots(topics, corpus_name=corpus_name)
  # save_topic_wordclouds(topics, corpus_name=corpus_name)

print('Done!')
