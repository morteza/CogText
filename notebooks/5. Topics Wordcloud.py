"""Here, topics will be extracted for each cognitive test/construct corpus and words in each topic will be
visualized as a word-cloud. It will produce one figure per corpus within that it contains one word-cloud per topic.

Note: All accesses to files assume current working directory is the project root folder.

Note: The fastest coherency scoring is `u_mass`, however, the default is `c_v`. I'm using `u_mass` here to
      speed things up on local machines.
"""

# %%
import sys
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm
import gensim

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('notebook')

# constraints and constants
N_TOPICS_MIN = 10
N_TOPICS_MAX = 30
N_TOPIC_WORDS = 10  # number of words per topic (for printing and plotting purposes)


def get_topics(texts: pd.DataFrame, corpus_name: str):

  texts['abstract'].fillna(texts['title'], inplace=True)

  tqdm.pandas(desc='Preprocessing')
  texts['preprocessed_abstract'] = \
      texts['abstract'].progress_apply(lambda abstract: gensim.parsing.preprocess_string(abstract))

  docs = texts['preprocessed_abstract'].to_list()
  words = gensim.corpora.Dictionary(docs)
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
  topics_df['term_score'] = topics_df['term'].apply(lambda x: x[0])
  topics_df['term'] = topics_df['term'].apply(lambda x: x[1])
  topics_df.attrs['corpus_name'] = corpus_name

  return topics_df


def plot_and_save_wordcloud(topics: pd.DataFrame,
                            corpus_name: str,
                            savefig_fname: Path = None):
  if savefig_fname is None:
    # default path
    savefig_fname = Path('outputs/topics_wordcloud') / (corpus_name.replace('/', '_') + '.histogram.png')

  grid = sns.FacetGrid(topics,
                       col='topic_index', hue='topic_index', palette='deep', col_wrap=4, sharex=False, sharey=False,
                       col_order=range(1, topics['topic_index'].nunique() + 1, 1))
  grid.map_dataframe(sns.barplot, x='term_score', y='term', orient='h', dodge=False)
  plt.suptitle(f'{corpus_name} topics', weight='bold', x=.2)
  plt.tight_layout()
  # plt.show()
  plt.savefig(savefig_fname)


# load and iterate corpora
corpora = Path('data/pubmed').glob('**/*.csv')

for fname in corpora:
  corpus_name = re.findall('.*/pubmed/(.*)\\.csv', str(fname))[0]
  print(f'>> {corpus_name}...', file=sys.stderr)
  df = pd.read_csv(fname)

  # store test/construct names in the dataframe attribute to save space
  df.attrs['corpus_name'] = corpus_name

  topics = get_topics(df, corpus_name=corpus_name)

  plt.figure()
  plot_and_save_wordcloud(topics, corpus_name=corpus_name)
  plt.close()   # to save memory and avoid memory leak warning

print('Done!')
