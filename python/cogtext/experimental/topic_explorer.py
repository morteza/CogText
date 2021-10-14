import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
from top2vec import Top2Vec
from pathlib import Path

from typing import Tuple

import matplotlib.pyplot as plt

st.title('Topic Explorer')


model_name = st.selectbox('Model Name', ['pubmed10pct_bertopic','pubmed10pct_top2vec'])
version = st.text_input('Model Version', 'v202110111')
dataset = st.selectbox('Dataset', list(Path('data').glob('pubmed_abstracts*.csv.gz')))

# @st.experimental_singleton
# def load_bertopic_model(version, dataset) -> Tuple[BERTopic, pd.DataFrame, np.array, np.array]:
#   indices = np.load(f'models/{model_name}_{version}.idx.npz')['arr_0']
#   model: BERTopic = BERTopic.load(f'models/{model_name}_{version}.model')
#   topics = np.load(f'models/{model_name}_{version}.topics.npz')['arr_0']
#   scores = np.load(f'models/{model_name}_{version}.probs.npz')['arr_0']

#   # documents
#   data = pd.read_csv(dataset)
#   data = data[data.index.isin(indices)]
#   data['label'] = data['subcategory'].astype('category')

#   return (model, data, topics, scores)


# @st.experimental_singleton
# def load_top2vec_model(version, dataset) -> Tuple[Top2Vec, pd.DataFrame, np.array, np.array]:

#   model: Top2Vec = Top2Vec.load(f'models/{model_name}_{version}.model')

#   # documents
#   data = pd.read_csv(dataset)
#   data = data.query('pmid in @model.document_ids').copy()
#   data['label'] = data['subcategory'].astype('category')

#   # scores data frame
#   scores = np.load(f'models/{model_name}_{version}.scores.npz')['arr_0']
#   scores = data['pmid'].apply(lambda pmid: pd.Series(scores[model.doc_id2index[pmid]]))
#   scores['pmid'] = data['pmid']
#   scores['doc_id'] = scores['pmid'].apply(lambda pmid: model.doc_id2index[pmid])
#   scores['label'] = data['label']

#   topics = model.get_topics()

#   return (model, data, topics, scores)


# model, _, topics, _ = load_top2vec_model(version, dataset)
# st.write(pd.DataFrame(topics[0]))

# # n_topics = model.get_num_topics()
# st.progress(0)

# for i in range(n_topics):
#   st.progress(i/n_topics)
#   model.generate_topic_wordcloud(i)
#   st.pyplot(plt)

# st.progress(1.0)

# model, data, topics, scores = load_bertopic_model(version, dataset)

# model_topics = model.get_topics()


# def get_topic_rep(topic_id):
#   words = model_topics[topic_id]
#   words.sort(key=lambda x: x[1], reverse=True)
#   return '/'.join([w[0] for w in words])


# # DEBUG irrelevant_topics = st.multiselect('Irrelevant Topics:', model_topics, format_func=lambda x: get_topic_rep(x))

# cboxes = []

# st.markdown('### Mark irrelevant topics')
# for topic_id in model.get_topics():
#   lbl = get_topic_rep(topic_id)
#   cboxes.append(st.checkbox(lbl))

# irrelevant_topics = [i - 1 for i, c in enumerate(cboxes) if c]

# st.sidebar.markdown('# Irrelevant Topics')
# st.sidebar.markdown(' '.join([f'`{i}`' for i in irrelevant_topics]))

# if -1 in irrelevant_topics:
#   irrelevant_topics.remove(-1)

# relevant_probs = np.delete(probs, irrelevant_topics, axis=1)
