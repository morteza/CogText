import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

from typing import Tuple

st.title('Topic Explorer')


version = st.text_input('Model Version', 'v202109291')
dataset = st.selectbox('Dataset', list(Path('data').glob('pubmed_abstracts*.csv.gz')))

@st.experimental_singleton
def load_model(version, dataset) -> Tuple[BERTopic, pd.DataFrame, np.array, np.array]:
  indices = np.load(f'outputs/models/pubmed1pct_bertopic_{version}.idx.npz')['arr_0']
  model: BERTopic = BERTopic.load(f'outputs/models/pubmed1pct_bertopic_{version}.model')
  topics = np.load(f'outputs/models/pubmed1pct_bertopic_{version}.topics.npz')['arr_0']
  probs = np.load(f'outputs/models/pubmed1pct_bertopic_{version}.probs.npz')['arr_0']

  data = pd.read_csv(dataset)
  data = data[data.index.isin(indices)]

  return (model, data, topics, probs)


model, data, topics, probs = load_model(version, dataset)

model_topics = model.get_topics()

def get_topic_rep(topic_id):
  words = model_topics[topic_id]
  words.sort(key=lambda x: x[1], reverse=True)
  return '/'.join([w[0] for w in words])


irrelevant_topics = st.multiselect('Irrelevant Topics:', model_topics, format_func=lambda x: get_topic_rep(x))

if -1 in irrelevant_topics:
  irrelevant_topics.remove(-1)

relevant_probs = np.delete(probs, irrelevant_topics, axis=1)
st.write(relevant_probs.shape)
