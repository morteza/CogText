# %%

# We're trying to learn two low-dimensional embeddings of tests and constructs.
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV_FILE = Path('data/pubmed/test_construct_matrix.csv')

PROBS = pd.read_csv(INPUT_CSV_FILE)
PROBS['p'] = PROBS['intersection_corpus_size'] / PROBS['union_corpus_size']

train, test = train_test_split(PROBS, test_size=0.2)

# %%
import keras
from keras.layers import Input, Embedding
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


n_tests, n_constructs = len(PROBS['test'].unique()), len(PROBS['construct'].unique())
embedding_dim = 3

# tests (M)
test_input = Input(shape=[1], name='test_input')
test_embedding = Embedding(input_dim=n_tests + 1,
                           output_dim=embedding_dim,
                           name='test_embedding')(test_input)
test_vec = keras.layers.Flatten(name='test_vec')(test_embedding)

# constructs (C)
construct_input = Input(shape=[1], name='construct')
construct_embedding = Embedding(input_dim=n_constructs + 1,
                                output_dim=embedding_dim,
                                name='construct_embedding')(construct_input)
construct_vec = keras.layers.Flatten(name='construct_vec')(construct_embedding)

# M*C
prod = keras.layers.Dot(axes=1)([test_vec, construct_vec])
model = keras.Model([test_input, construct_input], prod)
model.compile(optimizer='adam', loss='mse')

model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='HB')

# y = Dot(1, normalize=False)([user_vecs, item_vecs])
# model = Model(inputs=[user_id_input, item_id_input], outputs=y)
