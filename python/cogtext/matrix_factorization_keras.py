# %%

# NOTE: Morty! switch to conda/tf environment before running this code.

# NOTE `conda install --name -c condaforge graphviz`
# NOTE `pip install pydot tensorflow`

# We're trying to learn two low-dimensional embeddings of tests and constructs.
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import keras
from keras.layers import Input, Embedding
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


INPUT_CSV_FILE = Path('data/pubmed/test_construct_matrix.csv')

PROBS = pd.read_csv(INPUT_CSV_FILE)

PROBS['test_id'] = PROBS['test'].astype('category').cat.codes.values
PROBS['construct_id'] = PROBS['construct'].astype('category').cat.codes.values

PROBS['p'] = PROBS['intersection_corpus_size'] / PROBS['union_corpus_size']

# train/test split
train, test = train_test_split(PROBS, test_size=0.2)

n_tests, n_constructs = len(PROBS['test'].unique()), len(PROBS['construct'].unique())

# TODO use hyperparameter
embedding_dim = 3

# tests (M)
M_input = Input(shape=[1], name='test_input')
M_embedding = Embedding(input_dim=n_tests,
                        output_dim=embedding_dim,
                        name='test_embedding')(M_input)
M_vec = keras.layers.Flatten(name='test_vec')(M_embedding)

# constructs (C)
C_input = Input(shape=[1], name='construct_input')
C_embedding = Embedding(input_dim=n_constructs,
                        output_dim=embedding_dim,
                        name='construct_embedding')(C_input)
C_vec = keras.layers.Flatten(name='construct_vec')(C_embedding)

# X = M*C
prod = keras.layers.Dot(axes=1, name='dot_product')([M_vec, C_vec])
model = keras.Model([M_input, C_input], prod)
model.compile(optimizer='adam', loss='mse')
# DEBUG model.summary()


# plot
model_dot_plot = model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='HB').create_png()
Image(model_dot_plot)

# DEBUG tf.keras.utils.plot_model(model)


# %%
history = model.fit([train['test_id'], train['construct_id']], train['p'], epochs=100, verbose=0)

pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.show()

results = model.evaluate((test['test_id'], test['construct_id']), test['p'], batch_size=1)

# %%

y_hat = model.predict((test['test_id'], test['construct_id']))
y_true = test['p']
mean_squared_error(y_true, y_hat)

# %% embeddings

C = model.get_layer(name='construct_embedding').get_weights()[0]
M = model.get_layer(name='test_embedding').get_weights()[0]

pd.DataFrame(C)
pd.DataFrame(M)

y_hat = (M @ C.T).flatten()
y_true = PROBS['p'].values

mean_squared_error(y_true, y_hat)

import seaborn as sns
sns.displot(y_true - y_hat, kde=True)
plt.show()