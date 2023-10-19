# %%
import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp

tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root

# Data

# synthesized data parameteres
n_samples = 10000
n_labels = 64
n_topics = 12
n_categories = 2

# mock data
data = pd.DataFrame({
    'category': np.random.choice(n_categories, (n_samples,)),
    'label': np.random.choice(n_labels, (n_samples,)),
})
features = stats.truncnorm(0, 1).rvs(size=(n_samples, n_topics))
data = pd.concat([data, pd.DataFrame(features)], axis=1)

# test/train split
train_data = data.sample(frac=.8, random_state=0)
test_data = data.drop(train_data.index)


# Model
def model():
  rv_cat = yield Root(tfd.Categorical(tf.ones(n_categories)/n_categories, name='category'))
  cat_to_lbl = tf.constant(np.ones(n_categories), dtype='float32')[rv_cat]
  rv_lbl = yield tfd.Categorical(tf.ones(n_labels)/n_labels, name='label')
  lbl_to_prb = tf.constant(np.ones(n_labels), dtype='float32')[rv_lbl]
  rv_prb = yield tfd.HalfNormal(lbl_to_prb, name='prob')

joint = tfd.JointDistributionCoroutineAutoBatched(model)

# Model
X = data['label'].astype('category').cat.codes.values
y = data.drop(columns=['category','label'])

model = tf.keras.Sequential()
model.add(layers.Embedding(n_labels, n_topics))
model.compile('adam', 'mse')
history = model.fit(X, y, epochs=5)
y_pred = model.predict(data['label'])

# label embedding
H = model.get_layer(index=0).get_weights()[0]

# plot loss history
plt.plot(np.arange(len(history.history['loss'])), history.history['loss'])
plt.suptitle('training loss history')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

g = sns.clustermap(H)
g.ax_heatmap.set(xlabel='embedding dim', ylabel='label')
plt.show()
