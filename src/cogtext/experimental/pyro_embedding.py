#%%
from pathlib import Path
from numpyro.contrib.render import render_graph, render_model
import pandas as pd

import jax
import jax.numpy as jnp
import torch


import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam


PUBMED_PATH = Path('data/pubmed/abstracts_preprocessed.csv.gz')
MODEL_NAME = 'pubmed1pct_bertopic'
VERSION = 'v202109291'


rnd_key = jax.random.PRNGKey(0)

# indices = np.load(f'models/{MODEL_NAME}_{VERSION}.idx.npz')['arr_0']
# probs = np.load(f'models/{MODEL_NAME}_{VERSION}.probs.npz')['arr_0']
# PUBMED = pd.read_csv(PUBMED_PATH)
# PUBMED = PUBMED[PUBMED.index.isin(indices)]
# PUBMED['label'] = PUBMED['subcategory'].astype('category')
# labels = PUBMED['label'].cat.codes.values
# obs = probs

n_samples = 10000
n_labels = 100
n_topics = 1
n_categories = 2

cats = np.random.choice(n_categories, (n_samples,))
labels = np.random.choice(n_labels, (n_samples,))
obs = torch.tensor(np.random.normal((labels.shape[0], n_topics)))


# def model(category, label, data):

#   with pyro.plate('labels', n_labels):
#     z_mu = pyro.sample('z_mu', dist.HalfNormal(1.))
#     z_sigma = pyro.sample('z_sigma', dist.HalfNormal(1.))

#   # with numpyro.plate('data', data.shape[0]):
#   prob = pyro.sample('prob', dist.Normal(z_mu, z_sigma), obs=data)

#   print(z_mu.shape, z_sigma.shape, prob.shape)

#   return prob


# # SVI
# adam_params = {'lr': 0.0005, 'betas': (0.90, 0.999)}
# optimizer = Adam(adam_params)
# guide = AutoDiagonalNormal(model)
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
# svi_result = svi.step(cats, labels, obs)

# # render_model(model, cats, labels, obs)


import pymc3 as pm

with pm.Model() as model:
  z_mu = pm.HalfNormal(1., size=n_labels)[]
  z_sigma = pm.HalfNormal(1., n_labels)
  pm.Normal(z_mu, z_sigma)