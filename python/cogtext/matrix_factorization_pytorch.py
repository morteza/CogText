# %%
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


class MFNet(nn.Module):
  def __init__(self, n_tasks, n_constructs, n_embeddings):
    super(MFNet, self).__init__()
    self.task_embeddings = nn.Embedding(n_tasks, n_embeddings)
    self.construct_embeddings = nn.Embedding(n_constructs, n_embeddings)
    self.task_biases = torch.nn.Embedding(n_tasks, 1)
    self.construct_biases = torch.nn.Embedding(n_constructs, 1)
    self.decoder = nn.Linear(n_embeddings)

  def forward(self, task, construct):
    H = self.task_biases(task) + self.construct_biases(construct)
    H += (self.task_embeddings(task) * self.construct_embeddings(construct))
    y = self.decoder(H)
    return y

  def fit(self, data, n_epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=.001)

    # input features and outputs of the network
    X = data[['label_1', 'label_2']]
    y = data[['probability']]

    for epoch in range(n_epochs):
      X_train = X
      y_train = y
      optimizer.zero_grad()
      y_pred = self(X_train[:, 0], X_train[:, 1])
      loss = criterion(y_pred, y_train)
      loss.backward()
      optimizer.step()

      print(f'epoch={epoch}, loss={loss.detach().item():.3f}')

    return self


# prep data
from python.cogtext.utils import select_relevant_journals

INPUT_FILE = 'data/pubmed_abstracts.csv.gz'
PUBMED = (pd.read_csv(INPUT_FILE)
            .pipe(select_relevant_journals)
            .dropna(subset=['abstract']))

# only corpora with # of articles < DEV_MAX_CORPUS_SIZE
# subcats_cnt = PUBMED['label'].value_counts()
# small_subcats = subcats_cnt[subcats_cnt < DEV_MAX_CORPUS_SIZE].index.to_list()
# PUBMED = PUBMED.query('label in @small_subcats',).copy()

# DROP tasks/constructs with less than 5 articles (1/test + 1/valid + 4/train = 6)
valid_subcats = PUBMED['label'].value_counts()[lambda cnt: cnt > 5].index.to_list()
PUBMED = PUBMED.query('label in @valid_subcats')

# train/test split (80% train 20% test)
PUBMED_train, PUBMED_test = train_test_split(
    PUBMED,
    test_size=0.2,
    stratify=PUBMED['label'],
    random_state=0)
