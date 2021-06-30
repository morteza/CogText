import numpy as np
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


class MatrixFactorization(torch.nn.Module):
  def __init__(self, rows, cols, n_embeddings):
    super().__init__()
    self.rows = rows
    self.cols = cols
    self.row_embeddings = torch.nn.Embedding(len(rows), n_embeddings, sparse=True)
    self.col_embeddings = torch.nn.Embedding(len(cols), n_embeddings, sparse=True)
    self.row_biases = torch.nn.Embedding(len(rows), 1, sparse=True)
    self.col_biases = torch.nn.Embedding(len(cols), 1, sparse=True)

  def forward(self, row, col):
    pred = self.row_biases(row) + self.col_biases(col)
    pred += (self.row_embeddings(row) * self.col_embeddings(col)).sum(dim=1, keepdim=True)
    return pred

  def fit(self, data):
    loss_func = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss() # torch.nn.MSELoss()
    optimizer = torch.optim.SGD(self.parameters(), lr=.001)

    # input features and outputs of the network
    X = data[['row', 'col']]
    y = data[['probability']]

    # prep data formats
    X['row'] = X['row'].apply(lambda i: np.where(self.row == i)[0][0])
    X['col'] = X['col'].apply(lambda i: np.where(self.col == i)[0][0])
    X = X[['row', 'col']]
    y['correct'].replace({True: 1., False: 0.}, inplace=True)

    # TODO loop epochs

    for batch in range(1000):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
      X_train, y_train = Variable(torch.from_numpy(X_train.values)), Variable(torch.from_numpy(y_train.values))

      optimizer.zero_grad()
      y_pred = self(X_train[:, 0], X_train[:, 1])
      loss = loss_func(y_pred, y_train)
      loss.backward()
      optimizer.step()

      # print(f'batch={batch}, loss={loss.item():.3f}')
      # fig, ax = plt.subplots(1,1,figsize=(10,5))
      # ax.scatter(X_train[:,0].data.numpy(), y_train.data.numpy(), color = 'orange')
      # ax.scatter(X_train[:,0].data.numpy(), y_pred.data.numpy(), color='green')
      # plt.show()

    return self
