import numpy as np


def svd_embedding(mat, k: int) -> np.array:
  U, S, VT = np.linalg.svd(mat, full_matrices=False)

  U = U[:, :k]
  S = S[:k]
  VT = VT[:k, :]

  return U, S, VT, np.dot(U * S, VT)
