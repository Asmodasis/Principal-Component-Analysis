import numpy as np
from typing import Tuple


def compute_Z(X: np.ndarray, centering=True, scaling=False) -> np.ndarray:
    Z = X.copy()
    if centering: Z = Z - np.average(Z, axis=0)
    if scaling: Z = Z / np.std(Z, axis=0)
    return Z


def compute_covariance_matrix(Z: np.ndarray) -> np.ndarray:
    #print("{TEST} shape of Z.T.dot(Z) is   ", Z.T.dot(Z).shape) #REMOVE
    #print("{TEST} shape of Z COVARIACNCE   ", Z.T.shape) #REMOVE

    #X = Z.zeros

    #for row in Z:
    #    for col in Z[row]:
    #        x[][] = 


    return (Z.T).dot(Z)

def dotProduct(A, B):                                                   # computes the dot product
  if len(A) != len(B):                                                  # A and B have to be same size to dot product
    raise Exception("Can't Compute the Dot Product between varying size vectors")
                                                                        # raise error if they are not
  else:                                                             
    amount = 0
    for i, j  in A, B:                                                  # loop through A and B and multiply 
      amount += i*j                                                     # add amount together
  return amount                                                         # return the sum of the multipliers


def find_pcs(COV: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #print("{TEST} FIND_PCS shape of COV is   ", COV.shape) #REMOVE
    eigenValue, eigenVector = np.linalg.eig(COV)
    sortIndex = (-eigenValue).argsort()                                     # numpy sort, returns an array of sorted indices, negative set gets sorted largest to smalled
    eigenValue = eigenValue[sortIndex]                                      # sorts the eigenvalues based on the index array
    eigenVector = eigenVector.T[sortIndex].T                                # sorts the eigenvectors based on the index array
    #print("{TEST} The eigenvector is ",  eigenVector) #REMOVE
    return eigenValue, eigenVector


def project_data(Z: np.ndarray, PCS: np.ndarray, L: np.ndarray, k: int, var: float) -> np.ndarray:
    if k <= 0: k = _find_k_(L, var)
    #print("{TES}  Z.dot(PCS[:, :k]) is ",  Z.dot(PCS[:, :k])) #REMOVE
    return Z.dot(PCS[:, :k])


def _find_k_(L: np.ndarray, var: float) -> int:
    sum = L.sum()
    partial_sum = 0
    for k, eigenvalue in enumerate(L):
        partial_sum += eigenvalue
        if (partial_sum) / sum >= var: return k
    return -1
