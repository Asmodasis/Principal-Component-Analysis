import pca as pca
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(input_dir: str) -> np.ndarray:
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    return np.array([plt.imread(image).flatten('F').astype(float) for image in files])


def compress_images(DATA: np.ndarray, k: int) -> None:
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    X_compressed = Z_star.dot(PCS.T[:100])


    if not os.path.exists('./Output'):
        os.mkdir('./Output')
    for i, flattened_image in enumerate(X_compressed):
        plt.imsave('./Output/100output' + str(i) + '.png', flattened_image.reshape((60, 48), order='F').real.astype('float'), cmap='gray')


if __name__ == '__main__':
    X = load_data('Data/Train/')
    compress_images(X,100)
    X = np.array([[-1 ,-1] ,[-1 ,1] ,[1 , -1] ,[1 ,1]]) 
    Z = pca.compute_Z(X)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, 1 , 0)
