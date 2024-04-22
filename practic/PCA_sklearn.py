import numpy as np
from sklearn.decomposition import PCA as PCA_sk

X = np.random.random((3, 3))

pca = PCA_sk(n_components=2)
pca.fit(X)

pca_X = pca.transform(X)

print(pca_X)