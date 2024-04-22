import numpy as np

class PCA:
    methods: list[str] = ['svd', 'eigen']

    def __init__(self,
                 n_cut: int=2,
                 method: str='svd'
                 ) -> None:
        if method not in self.methods:
            raise ValueError(f"'{method}' is not a method implemented in this model")

        self.__n_cut = n_cut
        self.__method = method

    def fit(self,
            X: np.ndarray
            ) -> 'PCA':
        if self.__method == self.methods[0]:
            U, S, V = np.linalg.svd(a=X)
            self.__V = V[:self.__n_cut, :]
        
        elif self.__method == self.methods[1]:
            corr_mat = np.corrcoef(X.T)

            self.eig_vals, self.eig_vecs = np.linalg.eig(corr_mat)
            self.eig_pairs = [
                (np.abs(self.eig_vals[i]), self.eig_vecs[:, i])
                for i in range(len(self.eig_vals))
            ]
            self.eig_pairs.sort(key=lambda x: x[0], reverse=True)

            total = sum(self.eig_vals)
            self.explained_variance_ratio = [
                (i/total) * 100
                for i in sorted(self.eig_vals, reverse=True)
            ]
            self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

            self.matrix_w = np.hstack(
                list((
                    self.eig_pairs[i][1].reshape(np.size(X, axis=1), 1)
                    for i in range(self.__n_cut)
                ))
            )
        
        return self
    
    def transform(self, 
                  X: np.ndarray
                  ) -> np.ndarray:
        if self.__method == self.methods[0]:
            return X.dot(self.__.T)

        elif self.__method == self.methods[1]:
            return X.dot(self.matrix_w)