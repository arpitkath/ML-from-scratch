import numpy as np


class PCA:

    def __init__(self, data):#'data' is a matrix containing features in columns.
        self.data = data
        self.dataT = np.array(self.mean_norm()).T
        self.cov_matrix = self.calc_cov()
        self.eigen_pair = self.get_eigen_pair()
        # Creating a dictionary for eigenvalues and eigenvectors.
        self.eigen_values = {}
        self.eigen_vectors = {}
        for i in range(len(self.eigen_pair)):
            self.eigen_values[i] = self.eigen_pair[i][0]
            self.eigen_vectors[i] = self.eigen_pair[i][1]
        self._num_of_comp = self.num_of_comp(sorted(self.eigen_values.values(), reverse=True))
        if self._num_of_comp == 0:
            self._num_of_comp = 1
        self.new_comp_matrix = np.array([self.eigen_vectors[i] for i in range(0,self._num_of_comp)])
        self.new_data = np.dot(self.new_comp_matrix, self.dataT).T

    # To get the eigenvalue-eigenvector pair in sorted order of eigenvalues.
    def get_eigen_pair(self):
        eig = np.linalg.eig(self.cov_matrix)
        eigen_pair = []
        for j in range(len(eig[0])):
            vector = []
            for i in range(len(eig[1])):
                vector.append(eig[1][i][j])
            tup = (eig[0][j], np.array(vector))
            eigen_pair.append(tup)

        # Sort the eigen-pairs according to their eigen-values.
        eigen_pair = sorted(eigen_pair, key=lambda x: x[0])
        eigen_pair = eigen_pair[::-1]
        return eigen_pair

    # To calculate the co-variance matrix of data.
    def calc_cov(self):
        cov_matrix = np.cov(self.dataT)
        return cov_matrix

    # Doing mean normalization of data.
    def mean_norm(self):
        for j in range(len(self.data[0])):
            mean = 0
            for i in range(len(self.data)):
                mean += self.data[i][j]
            mean /= len(self.data)
            for i in range(len(self.data)):
                self.data[i][j] -= mean
        return self.data




#data = [[1, 2, 3], [3, 8, 4], [9, 6, 11]]
#data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]

#p = PCA(data)
#print(p.new_data)