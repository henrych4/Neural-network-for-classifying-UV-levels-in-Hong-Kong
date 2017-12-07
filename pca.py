import numpy as np
import argparse

from sklearn.decomposition import PCA, RandomizedPCA

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()

data = np.load(args.data)['data']
data = np.random.permutation(data)

X = data[:, :-1]
y = data[:, -1]

pca = PCA(n_components=args.n)
pca.fit(X)
print(pca.explained_variance_ratio_)

X_pca = pca.fit_transform(X)
y = np.asarray([y])
input_xy = np.concatenate((X_pca, y.T), axis=1)
np.savez("data_reduced_{}".format(args.n), data=input_xy)
