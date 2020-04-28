import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

np.random.seed(5)

def make_data(n_data, means, covariances, weights):
    """creates a list of data points"""
    n_clusters, n_features = means.shape
    list_clusters = []

    data = np.zeros((n_data, n_features))
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size = 1, p = weights)[0]
        list_clusters.append(k)
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x
   
    return data, list_clusters

def generate_data():

	X_list = []
	clusters_list = []
	max_components = 8
	i = 1

	for k in range(max_components):

		init_means = []
		init_covariances = []
		init_weights = []

		min_val = -10
		max_val = 10

		for j in range(i):
			x = np.random.randint(min_val,max_val)
			y = np.random.randint(min_val,max_val)
			init_means.append([x,y])
			init_weights.append(np.random.uniform())
			init_covariances.append(make_spd_matrix(2))

		init_means = np.array(init_means)
		init_weights = np.array(init_weights)
		init_weights = init_weights / sum(init_weights)

		# generate data
		X, list_clusters = make_data(500, init_means, init_covariances, init_weights)
		X_list.append(X)
		clusters_list.append(list_clusters)

		i+=1

	return X_list, clusters_list