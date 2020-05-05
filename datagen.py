import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

np.random.seed(4)

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

def gen_gmm_param(shift=0):

    init_means = []
    init_covariances = []
    init_weights = []

    min_val = -10
    max_val = 10

    for j in range(6):
        x = np.random.randint(min_val,max_val)-shift
        y = np.random.randint(min_val,max_val)-shift
        init_means.append([x,y])
        init_weights.append(np.random.uniform())
        init_covariances.append(make_spd_matrix(2))

    init_means = np.array(init_means)
    init_weights = np.array(init_weights)
    init_weights = init_weights / sum(init_weights)
    
    return init_means, init_covariances, init_weights

def data_kmeans():

    mean_0 = [0,0]
    cov_0 = np.array([[1,0.8], [0.8, 1]])
    mean_1 = [0,0]
    cov_1 = np.array([[1,-0.8], [-0.8, 1]])
    
    init_means = [mean_0, mean_1]
    cov_init = np.array([cov_0, cov_1])
    init_means = np.array(init_means)

    X, list_clusters = make_data(1000, init_means, cov_init, [0.5, 0.5])

    return X, list_clusters

def gen_gmm_1d(mu1=-5, cov1 = 2, mu2 = 5, cov2= 4, num_data=5000):
    
    list_data = []
    
    list_cluster = []
    
    for k in range(num_data):
        val1 = np.random.normal(mu1, cov1)
        list_data.append(val1)
        list_cluster.append(1)

    for k in range(num_data):
        val2 = np.random.normal(mu2, cov2)
        list_data.append(val2)
        list_cluster.append(2)

    return list_data, list_cluster

def generate_data_3d():

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
            z = np.random.randint(min_val,max_val)
            init_means.append([x,y, z])
            init_weights.append(np.random.uniform())
            init_covariances.append(make_spd_matrix(3))

        init_means = np.array(init_means)
        init_weights = np.array(init_weights)
        init_weights = init_weights / sum(init_weights)

        # generate data
        X, list_clusters = make_data(500, init_means, init_covariances, init_weights)
        X_list.append(X)
        clusters_list.append(list_clusters)

        i+=1

    return X_list, clusters_list

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