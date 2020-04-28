import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal
import streamlit as st
import plotly.express as px
from scipy.stats import random_correlation
from sklearn.datasets import make_spd_matrix
import plotly.graph_objects as go
from gmm import GMM

algo = st.sidebar.selectbox("Algorithm", ['GMM', 'HMM'])

st.title("EM for %s"%algo)

st.sidebar.title("Parameters")

st.header("Initial data")
st.markdown("")

np.random.seed(4)

def generate_data(n_data, means, covariances, weights):
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

def plot_contours(data, means, covs, title, min_x, max_x, min_y, max_y, list_clusters):
	"""visualize the gaussian components over the data"""

	delta = 0.05
	k = means.shape[0]
	x = np.arange(min_x, max_x, delta)
	y = np.arange(min_y, max_y, delta)
	x_grid, y_grid= np.meshgrid(x, y)

	coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

	z_grid_list = []

	for i in range(k):
	    mean = means[i]
	    cov = covs[i]
	    z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
	    z_grid_list.append(z_grid)

	#fig2.add_scatter(x=data[:, 0], y=data[:, 1], mode="markers")

	fig2 = go.Figure(data=go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', marker_color=list_clusters), layout={
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
    })

	fig2.add_contour( # = go.Figure(data=go.Contour(
		z = z_grid_list[0],
		x = x_grid[0,:],
		y = y_grid[:,0],
		contours_coloring='lines',
		showscale=False,
    	line_width=2)

	fig2.update_layout(
              width=700, height=600)

	colorscale_list = ['Hot', 'Electric', 'Inferno', 'Bluered_r', 'Viridis', 'Cividis', 'RdBu']

	for i in range(1, k):

		fig2.add_contour(x=x_grid[0,:], y=y_grid[:,0], z = z_grid_list[i], contours_coloring='lines',
			showscale=False,
        	line_width=2, colorscale=colorscale_list[i])

	st.plotly_chart(fig2)

	#plt.title(title)
	#plt.tight_layout()

num_components = st.sidebar.slider("Number of components", 1,8,3)
num_iters = st.sidebar.slider("Number of iterations", 1,50,3)
num_data = st.sidebar.slider("Number of data points", 1,1000,100)

init_means = []
init_covariances = []
init_weights = []

min_val = -10
max_val = 10

for i in range(num_components):
	x = np.random.randint(min_val,max_val)
	y = np.random.randint(min_val,max_val)
	init_means.append([x,y])
	init_weights.append(np.random.uniform())
	init_covariances.append(make_spd_matrix(2))


init_means = np.array(init_means)
init_weights = np.array(init_weights)
init_weights = init_weights / sum(init_weights)

# generate data
X, list_clusters = generate_data(num_data, init_means, init_covariances, init_weights)

min_x = int(min(X[:, 0])) - 1
max_x = int(max(X[:, 0])) + 1

min_y = int(min(X[:, 1])) - 1
max_y = int(max(X[:, 1])) + 1

fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker_color=list_clusters), layout={
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
    })
fig.update_layout(
          width=700, height=600)

#fig = px.scatter(x=X[:, 0], y=X[:, 1], color=list_clusters, width=800, height=600)
fig.update_traces(marker=dict(showscale=False))

st.plotly_chart(fig)

st.header("After EM")
st.markdown("")
# use our implementation of the EM algorithm 
# and fit a mixture of Gaussians to the simulated data
gmm = GMM(n_components = num_components, n_iters = num_iters, tol = 1e-4, seed = 4)
gmm.fit(X)

plot_contours(X, gmm.means, gmm.covs, 'Initial clusters', min_x, max_x, min_y, max_y, list_clusters)
