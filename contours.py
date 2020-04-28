# General
import numpy as np
from scipy.stats import multivariate_normal

# Visualization
import plotly.graph_objects as go
import dash
import dash_core_components as dcc

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
              width=500, height=500)

	colorscale_list = ['Hot', 'Electric', 'Inferno', 'Bluered_r', 'Viridis', 'Cividis', 'RdBu', 'Hot', 'Electric', 'Inferno', 'Bluered_r', 'Viridis', 'Cividis', 'RdBu']

	for i in range(1, k):

		fig2.add_contour(x=x_grid[0,:], y=y_grid[:,0], z = z_grid_list[i], contours_coloring='lines',
			showscale=False,
        	line_width=2, colorscale=colorscale_list[i])

	return dcc.Graph(figure=fig2)