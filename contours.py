# General
import numpy as np
from scipy.stats import multivariate_normal

# Visualization
import plotly.graph_objects as go
import dash
import dash_core_components as dcc

def plot_concat_contours(data, means, covs, title, min_x, max_x, min_y, max_y, list_clusters, data2, means2, covs2, title2, min_x2, max_x2, min_y2, max_y2, list_clusters2):
	"""visualize the gaussian components over the data"""

	x_min = min(min_x, min_x2)
	x_max = min(max_x, max_x2)
	y_min = min(min_y, min_y2)
	y_max = min(max_y, max_y2)

	delta = 0.05
	k = means.shape[0]
	x = np.arange(x_min, x_max, delta)
	y = np.arange(y_min, y_max, delta)
	x_grid, y_grid= np.meshgrid(x, y)

	coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

	z_grid_list = []
	z_grid_list2 = []

	for i in range(k):
	    mean = means[i]
	    cov = covs[i]
	    z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
	    z_grid_list.append(z_grid)

	    mean = means2[i]
	    cov = covs2[i]
	    z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
	    z_grid_list2.append(z_grid)

	fig3 = go.Figure(data=go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', marker_color=list_clusters, opacity=0.5), layout={
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}})

	fig3.add_scatter(x=data2[:, 0], y=data2[:, 1], mode='markers', marker_color=list_clusters2, opacity=0.5)

	fig3.add_contour( # = go.Figure(data=go.Contour(
		z = z_grid_list[0],
		x = x_grid[0,:],
		y = y_grid[:,0],
		contours_coloring='lines',
		showscale=False,
    	line_width=2, colorscale=['blue', 'blue', 'blue'])

	fig3.add_contour( # = go.Figure(data=go.Contour(
		z = z_grid_list2[0],
		x = x_grid[0,:],
		y = y_grid[:,0],
		contours_coloring='lines',
		showscale=False,
    	line_width=2, colorscale=['red', 'red', 'red'])

	fig3.update_layout(
              width=500, height=500, showlegend=False)

	colorscale_list = ['Hot', 'Electric', 'Inferno', 'Bluered_r', 'Viridis', 'Cividis', 'RdBu', 'Hot', 'Electric', 'Inferno', 'Bluered_r', 'Viridis', 'Cividis', 'RdBu']

	for i in range(1, k):

		fig3.add_contour(x=x_grid[0,:], y=y_grid[:,0], z = z_grid_list[i], contours_coloring='lines',
			showscale=False, 
        	line_width=2, colorscale= ['blue', 'blue', 'blue'] )

		fig3.add_contour(x=x_grid[0,:], y=y_grid[:,0], z = z_grid_list2[i], contours_coloring='lines',
			showscale=False, 
        	line_width=2, colorscale=['red', 'red', 'red'])

	return dcc.Graph(figure=fig3)

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