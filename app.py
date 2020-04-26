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
np.random.seed(5) #4 causes issue

from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import time

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
	X, list_clusters = generate_data(500, init_means, init_covariances, init_weights)
	X_list.append(X)
	clusters_list.append(list_clusters)

	i+=1

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

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

app.title = "EM for GMM and HMM"

# Add a title
app.layout = html.Div([  
    html.H1(
      children='EM for HMM and GMM',
      style={
         'textAlign': 'center'
      }
    ),
    
    dcc.Markdown(children= '''
        Expectation maximization is widely used in HMMs and GMMs.
    '''
    ),
    
    html.Br(),
    
    dcc.Tabs([
        dcc.Tab(label='EM for GMM', children=[

			html.Br(),

            # Two columns charts
            html.Div([
                html.Div([

		            html.H2('Parameters'),
		          
		            html.Br(),

		            html.Div([

		                html.Div([

		                	html.H5('Number of components'),

		                ], className="six columns"),

		                html.Div([

							dcc.Slider(
							    id='num_components',
							    min=1,
							    max=8,
							    step=1,
							    value=3,
							    marks={
							        1: '1',
							        2: '2',
							        3: '3',
							        4: '4',
							        5: '5',
							        6: '6',
							        7: '7',
							        8: '8'
							    },
							),

		                ], className="six columns")

		            ], className="row"),

					html.Br(),

		            html.Div([

		                html.Div([

		                	html.H5('Number of iterations'),

		                ], className="six columns"),

		                html.Div([

							dcc.Slider(
							    id='num_iters',
							    min=0,
							    max=50,
							    step=1,
							    value=0,
							    marks={
							        0: 'No EM',
							        10: '10',
							        20: '20',
							        30: '30',
							        40: '40',
							        50: '50'
							    },
							),

		                ], className="six columns"),

		            ], className="row"),

                ], className="six columns"),

                html.Div([
                ], className="six columns", id='output_viz'),

            ], className="row"),
            
        ]),

        # Second tab
        dcc.Tab(label='EM for HMM', children=[ ])        
    ])
], style={'width': '90%', 'textAlign': 'center', 'margin-left':'5%', 'margin-right':'0'})


@app.callback(
    dash.dependencies.Output('output_viz', 'children'),
    [dash.dependencies.Input('num_components', 'value'), dash.dependencies.Input('num_iters', 'value')])
def gen_fig(num_components, num_iters):

	X = X_list[num_components - 1]
	list_clusters = clusters_list[num_components - 1]

	min_x = int(min(X[:, 0])) - 1
	max_x = int(max(X[:, 0])) + 1

	min_y = int(min(X[:, 1])) - 1
	max_y = int(max(X[:, 1])) + 1

	if num_iters == 0:

		fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker_color=list_clusters), layout={
		        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
		    })

		fig.update_layout(
		          width=500, height=500) #plot_bgcolor='rgb(255,255,255)'

		fig.update_traces(marker=dict(showscale=False))
		return dcc.Graph(figure=fig)

	else:

		gmm = GMM(n_components = num_components, n_iters = num_iters, tol = 1e-4, seed = 4)
		gmm.fit(X)

		return plot_contours(X, gmm.means, gmm.covs, 'Initial clusters', min_x, max_x, min_y, max_y, list_clusters)


if __name__ == '__main__':
    app.run_server(debug=True)