# General
import numpy as np
from scipy.stats import norm, multivariate_normal
import time

# Data Generation
from sklearn.datasets import make_spd_matrix
from gmm import GMM
from datagen import generate_data, data_kmeans, gen_gmm_1d, generate_data_3d
from sklearn.cluster import KMeans

# Visualization and Dash
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from contours import plot_contours, plot_concat_contours

# Sound
import base64
from gender import pipeline

# Latex
import dash_defer_js_import as dji

# Vector Quantization
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.misc import face

# Background substraction
from PIL import Image

face = face(gray=True)
f2 = px.imshow(face, color_continuous_scale='gray')
f2.update_layout(
	          width=500, height=500)
f2.layout.coloraxis.showscale = False
f2.layout.margin= {'l': 0, 'r': 0, 't': 0, 'b': 0}

md0 = open("assets/intro.md", "r").read()
md = open("assets/gmm.md", "r").read()
md2 = open("assets/kmeans.md", "r").read()
md3 = open("assets/em_gmm.md", "r").read()
md4 = open("assets/gender.md", "r").read()
md5 = open("assets/vecquant.md", "r").read()
md6 = open("assets/back.md", "r").read()

# Initilize the random seed
np.random.seed(5) #4 causes issue

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css'], assets_external_path='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML')
app.scripts.config.serve_locally = False
server = app.server

sound_filename = 'gender/clips/female.wav'  # replace with your own .mp3 file
sound_female = base64.b64encode(open(sound_filename, 'rb').read())

sound_filename = 'gender/clips/male.wav'  # replace with your own .mp3 file
sound_male = base64.b64encode(open(sound_filename, 'rb').read())

e_png = 'images/e-step.png'
e_base64 = base64.b64encode(open(e_png, 'rb').read()).decode('ascii')

m_png = 'images/m-step.png'
m_base64 = base64.b64encode(open(m_png, 'rb').read()).decode('ascii')

app.title = "EM for GMM and HMM"

###### important for latex ######
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
'''

axis_latex_script = dji.Import(src="https://codepen.io/yueyericardo/pen/pojyvgZ.js")
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

X_list, clusters_list = generate_data()
X_3d, clusters_list3d = generate_data_3d()

# K-Means vs. GMM
X_k, clus_k = data_kmeans()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_k)
figk0 = go.Figure(data=go.Scatter(x=X_k[:, 0], y=X_k[:, 1], mode='markers', marker_color=kmeans.predict(X_k)), layout={
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}})
figk0.update_traces(marker=dict(showscale=False))
figk0.update_layout(width=700, height=500)
gmm = GMM(n_components = 2, n_iters = 75, tol = 1e-4, seed = 5)
gmm, ll = gmm.fit(X_k)
figk1 = plot_contours(X_k, gmm.means, gmm.covs, 'Initial clusters', -4, 4, -4, 4, clus_k)
figk1.update_layout(width=700, height=500)
ret, res = gen_gmm_1d()
fig3 = px.histogram(x=ret, color=res, marginal="rug")
fig3.update_layout(height=500)
fig3.layout.update(showlegend=False)

app.layout = html.Div([  
	html.Br(),

    html.H1(
      children='EM for HMM and GMM',
      style={
         'textAlign': 'center'
      }
    ),

    html.Br(),
    

    dcc.Tabs([

    	dcc.Tab(label="Info", children=[
    		html.Br(),
			html.Br(),

  			dcc.Markdown(md0, dangerously_allow_html=True, style={'textAlign': 'justify'}),

  			html.Br(),
			    
	    	html.Br(),
	    ]),
    	dcc.Tab(label="Slides", children=[
    		html.Br(),

    		html.Iframe(id="embedded-pdf", src="assets/EM.pdf", style={"width": "1000px", "height": "600px"}),

		]),

        dcc.Tab(label='EM for GMM', children=[

			html.Br(),

		    html.H3(
		      children='Introduction to GMMs',
		    ),

		    html.Br(),

  			dcc.Markdown(md, dangerously_allow_html=True, style={'textAlign': 'justify'}),

  			html.Br(),

  			html.H5('Generating data from GMMs in 1-dimension'),

  			dcc.Graph(figure=fig3),

            html.Br(),
            html.Br(),

  			#mathjax_script,
  			#axis_latex_script, 

  			html.H5('Generating data from GMMs in 2-dimensions'),

            # Two columns charts
            html.Div([

            	html.Br(),

                html.Div([

		            html.Div([

		                html.Div([

		                	html.H5('Number of components'),

		                ], className="six columns"),

		                html.Div([

							dcc.Slider(
							    id='num_components_gmm',
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

				dcc.Loading(
                    id="loading0",
                    children=[html.Div(id="output_inp0")],
                    type="default",
                ),

		        html.Br(),

                ], className="four columns"),

                html.Div([
                ], className="seven columns", id='output_gmm')

            ], className="row"),

            html.Br(),
            html.Br(),


  			html.H5('Generating data from GMMs in 3-dimensions'),

            # Two columns charts
            html.Div([

            	html.Br(),
            	html.Br(),

                html.Div([

		            html.Div([

		                html.Div([

		                	html.H5('Number of components'),

		                ], className="six columns"),

		                html.Div([

							dcc.Slider(
							    id='num_components_gmm3d',
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

	            html.Br(),

				dcc.Loading(
                    id="loading3d",
                    children=[html.Div(id="output_inp3d")],
                    type="default",
                ),

		        html.Br(),

                ], className="four columns"),

                html.Div([
                ], className="seven columns", id='output_gmm3d')

            ], className="row"),

		    html.Br(),

		    html.Hr(),
		    html.Hr(),

		    html.Br(),

		    # K-Means
			html.H3(
		      children='K-Means, a special case of GMM',
		    ),

  			dcc.Markdown(md2, dangerously_allow_html=True, style={'textAlign': 'justify'}),

		    html.Br(),

            # Two columns charts
            html.Div([
                html.Div([

		            html.H2('K-Means'),
		 
					dcc.Graph(figure=figk0),

		            html.Br(),

		        ], className="six columns"),

                html.Div([

		            html.H2('GMMs'),

		            dcc.Graph(figure=figk1),
		          
		            html.Br(),

		        ], className="six columns")

		    ], className="row"),

		    html.Br(),
	        html.Hr(),
	        html.Br(),

		    html.H3(
		      children='EM on Gaussian Data',
		    ),

		    html.Br(),


		    dcc.Markdown(md3, dangerously_allow_html=True, style={'textAlign': 'justify'}),

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

	            html.Br(),

				dcc.Loading(
                    id="loading",
                    children=[html.Div(id="output_inp")],
                    type="default",
                ),

		        html.Br(),

                ], className="six columns"),

                html.Div([
                ], className="six columns", id='output_viz')

            ], className="row"),
        
        html.Br(),

        html.Div(children=[], id='ll'),

	    html.Hr(),

	    html.H3(
	    	children='EM on GMM for gender detection',
	    ),

	    html.Br(),

		dcc.Markdown(md4, dangerously_allow_html=True, style={'textAlign': 'justify'}),

		html.Br(),

        # Two columns charts
        html.Div([
            html.Div([

            	html.H2('Recordings'),


            	html.Div([

            		html.Div([

		    		    html.H5("Male recording"),

		    		], className="six columns"),

            		html.Div([

		    		    html.Audio(src='data:audio/mpeg;base64,{}'.format(sound_male.decode()),
		              		controls=True, autoPlay=False,
		        		),
		        	], className="six columns"),

		        ], className="row"),


		        html.Div([

            		html.Div([

		    		    html.H5("Female recording"),
		    		    
		    		], className="six columns"),

            		html.Div([

		    		    html.Audio(src='data:audio/mpeg;base64,{}'.format(sound_female.decode()),
		              		controls=True, autoPlay=False,
		        		),
		        	], className="six columns"),

		        ], className="row"),


	            html.H2('Parameters'),
	          
	            html.Br(),

	            html.Div([

	                html.Div([

	                	html.H5('Number of components'),

	                ], className="six columns"),

	                html.Div([

						dcc.Slider(
						    id='num_components2',
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
						    id='num_iters2',
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

            html.Br(),

			dcc.Loading(
                id="loading2",
                children=[html.Div(id="output_inp2")],
                type="default",
            ),

	        html.Br(),

            ], className="six columns"),

            html.Div([
            ], className="six columns", id='output_viz2')

        ], className="row"),

	    html.Br(),
	    html.Hr(),
	    html.Br(),

        # Vector Quantization
        html.H3(children="Vector Quantization with k-Means"),

        html.Br(),

		dcc.Markdown(md5, dangerously_allow_html=True, style={'textAlign': 'justify'}),

	    html.Br(),

        html.Div([
            html.H5('Number of components'),

			dcc.Slider(
			    id='num_components_vec',
			    min=1,
			    max=8,
			    step=2,
			    value=3,
			    marks={
			        2: '2',
			        3: '3',
			        4: '4',
			        5: '5',
			        6: '6',
			        7: '7',
			        8: '8'
			    },
			),
		], style={'width': '40%'}),

        # Two columns charts
        html.Div([
            html.Div([

	            html.H5('Original image'),
	 
				dcc.Graph(figure=f2),

	            html.Br(),

	        ], className="six columns"),

            html.Div([

            	html.H5('Compressed image'),
            	html.Div([], id="vecquant")
	        ], className="six columns")

	    ], className="row"),

	    html.Br(),
	    html.Hr(),
	    html.Hr(),

	    html.Br(),

       	html.H3(children="GMM background substraction"),

       	html.Br(),

		dcc.Markdown(md6, dangerously_allow_html=True, style={'textAlign': 'justify'}),

	    html.Br(),

        # Two columns charts

        html.Div([

            html.H5('Choose frame'),

            dcc.Slider(
			    id='num_components_back',
			    min=1,
			    max=886,
			    step=1,
			    value=73,
			    #vertical=True
			),

            html.Br(),

        ], style={'width': '40%', 'textAlign': "center"}),

        html.Div([

        	html.H5('Substracted background'),
        	html.Div([], id="back")
        ]),

        html.Br(),

        ]),

        # Second tab
        dcc.Tab(label='EM for HMM', children=[ ]),
  
    ])
], style={'width': '85%', 'textAlign': 'center', 'margin-left':'7.5%', 'margin-right':'0'})


@app.callback(
    dash.dependencies.Output('vecquant', 'children'),
    [dash.dependencies.Input('num_components_vec', 'value')])
def vec_quant(n_clusters):

	X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
	k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
	k_means.fit(X)
	values = k_means.cluster_centers_.squeeze()
	labels = k_means.labels_

	# create an array from labels and values
	face_compressed = np.choose(labels, values)
	face_compressed.shape = face.shape

	f = px.imshow(face_compressed, color_continuous_scale='gray')

	f.update_layout(
	          width=500, height=500)

	f.layout.coloraxis.showscale = False
	f.layout.margin= {'l': 0, 'r': 0, 't': 0, 'b': 0}

	return dcc.Graph(figure=f)

@app.callback(
    dash.dependencies.Output('back', 'children'),
    [dash.dependencies.Input('num_components_back', 'value')])
def show_img(idx):

	image = Image.open('back/frame%s.jpg'%str(idx))
	data = np.asarray(image)
	fig = px.imshow(data, color_continuous_scale='gray')
	#fig.update_layout(
		          #width=800, height=800)
	fig.layout.coloraxis.showscale = False
	fig.layout.margin= {'l': 0, 'r': 0, 't': 0, 'b': 0}

	return dcc.Graph(figure=fig)

@app.callback(
    dash.dependencies.Output('output_inp3d', 'children'),
    [dash.dependencies.Input('num_components_gmm3d', 'value')])
def return_val3d(num_components):
	time.sleep(0.5)

@app.callback(
    dash.dependencies.Output('output_inp0', 'children'),
    [dash.dependencies.Input('num_components_gmm', 'value')])
def return_val(num_components):
	time.sleep(0.5)

@app.callback(
    dash.dependencies.Output('output_inp', 'children'),
    [dash.dependencies.Input('num_components', 'value'), dash.dependencies.Input('num_iters', 'value')])
def return_val(num_components, num_iters):
	if num_iters < 10:
		time.sleep(1)
	elif num_iters < 20:
		time.sleep(2)
	elif num_iters <40:
		time.sleep(3)
	else:
		time.sleep(4)

@app.callback(
    dash.dependencies.Output('output_inp2', 'children'),
    [dash.dependencies.Input('num_components2', 'value'), dash.dependencies.Input('num_iters2', 'value')])
def return_val2(num_components, num_iters):
	if num_iters < 10:
		time.sleep(1)
	elif num_iters < 20:
		time.sleep(2)
	elif num_iters <40:
		time.sleep(3)
	else:
		time.sleep(4)

@app.callback(
    dash.dependencies.Output('output_gmm3d', 'children'),
    [dash.dependencies.Input('num_components_gmm3d', 'value')])
def gen_gmm3d(num_components):

	X = X_3d[num_components - 1]
	list_clusters = clusters_list3d[num_components - 1]

	min_x = int(min(X[:, 0])) - 1
	max_x = int(max(X[:, 0])) + 1

	min_y = int(min(X[:, 1])) - 1
	max_y = int(max(X[:, 1])) + 1

	min_z = int(min(X[:, 1])) - 1
	max_z = int(max(X[:, 1])) + 1

	fig = go.Figure(data=go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker_color=list_clusters), layout={
	        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
	    })

	fig.update_layout(
	          width=700, height=500)

	fig.update_traces(marker=dict(showscale=False))
	return dcc.Graph(figure=fig)

@app.callback(
    dash.dependencies.Output('output_gmm', 'children'),
    [dash.dependencies.Input('num_components_gmm', 'value')])
def gen_gmm2d(num_components):

	X = X_list[num_components - 1]
	list_clusters = clusters_list[num_components - 1]

	min_x = int(min(X[:, 0])) - 1
	max_x = int(max(X[:, 0])) + 1

	min_y = int(min(X[:, 1])) - 1
	max_y = int(max(X[:, 1])) + 1

	fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker_color=list_clusters), layout={
	        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
	    })

	fig.update_layout(
	          width=700, height=500)

	fig.update_traces(marker=dict(showscale=False))
	return dcc.Graph(figure=fig)

@app.callback(
    [dash.dependencies.Output('output_viz', 'children'), dash.dependencies.Output('ll', 'children')],
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
		return [dcc.Graph(figure=fig), ""]

	else:

		gmm = GMM(n_components = num_components, n_iters = num_iters, tol = 1e-4, seed = 4)
		gmm, ll = gmm.fit(X)

		fig5 = px.line(x=range(len(ll)), y=ll, title='Log-likelihood through iterations')
		fig5.update_layout(
		    xaxis_title="Number of iterations",
		    yaxis_title="Likelihood",
		)

		return [dcc.Graph(figure=plot_contours(X, gmm.means, gmm.covs, 'Initial clusters', min_x, max_x, min_y, max_y, list_clusters)), dcc.Graph(figure=fig5)]

@app.callback(
    dash.dependencies.Output('output_viz2', 'children'),
    [dash.dependencies.Input('num_components2', 'value'), dash.dependencies.Input('num_iters2', 'value')])
def fig_gender(num_components, num_iters):

	male_train, female_train, male_male, male_female, female_male, female_female, gmm_male_means_, gmm_female_means_, gmm_male_covariances_, gmm_female_covariances_ = pipeline(num_components, num_iters)

	male_train = male_train[:5000]
	female_train = female_train[:5000]

	list_clusters_male = [1]*len(male_train)
	list_clusters_female = [0]*len(male_train)

	# For males

	min_x_male = int(min(male_train[:, 0])) - 1
	max_x_male = int(max(male_train[:, 0])) + 1

	min_y_male = int(min(male_train[:, 1])) - 1
	max_y_male = int(max(male_train[:, 1])) + 1

	min_x_female = int(min(female_train[:, 0])) - 1
	max_x_female = int(max(female_train[:, 0])) + 1

	min_y_female = int(min(female_train[:, 1])) - 1
	max_y_female = int(max(female_train[:, 1])) + 1

	if num_iters == 0:

		fig = go.Figure(data=go.Scatter(x=male_train[:, 0], y=male_train[:, 1], mode='markers', marker_color=list_clusters_male), layout={
		        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
		    })

		fig.update_traces(marker=dict(showscale=False))
		fig.add_scatter(x=female_train[:, 0], y=female_train[:, 1], mode='markers', marker_color=list_clusters_female)
		fig.update_layout(
		          width=500, height=500, showlegend=False)
		return dcc.Graph(figure=fig)

	else:

		return plot_concat_contours(male_train, gmm_male_means_, gmm_male_covariances_, 'Initial clusters', min_x_male, max_x_male, min_y_male, max_y_male, list_clusters_male, female_train, gmm_female_means_, gmm_female_covariances_, 'Initial clusters', min_x_female, max_x_female, min_y_female, max_y_female, list_clusters_female)

if __name__ == '__main__':
    app.run_server(debug=True)