# General
import numpy as np
from scipy.stats import norm, multivariate_normal
import time

# Data Generation
from sklearn.datasets import make_spd_matrix
from gmm import GMM
from datagen import generate_data

# Visualization and Dash
import plotly.graph_objects as go
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

md = open("description_gmm.md", "r").read()

# Initilize the random seed
np.random.seed(5) #4 causes issue

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css'], assets_external_path='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML')
app.scripts.config.serve_locally = False
server = app.server

sound_filename = 'gender/clips/female.wav'  # replace with your own .mp3 file
sound_female = base64.b64encode(open(sound_filename, 'rb').read())

sound_filename = 'gender/clips/male.wav'  # replace with your own .mp3 file
sound_male = base64.b64encode(open(sound_filename, 'rb').read())


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

		    html.H3(
		      children='Introduction to GMMs',
		    ),

		    html.Br(),

  			dcc.Markdown(md, dangerously_allow_html=True, style={'textAlign': 'justify'}),

  			html.Br(),

  			mathjax_script,
  			axis_latex_script, 

		    html.Hr(),

		    html.H3(
		      children='EM on Gaussian Data',
		    ),

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
            


	    html.Hr(),

	    html.H3(
	    	children='EM on GMM for gender detection',
	    ),

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


        ]),

        # Second tab
        dcc.Tab(label='EM for HMM', children=[ ]),

        # Third tab
        dcc.Tab(label='EM for GMM/HMM', children=[ ])        
    ])
], style={'width': '90%', 'textAlign': 'center', 'margin-left':'5%', 'margin-right':'0'})

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
	#time.sleep(num_iters*0.15)

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