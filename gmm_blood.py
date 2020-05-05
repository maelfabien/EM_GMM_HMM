#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#data manipulation
import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_html_components as html

#evaluation of the model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Dimensionality Reduction
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import KernelPCA

from sklearn.mixture import GaussianMixture

# defining functions and importing functions
#from gmm import GMM 
import contours
import plotly.graph_objects as go


def data_reduction_analysis(input_data, input_label):
    # function to produce perfom three types of dimensionality reduction
    
    #isomap
    isomap = Isomap(n_components=2)
    X_reduced_isomap = isomap.fit_transform(input_data)
    
    #tsne
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced_tsne = tsne.fit_transform(input_data)
     
    # PCA Analysis
    PCA_train_x = KernelPCA(n_components = 2).fit_transform(input_data)
    
    data_reduced = [X_reduced_isomap, X_reduced_tsne, PCA_train_x]
    return data_reduced

def plot_ic():

    # Reading the Electrical Impedance Tomography Data
    EIT_data = pd.read_excel('EIT_data.xlsx', sheet_name='Data')
    EIT_features = EIT_data.iloc[:, 2:]
    EIT_label = EIT_data['Class']
    
    # One-Hot encoding as there is several classes
    label_enconder = LabelEncoder()
    EIT_label = label_enconder.fit_transform(EIT_label)

    # Scaler
    scaler = StandardScaler()
    GMM_features = data_reduction_analysis(scaler.fit_transform(EIT_features), EIT_label)

    # Select one from 0-2: ISOMAP, tSNE and PCA
    GMM_features = GMM_features[2]
    GMM_label = EIT_label

    n_components = np.arange(1, 10)
    clfs = [GaussianMixture(n, max_iter = 1000).fit(GMM_features) for n in n_components]
    bics = [clf.bic(GMM_features) for clf in clfs]
    aics = [clf.aic(GMM_features) for clf in clfs]

    fig = go.Figure(data=go.Scatter(x=n_components, y=bics, mode='lines+markers', name='BIC'))
    fig.add_trace(go.Scatter(x=n_components, y=aics,
                    mode='lines+markers',
                    name='AIC'))

    fig.update_layout(
        title="AIC and BIC over the number of components",
        xaxis_title="AIC / BIC",
        yaxis_title="Number of components",
    )

    return dcc.Graph(figure=fig)


def plot_data(num_components=6, num_iters = 20):

    # Reading the Electrical Impedance Tomography Data
    EIT_data = pd.read_excel('EIT_data.xlsx', sheet_name='Data')
    EIT_features = EIT_data.iloc[:, 2:]
    EIT_label = EIT_data['Class']
    
    # One-Hot encoding as there is several classes
    label_enconder = LabelEncoder()
    EIT_label = label_enconder.fit_transform(EIT_label)

    # Scaler
    scaler = StandardScaler()
    GMM_features = data_reduction_analysis(scaler.fit_transform(EIT_features), EIT_label)

    # Select one from 0-2: ISOMAP, tSNE and PCA
    GMM_features = GMM_features[2]
    GMM_label = EIT_label

    gmm = GaussianMixture(n_components=num_components, max_iter=num_iters)

    # Creating the GMM model and fitting with the data
    #gmm = GMM(n_components=num_components, n_iters=num_iters, tol=1e-5, seed=10)
    gmm.fit(GMM_features)
    
    min_x = int(min(GMM_features[:, 0])*1.7)
    max_x = int(max(GMM_features[:, 0])*1.7)
    min_y = int(min(GMM_features[:, 1])*1.7)
    max_y = int(max(GMM_features[:, 1])*1.7)
    
    # Plotting the contours and the results 
    fig2 = contours.plot_contours(GMM_features, gmm.means_, gmm.covariances_, 'Initial clusters', 
                           min_x, max_x, min_y, max_y, GMM_label)

    return dcc.Graph(figure=fig2)
