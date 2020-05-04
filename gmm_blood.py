#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#data manipulation
import pandas as pd
import numpy as np

#evaluation of the model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Dimensionality Reduction
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import KernelPCA

#visuaization 
import plotly.io as pio
png_renderer = pio.renderers["png"]
png_renderer.width = 500
png_renderer.height = 500

pio.renderers.default = "png"

# defining functions and importing functions
from gmm import GMM 
import contours

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


# Main
if __name__ == '__main__':
    
    # Reading the Electrical Impedance Tomography Data
    EIT_data = pd.read_excel('EIT_data.xlsx', sheet_name='Data')
    EIT_features = EIT_data.iloc[:, 2:]
    EIT_label = EIT_data['Class']
    
    # One-Hot encoding as there is several classes
    label_enconder = LabelEncoder()
    EIT_label = label_enconder.fit_transform(EIT_label)

    
    # Scaler
    scaler = StandardScaler()
    
    GMM_features = data_reduction_analysis(scaler.fit_transform(EIT_features), 
                                                 EIT_label)
    # Select one from 0-2: ISOMAP, tSNE and PCA
    GMM_features = GMM_features[2]
    GMM_label = EIT_label

    # Creating the GMM model and fitting with the data
    gmm = GMM(n_components=6, n_iters=20, tol=1e-5, seed=10)
    
    gmm.fit(GMM_features)
    
    min_x = int(min(GMM_features[:, 0])*1.7)
    max_x = int(max(GMM_features[:, 0])*1.7)
    min_y = int(min(GMM_features[:, 1])*1.7)
    max_y = int(max(GMM_features[:, 1])*1.7)
    
    
    # Plotting the contours and the results 
    fig2 = contours.plot_contours(GMM_features, gmm.means, gmm.covs, 'Initial clusters', 
                           min_x, max_x, min_y, max_y, GMM_label)
    fig2.show()
    
