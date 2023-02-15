import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from streamlit_yellowbrick import st_yellowbrick
from yellowbrick.cluster import KElbowVisualizer 


# Plot Explained Variance PCA
def explained_var(source):
    pca = PCA()
    pca.fit(source)  # X_scaled
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig1=px.area(
        x=range(1,  exp_var_cumul.shape[0] + 1), 
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}, 
        title="Explained Variance per Principal Component",
        height=500, width=800)
    fig1.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10)
    fig1.update_xaxes(tick0=1, dtick=1)
    return st.plotly_chart(fig1, use_container_width=True)


def silhouette_viz(source):
    silhouette_avg = []
    for i in range(2,11):
        labels=cluster.KMeans(n_clusters=i,random_state=200).fit(source).labels_
        silhouette_avg.append(metrics.silhouette_score(source, labels ,metric="euclidean",sample_size=1000,random_state=200))
    
    round_silhouette_avg = [round(num, 6) for num in silhouette_avg]
    fig = px.line(x=range(2,11), y=silhouette_avg, title='Silhouette analysis for optimal k', markers=True, 
        height=500, width=800,
        labels={
        'x':'Values of K',
        'y':'Silhouette score'
        }, text=round_silhouette_avg)  
    fig.update_traces(textposition='top right')
    return st.plotly_chart(fig, use_container_width=True)


def visualice_elbow(source):
    # Instantiate the clustering model and visualizer
    km = KMeans(random_state=42)
    titleKElbow = "The Optimal K-Cluster with Elbow Method"
    visualizer = KElbowVisualizer(km, k=(2,11), timings=False, size=(800, 450), title = titleKElbow, font_size=8)
    # Fit the data to the visualizer
    visualizer.fit(source) 
    # Format
    font = {'family' : 'normal', 'size'   : 9}   #'weight' : 'bold',
    plt.rc('font', **font)  
    plt.rcParams['text.color'] = 'grey'
    plt.grid(False)
    plt.xlabel('xlabel', fontsize=9)
    plt.xticks(fontsize=8, weight='normal')
    plt.ylabel('ylabel', fontsize=10)
    plt.yticks(fontsize=8, weight='normal')

    # Finalize and render the figure 
    visualizer.show()        
    return st_yellowbrick(visualizer)   