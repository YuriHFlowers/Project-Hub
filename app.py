
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_ext as ste
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from Utils.colors import *
from Utils.visualizations import *
from Utils.download import *
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

st.set_page_config(layout="wide")
left, middle, right = st.columns([1, 5, 1])

with middle:
    st.markdown("<h1 style='text-align: center; color: #FF0000;'>K-Means Cluster Analysis</h1>", unsafe_allow_html=True)  
    uploadFile = st.container()
    with uploadFile:
        st.markdown("#### Please Upload a csv File")
        uploaded_file = st.file_uploader(" ") 
        if uploaded_file is not None and uploaded_file.type=='text/csv': 
                data = pd.read_csv(uploaded_file, sep=',')

    colors = get_colors()

    columns = ['Corr_Somme par jour_rain',
                'Corr_Somme par jour_temperaturefeelslike',
                'Corr_Somme par jour_windspeed',
                'Corr_Somme par jour_humidity',
                'Corr_Weekday_Somme par jour_temperaturefeelslike',
                'Corr_Somme par jour_pressure',
                'Corr_Somme par jour_visibility',
                'part_1_relat',
                'part_2_relat',
                'part_3_relat',
                'ratio_school',
                'ratio_public',
                'ratio_Saturday',
                'ratio_Sunday',
                'cv_Monday',
                'cv_Tuesday',
                'cv_Wednesday',
                'cv_Thursday',
                'cv_Friday',
                'cv_Saturday',
                'cv_Sunday',
                'cv_weekday',
                'cv_ratio_Saturday',
                'cv_ratio_Sunday',
                'Coef_Var', 'startdate',
        ]

    sidebar = st.sidebar
    st.sidebar.title('Select Data')

    with st.form('form'):
        features = sidebar.multiselect("Select variables", columns, help='Select a variable.')  
        drop_na = st.checkbox('Drop rows with missing values', value=True)          
        drop = st.checkbox('Drop  sensors that started after June 2022', value=True)         
        submitted = st.form_submit_button("Submit")

    # Select to use PCA
    use_pca = sidebar.radio("Use PCA?",["Yes","No"], help='If you select less than 3 variables not use PCA.')  
    # Two Principal Components
    comp = sidebar.number_input("Principal Components",1,2,2, help='2 Principal Components')   

    # Select number of clusters
    ks = sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=2, help='Select number of clusters for the visualization')

    if submitted:
        try:
            udf = data[features]
            if drop_na:
                udf = data.sort_values("zone_id").dropna(subset=features)
            if drop:
                udf = data.loc[(data['startdate'] < '2022-06-01')]
                udf.reset_index(drop=True, inplace=True)
        except NameError:
            st.markdown("## No File :file_folder:")  
        
        if len(features)>=2:
            tdf= udf.copy()
            X =  tdf[features]  
            if use_pca=="No":
                st.markdown("### Not Using PCA")
                for c in range(1,ks+1):
                    X = tdf[features]                
                    model = KMeans(n_clusters=c)
                    model.fit(X)
                    y_kmeans = model.predict(X)
                    tdf["cluster"] = y_kmeans 

                    fig = go.Figure()
                    trace0 = go.Scatter(x=tdf[features[0]],y=tdf[features[1]],mode='markers+text', text=tdf["Quartier"],
                                        textposition="top center",
                                        textfont_size=11,
                                        marker=dict(
                                                size=tdf.avg_visits_perSensor,
                                                color=tdf.cluster.apply(lambda x: colors[x]),
                                                sizemode='area',
                                                sizeref=3.*max(tdf.avg_visits_perSensor)/30.**2,
                                                opacity = 0.9,
                                                ),
                                        name="Zone", )  
                                    
                    trace1 = go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1],
                                        mode='markers', 
                                        marker=dict(
                                            color=colors,
                                            size=10,
                                            symbol="diamond",
                                            showscale=True,
                                            line = dict(
                                                width=1,
                                                color='rgba(231, 63, 116)'
                                                )
                                            ),
                                        name="Cluster center")

                    data7 = go.Data([trace0, trace1])
                    fig = go.Figure(data=data7)
                    layout = go.Layout(
                                height=600, width=700, title=f"K-Means cluster size {c}",
                                xaxis=dict(
                                    title=features[0],
                                ),
                                yaxis=dict(
                                    title=features[1]
                                ) ) 

                    fig.update_layout(layout)
                    st.plotly_chart(fig)

                    # Join with the same dataframe to select variables 
                    joindf= pd.concat([tdf[features], tdf[["zone_id", "cluster", "Quartier"]]],axis = 1, join = 'outer', ignore_index=False, sort=False)
                    # Download xlsx
                    df_xlsx = to_excel(joindf)
                    ste.download_button(label=f'📥 (No PCA) Download xlsx k = {c}',
                                                    data=df_xlsx ,
                                                    file_name= f'file_k{c}.xlsx')
                    st.markdown("***")


            if use_pca=="Yes":
                st.markdown("### Using PCA") 
                tdf= udf.copy()
                X = tdf[features]  
                pca = PCA(n_components=int(comp))
                principalComponents = pca.fit_transform(X)  
                feat = list(range(pca.n_components_))
                PCA_components = pd.DataFrame(principalComponents, columns=list(range(len(feat))))
                choosed_component = sidebar.multiselect("Axis",feat,default=[0,1])
                choosed_component=[int(i) for i in choosed_component]
                loadings = pca.components_
                xs = loadings[0]
                ys = loadings[1]
                # Visualizations
                tab4, tab5, tab6 = st.tabs(["Explained Variance PC", "KElbow Visualizer", 'Silhouette score'])
                with tab4:
                    exp_var = explained_var(X)
                with tab5:
                    elbow = visualice_elbow(PCA_components)
                with tab6:
                    silhouette = silhouette_viz(principalComponents)

                if len(choosed_component)>1:
                    for c in range(1,ks+1):
                        X = PCA_components[choosed_component]
                        model = KMeans(n_clusters=c)
                        model.fit(X)
                        y_kmeans = model.predict(X)
                        X["cluster"] = y_kmeans
                        fig = go.Figure()
                        fig = px.scatter(x=X[choosed_component[0]],y=X[choosed_component[1]],  
                            size=tdf.avg_visits_perSensor,
                            color= X.cluster,
                            text=tdf["shop"], 
                            labels={'color': 'Cluster', 'text': 'Shop'},
                            opacity = 1,
                            ) 
                        
                        fig.update_layout(legend_title='Zones, centers')
                        fig.update_traces(marker_color= X.cluster.apply(lambda x: colors[x]), 
                                        marker_size=tdf.avg_visits_perSensor,
                                        marker_sizemode='area',
                                        marker_sizeref=3.*max(tdf.avg_visits_perSensor)/30.**2,
                                        showlegend=False)  
                        
                        fig.update_traces(textposition='top center', textfont_size=12, showlegend=True, name='Zone')
                        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide') 

                        for i in X.cluster.unique():
                            # get the convex hull
                            try:
                                points = X[X.cluster == i][[choosed_component[0], choosed_component[1]]].values
                                hull = ConvexHull(points)
                                x_hull = np.append(points[hull.vertices,0],
                                                points[hull.vertices,0][0])
                                y_hull = np.append(points[hull.vertices,1],
                                                points[hull.vertices,1][0])
                                # interpolate
                                dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
                                dist_along = np.concatenate(([0], dist.cumsum()))
                                spline, u = interpolate.splprep([x_hull, y_hull], 
                                                                u=dist_along, s=0, per=1)
                                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                                interp_x, interp_y = interpolate.splev(interp_d, spline)
                                # plot shape
                                fig.add_trace(go.Scatter(x=interp_x, y=interp_y, fill="toself", opacity = .2, showlegend=False))
                            except QhullError:
                                print('No area')

                        for i, varnames in enumerate(tdf[features]):
                            fig.add_annotation(
                            x=xs[i],  
                            y=ys[i],  
                            ax=0,  
                            ay=0,  
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            text='', 
                            showarrow=True,
                            arrowhead=1,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='grey'

                        )  
                            fig.add_annotation(
                            x=xs[i],  
                            y=ys[i], 
                            text = varnames,
                            showarrow=False,
                            font_size=12, font_color='blue', 
                            xanchor='left',
                            yanchor='bottom'
                        )

                        fig.add_traces(px.scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1]).data[0] )
                        fig.update_traces(marker_size=11, marker_color=colors, marker_symbol = 'diamond', 
                                        marker_line=dict(width=1, color='rgb(255,127,0)'), selector=dict(mode='markers'), showlegend=True, name='Cluster center')
                    

                
                        layout = go.Layout(height=650, width=700, 
                                    title=f"K-Means Cluster size {c} \n Total explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%",
                                    xaxis=dict(
                                        title=f"PC {choosed_component[0]} Explained variance {pca.explained_variance_ratio_[0]*100:.2f}%",
                                    ),
                                    yaxis=dict(
                                        title=f"PC {choosed_component[1]} Explained variance  {pca.explained_variance_ratio_[1]*100:.2f}%"
                                    ) ) 
                        fig.update_layout(layout)
                        st.plotly_chart(fig)

                        joindf= pd.concat([X, tdf],axis = 1, join = 'outer', ignore_index=False, sort=False)
                    
                        # Download xlsx
                        df_xlsx = to_excel(joindf)
                        ste.download_button(label=f'📥 (PCA) Download xlsx k = {c}',
                                                        data=df_xlsx ,
                                                        file_name= f'file_k{c}.xlsx')
                        st.markdown("***")
            
        else:
            st.markdown("##### No variables selected")
