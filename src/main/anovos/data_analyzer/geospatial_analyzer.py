from anovos.data_ingest.data_ingest import read_dataset
from anovos.shared.spark import spark
from anovos.shared.utils import ends_with
from anovos.data_ingest.geo_auto_detection import ll_gh_cols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from branca.element import Figure
from folium.plugins import FastMarkerCluster, HeatMapWithTime
from pyspark.sql import functions as F
from pyspark.sql import types as T
import geohash2 as gh
from folium import plugins
from sklearn.cluster import DBSCAN
import json
import os
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import dateutil.parser
import plotly.tools as tls
import warnings

warnings.filterwarnings("ignore")

global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
global_plot_bg_color = "rgba(0,0,0,0)"
global_paper_bg_color = "rgba(0,0,0,0)"
default_template = (dp.HTML("""<html><img src="https://mobilewalla-anovos.s3.amazonaws.com/anovos.png"style="height:100px;display:flex;margin:auto;float:right"/></html>"""),\
                    dp.Text("# ML-Anovos Report"))

def descriptive_stats_gen(df,lat_col=None,long_col=None,geohash_col=None,id_col="",max_val=100,master_path="."):
        
    if (lat_col is not None) & (long_col is not None):
        
        dist_lat_long,dist_lat,dist_long = df.select(lat_col,long_col).distinct().count(),\
                                           df.select(lat_col).distinct().count(),\
                                           df.select(long_col).distinct().count()
        
        
        
        top_lat_long = df.withColumn("lat_long_pair",F.concat(F.lit("["),F.col(lat_col),F.lit(","),F.col(long_col),F.lit("]")))\
                         .groupBy("lat_long_pair")\
                         .agg(F.countDistinct(id_col).alias("count_id"))\
                         .orderBy("count_id",ascending=False).limit(max_val)
        
        top_lat =      df.groupBy(lat_col)\
                         .agg(F.countDistinct(id_col).alias("count_id"))\
                         .orderBy("count_id",ascending=False).limit(max_val).toPandas()
        

        top_long =     df.groupBy(long_col)\
                         .agg(F.countDistinct(id_col).alias("count_id"))\
                         .orderBy("count_id",ascending=False).limit(max_val).toPandas()       
    
        most_lat_long = top_lat_long.rdd.flatMap(lambda x: x).collect()[0]
        most_lat_long_cnt = top_lat_long.rdd.flatMap(lambda x: x).collect()[1]
        
        top_lat_long = top_lat_long.toPandas()
        
        d1 = dist_lat_long,dist_lat,dist_long,most_lat_long,most_lat_long_cnt
        d1_desc = ("Distinct {Lat, Long} Pair","Distinct Latitude","Distinct Longitude","Most Common {Lat, Long} Pair","Most Common {Lat, Long} Pair Occurence")
        
        gen_stats = pd.DataFrame(d1,d1_desc).reset_index().rename(columns={0 : "", 'index' : ''})
        l = ["Overall_Summary","Top_" + str(max_val) + "_Lat", "Top_" + str(max_val) + "_Long", "Top_" + str(max_val) + "_Lat_Long"]
        
        for idx,i in enumerate([gen_stats,top_lat,top_long,top_lat_long]):
            
            i.to_csv(ends_with(master_path) + l[idx] + "_1_" + lat_col + "_" + long_col + ".csv",index=False)
            
    
    if geohash_col is not None:
        
        dist_geohash = df.select(geohash_col).distinct().count()
        precision_geohash = df.select(F.max(F.length(F.col(geohash_col)))).rdd.flatMap(lambda x: x).collect()[0]
        max_occuring_geohash = df.groupBy(geohash_col).agg(F.countDistinct(id_col).alias("count_id"))\
                                 .orderBy("count_id",ascending=False).limit(1)
        
        geohash_val = max_occuring_geohash.rdd.flatMap(lambda x: x).collect()[0]
        geohash_cnt = max_occuring_geohash.rdd.flatMap(lambda x: x).collect()[1]
        
        l = ["Overall_Summary","Top_" + str(max_val) + "_Geohash_Distribution"]
        
        pd.DataFrame([["Total number of Distinct Geohashes", str(dist_geohash)],\
                      ["The Precision level observed for the Geohashes",str(precision_geohash)],\
                      ["The Most Common Geohash", str(geohash_val) + " , " + str(geohash_cnt)]],columns=["",""])\
          .to_csv(ends_with(master_path) + l[0] + "_2_" + geohash_col + ".csv",index=False)
        
        df.withColumn("geohash_" + str(precision_geohash),F.substring(F.col(geohash_col),1,precision_geohash)).groupBy("geohash_" + str(precision_geohash)).agg(F.countDistinct(id_col).alias("count_id")).orderBy("count_id",ascending=False)\
          .limit(max_val)\
          .toPandas()\
          .to_csv(ends_with(master_path) + l[1] + "_2_" + geohash_col + ".csv",index=False)
        
        

def lat_long_col_stats_gen(idf,lat_col,long_col,id_col,max_val=100):
        
    lat_col_ = lat_col.split("|")
    long_col_ = long_col.split("|")
    ll = []
    
    if len(lat_col_) == 1 & len(long_col_) == 1:
        descriptive_stats_gen(idf,lat_col=lat_col_[0],long_col=long_col_[0],id_col=id_col,max_val=max_val)
        
    else:
        for i in range(0,len(lat_col_)):
            descriptive_stats_gen(idf,lat_col=lat_col_[i],long_col=long_col_[i],id_col=id_col,max_val=max_val)
        
        
def geohash_col_stats_gen(idf,geohash_col,id_col,max_val=100):
        
    geohash_col_ = geohash_col.split("|")
    ll = []
    
    if len(geohash_col) == 1:
        descriptive_stats_gen(idf,geohash_col=geohash_col_[0],id_col=id_col,max_val=max_val)
    else:
        for i in range(0,len(geohash_col_)):
            descriptive_stats_gen(idf,geohash_col=geohash_col_[i],id_col=id_col,max_val=max_val)
            

def stats_gen_lat_long_geo(df,lat_col,long_col,geohash_col,id_col,max_val=100):
        
    if lat_col:
        len_lat = len(lat_col.split("|"))
        ll_stats = lat_long_col_stats_gen(df,lat_col,long_col,id_col,max_val)
        
    else:
        len_lat = 0
        
    if geohash_col:
        len_geohash_col = len(geohash_col.split("|"))
        geohash_stats = geohash_col_stats_gen(df,geohash_col,id_col,max_val)
        
    else:
        len_geohash_col = 0
    
    if (len_lat + len_geohash_col) == 1:
        
        if len_lat == 0:
            return geohash_stats
        else:
            return ll_stats
        
    elif (len_lat + len_geohash_col) > 1:
        
        if (len_lat > 1) and (len_geohash_col == 0):
            
            return ll_stats
        
        elif (len_lat == 0) and (len_geohash_col > 1):
            
            return geohash_stats

        elif (len_lat >= 1) and (len_geohash_col >= 1):
            
            return ll_stats, geohash_stats


def geo_cluster_analysis(df,lat_col,long_col,max_cluster=20,eps=0.6,min_samples=25,master_path=".",col_name):
    
    df_ = df[[lat_col,long_col]]
    max_k = int(max_cluster)
    ## iterations
    distortions = [] 
    for i in range(2, max_k+1):
        if len(df_) >= i:
           model = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
           model.fit(df_)
           distortions.append(model.inertia_)
    ## best k: the lowest derivative
    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i in np.diff(distortions,2)]))
    ## plot
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=list(range(1, len(distortions)+1)), y=distortions,mode='lines+markers',name='lines+markers',line = dict(color=global_theme[2], width=2, dash='dash'),marker=dict(size=10)))
    f1.update_xaxes(showgrid=True, gridwidth=1, gridcolor=px.colors.sequential.gray[10])
    f1.update_yaxes(showgrid=True, gridwidth=1, gridcolor=px.colors.sequential.gray[10])
    f1.add_vline(x=k, line_width=3, line_dash="dash", line_color=global_theme[4])
    f1.update_layout(title_text=str("Elbow Curve Showing the Optimal Number of Clusters [K : " + str(k) + "]"))
    f1.layout.plot_bgcolor = global_plot_bg_color
    f1.layout.paper_bgcolor = global_paper_bg_color
    
    f1.write_json(ends_with(master_path) + "cluster_plot_1_" + col_name)
    
    model = MiniBatchKMeans(n_clusters=k,init='k-means++', max_iter=300, n_init=10, random_state=0)
    df_["cluster"] = model.fit_predict(df_)
    df_.to_csv(ends_with(master_path) + "cluster_output_" + col_name + ".csv", index=False)
    
    zoom, center = zoom_center(lons=df_[long_col].values.tolist(),lats=df_[lat_col].values.tolist())

    f2 = px.scatter_mapbox(df_, lat=lat_col, lon=long_col, color="cluster",color_continuous_scale=global_theme,mapbox_style="carto-positron",zoom=zoom,center=center, size="cluster")
    f2.update_geos(fitbounds="locations")
    f2.update_layout(mapbox_style="carto-positron")
    f2.update_layout(coloraxis_showscale=True)
    f2.write_json(ends_with(master_path) + "cluster_plot_2_" + col_name)
    
    # Use `hole` to create a donut-like pie chart
    cluster_dtls = df_.groupby(['cluster']).size().reset_index(name='counts')
    f3 = go.Figure()
    f3.add_trace(go.Pie(labels=list(set(cluster_dtls.cluster.values)), values=list(set(cluster_dtls.counts.values)), hole=.3,marker_colors=global_theme,text=list(set(cluster_dtls.cluster.values))))
    f3.write_json(ends_with(master_path) + "cluster_plot_3_" + col_name)
    
    # Reading in 2D Feature Space
    df_ = df[[lat_col,long_col]]

    # DBSCAN model with parameters
    model = DBSCAN(eps=0.6, min_samples=25).fit(df_)
    df_["outlier"] = model.labels_
    df_ = df_[df_.outlier.values==-1]
    df_['outlier'] = 1
    
    f4 = go.Figure(go.Scatter(mode="markers", x=df_[long_col], y=df_[lat_col], marker_symbol='x-thin',marker_line_color="black", marker_color="black",marker_line_width=2, marker_size=15))
    f4.layout.plot_bgcolor = global_plot_bg_color
    f4.layout.paper_bgcolor = global_paper_bg_color
    f4.update_xaxes(title_text='longitude')
    f4.update_yaxes(title_text='latitude')
    f4.write_json(ends_with(master_path) + "cluster_plot_4_" + col_name)
    
    
    

def geo_cluster_generator(df, lat_col_list, long_col_list, geo_col_list,max_cluster=20,eps=0.6,min_samples=25,master_path="."):
    
    if isinstance(df, pd.DataFrame):
        pass
    else:
        df = df.toPandas()

    try:
        lat_col = lat_col_list.split("|")
        long_col = long_col_list.split("|")
    except:
        lat_col = []
    try:
        geohash_col = geo_col_list.split("|")
    except:
        geohash_col = []
        
    
    if len(lat_col) >= 1:
        for idx, i in enumerate(lat_col):
            col_name = lat_col[idx] + "_" + long_col[idx]
            geo_cluster_analysis(df,lat_col[idx],long_col[idx],max_cluster,eps,min_samples,master_path,col_name)
    
    if len(geohash_col) >=1:
        for idx, i in enumerate(geohash_col):
            col_name = geohash_col[idx]
            df_ = df
            df_["latitude"]  = df_.apply(lambda x: geo_to_latlong(x[col_name], 0), axis=1)
            df_["longitude"] = df_.apply(lambda x: geo_to_latlong(x[col_name], 1), axis=1)

            geo_cluster_analysis(df_,"latitude","longitude",max_cluster,eps,min_samples,master_path,col_name)


def geospatial_autodetection(df,frac_sample=0.1,max_records=10000, top_geo_records = 100, master_path=".",id_col,max_cluster=20,eps=0.6,min_samples=25):
    
    df_sample = data_sampling.data_sample(df,strata_cols="all",fraction=frac_sample)
    df_sample.toPandas().to_csv(ends_with(master_path) + "geospatial_viz_data.csv",index=False)
    df_sample_count = df_sample.count()
    lat_cols,long_cols,gh_cols = ll_gh_cols(df,max_records)
    stats_gen_lat_long_geo(df,lat_cols,long_cols,gh_cols,id_col,top_geo_records)
    geo_cluster_generator(df, lat_col_list, long_col_list, geo_col_list,max_cluster,eps,min_samples,master_path)
    
    return df_sample_count, lat_cols, long_cols, gh_cols, df_sample


