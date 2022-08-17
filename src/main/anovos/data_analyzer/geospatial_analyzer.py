# coding=utf-8

"""This module help to analyze & summarize the geospatial fields identified through the auto-detection module. Additionally, it generates the intermediate output which are fed in to the reporting section

As a part of generation of final output, there are various functions created such as -

- descriptive_stats_gen
- lat_long_col_stats_gen
- geohash_col_stats_gen
- stats_gen_lat_long_geo
- geo_cluster_analysis
- geo_cluster_generator
- generate_loc_charts_processor
- generate_loc_charts_controller
- geospatial_autodetection

Respective functions have sections containing the detailed definition of the parameters used for computing.

"""

from anovos.data_ingest.data_ingest import read_dataset
from anovos.shared.spark import spark
from anovos.shared.utils import ends_with, output_to_local, path_ak8s_modify
from anovos.data_ingest import data_sampling
from anovos.data_ingest.geo_auto_detection import ll_gh_cols, geo_to_latlong
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from itertools import product
from pathlib import Path

# from branca.element import Figure
# from folium.plugins import FastMarkerCluster, HeatMapWithTime
# from folium import plugins
from pyspark.sql import functions as F
from pyspark.sql import types as T
import geohash2 as gh
from sklearn.cluster import DBSCAN
import json
import os
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
global_plot_bg_color = "rgba(0,0,0,0)"
global_paper_bg_color = "rgba(0,0,0,0)"


blank_chart = go.Figure()
# blank_chart.update_layout(autosize=False, width=10, height=10)
blank_chart.layout.plot_bgcolor = global_plot_bg_color
blank_chart.layout.paper_bgcolor = global_paper_bg_color
blank_chart.update_xaxes(visible=False)
blank_chart.update_yaxes(visible=False)

mapbox_list = [
    "open-street-map",
    "white-bg",
    "carto-positron",
    "carto-darkmatter",
    "stamen- terrain",
    "stamen-toner",
    "stamen-watercolor",
]


# def zoom_center(
#     lons: tuple = None,
#     lats: tuple = None,
#     lonlats: tuple = None,
#     format: str = "lonlat",
#     projection: str = "mercator",
#     width_to_height: float = 2.0,
# ) -> (float, dict):
#     """Finds optimal zoom and centering for a plotly mapbox.
#     Must be passed (lons & lats) or lonlats.
#     Temporary solution awaiting official implementation, see:
#     https://github.com/plotly/plotly.js/issues/3434

#     Parameters
#     --------
#     lons: tuple, optional, longitude component of each location
#     lats: tuple, optional, latitude component of each location
#     lonlats: tuple, optional, gps locations
#     format: str, specifying the order of longitud and latitude dimensions,
#         expected values: 'lonlat' or 'latlon', only used if passed lonlats
#     projection: str, only accepting 'mercator' at the moment,
#         raises `NotImplementedError` if other is passed
#     width_to_height: float, expected ratio of final graph's with to height,
#         used to select the constrained axis.

#     Returns
#     --------
#     zoom: float, from 1 to 20
#     center: dict, gps position with 'lon' and 'lat' keys

#     >>> print(zoom_center((-109.031387, -103.385460),
#     ...     (25.587101, 31.784620)))
#     (5.75, {'lon': -106.208423, 'lat': 28.685861})
#     """
#     if lons is None and lats is None:
#         if isinstance(lonlats, tuple):
#             lons, lats = zip(*lonlats)
#         else:
#             raise ValueError("Must pass lons & lats or lonlats")

#     maxlon, minlon = max(lons), min(lons)
#     maxlat, minlat = max(lats), min(lats)
#     center = {
#         "lon": round((maxlon + minlon) / 2, 6),
#         "lat": round((maxlat + minlat) / 2, 6),
#     }

#     # longitudinal range by zoom level (20 to 1)
#     # in degrees, if centered at equator
#     lon_zoom_range = np.array(
#         [
#             0.0007,
#             0.0014,
#             0.003,
#             0.006,
#             0.012,
#             0.024,
#             0.048,
#             0.096,
#             0.192,
#             0.3712,
#             0.768,
#             1.536,
#             3.072,
#             6.144,
#             11.8784,
#             23.7568,
#             47.5136,
#             98.304,
#             190.0544,
#             360.0,
#         ]
#     )

#     if projection == "mercator":
#         margin = 1.2
#         height = (maxlat - minlat) * margin * width_to_height
#         width = (maxlon - minlon) * margin
#         lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
#         lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
#         zoom = round(min(lon_zoom, lat_zoom), 2)
#     else:
#         raise NotImplementedError(f"{projection} projection is not implemented")

#     return zoom, center


def descriptive_stats_gen(
    df, lat_col, long_col, geohash_col, id_col, master_path, max_val
):

    """

    This function helps to produce descriptive stats for the analyzed geospatial fields


    Parameters
    ----------

    df
        Analysis DataFrame

    lat_col
        Latitude column
    long_col
        Longitude column
    geohash_col
        Geohash column
    id_col
        ID column
    master_path
        Path containing all the output from analyzed data
    max_val
        Top geospatial records displayed

    Returns
    -------
    DataFrame[CSV]
    """

    if (lat_col is not None) & (long_col is not None):

        dist_lat_long, dist_lat, dist_long = (
            df.select(lat_col, long_col).distinct().count(),
            df.select(lat_col).distinct().count(),
            df.select(long_col).distinct().count(),
        )

        top_lat_long = (
            df.withColumn(
                "lat_long_pair",
                F.concat(
                    F.lit("["), F.col(lat_col), F.lit(","), F.col(long_col), F.lit("]")
                ),
            )
            .groupBy("lat_long_pair")
            .agg(
                F.countDistinct(id_col).alias("count_id"),
                F.count(id_col).alias("count_records"),
            )
            .orderBy("count_id", ascending=False)
            .limit(max_val)
        )

        top_lat = (
            df.groupBy(lat_col)
            .agg(
                F.countDistinct(id_col).alias("count_id"),
                F.count(id_col).alias("count_records"),
            )
            .orderBy("count_id", ascending=False)
            .limit(max_val)
            .toPandas()
        )

        top_long = (
            df.groupBy(long_col)
            .agg(
                F.countDistinct(id_col).alias("count_id"),
                F.count(id_col).alias("count_records"),
            )
            .orderBy("count_id", ascending=False)
            .limit(max_val)
            .toPandas()
        )

        most_lat_long = top_lat_long.rdd.flatMap(lambda x: x).collect()[0]
        most_lat_long_cnt = top_lat_long.rdd.flatMap(lambda x: x).collect()[1]

        top_lat_long = top_lat_long.toPandas()

        d1 = dist_lat_long, dist_lat, dist_long, most_lat_long, most_lat_long_cnt
        d1_desc = (
            "Distinct {Lat, Long} Pair",
            "Distinct Latitude",
            "Distinct Longitude",
            "Most Common {Lat, Long} Pair",
            "Most Common {Lat, Long} Pair Occurence",
        )

        gen_stats = (
            pd.DataFrame(d1, d1_desc)
            .reset_index()
            .rename(columns={"index": "Stats", 0: "Count"})
        )
        # l = [
        #     "Overall_Summary",
        #     "Top_" + str(max_val) + "_Lat",
        #     "Top_" + str(max_val) + "_Long",
        #     "Top_" + str(max_val) + "_Lat_Long",
        # ]

        l = ["Overall_Summary", "Top_" + str(max_val) + "_Lat_Long"]

        # for idx, i in enumerate([gen_stats, top_lat, top_long, top_lat_long]):
        for idx, i in enumerate([gen_stats, top_lat_long]):

            i.to_csv(
                ends_with(master_path)
                + l[idx]
                + "_1_"
                + lat_col
                + "_"
                + long_col
                + ".csv",
                index=False,
            )

    if geohash_col is not None:

        dist_geohash = df.select(geohash_col).distinct().count()
        precision_geohash = (
            df.select(F.max(F.length(F.col(geohash_col))))
            .rdd.flatMap(lambda x: x)
            .collect()[0]
        )
        max_occuring_geohash = (
            df.groupBy(geohash_col)
            .agg(F.count(id_col).alias("count_records"))
            .orderBy("count_records", ascending=False)
            .limit(1)
        )

        geohash_val = max_occuring_geohash.rdd.flatMap(lambda x: x).collect()[0]
        geohash_cnt = max_occuring_geohash.rdd.flatMap(lambda x: x).collect()[1]

        l = ["Overall_Summary", "Top_" + str(max_val) + "_Geohash_Distribution"]
        geohash_area_width_height_1_12 = [
            "5,009.4km x 4,992.6km",
            "1,252.3km x 624.1km",
            "156.5km x 156km",
            "39.1km x 19.5km",
            "4.9km x 4.9km",
            "1.2km x 609.4m",
            "152.9m x 152.4m",
            "38.2m x 19m",
            "4.8m x 4.8m",
            "1.2m x 59.5cm",
            "14.9cm x 14.9cm",
            "3.7cm x 1.9cm",
        ]

        pd.DataFrame(
            [
                ["Total number of Distinct Geohashes", str(dist_geohash)],
                [
                    "The Precision level observed for the Geohashes",
                    str(precision_geohash)
                    + " [Reference Area Width x Height : "
                    + str(geohash_area_width_height_1_12[precision_geohash - 1])
                    + "] ",
                ],
                [
                    "The Most Common Geohash",
                    str(geohash_val) + " , " + str(geohash_cnt),
                ],
            ],
            columns=["Stats", "Count"],
        ).to_csv(
            ends_with(master_path) + l[0] + "_2_" + geohash_col + ".csv", index=False
        )

        df.withColumn(
            "geohash_" + str(precision_geohash),
            F.substring(F.col(geohash_col), 1, precision_geohash),
        ).groupBy("geohash_" + str(precision_geohash)).agg(
            F.countDistinct(id_col).alias("count_id"),
            F.count(id_col).alias("count_records"),
        ).orderBy(
            "count_id", ascending=False
        ).limit(
            max_val
        ).toPandas().to_csv(
            ends_with(master_path) + l[1] + "_2_" + geohash_col + ".csv", index=False
        )


def lat_long_col_stats_gen(df, lat_col, long_col, id_col, master_path, max_val):

    """

    This function helps to produce descriptive stats for the latitude longitude columns


    Parameters
    ----------

    df
        Analysis DataFrame

    lat_col
        Latitude column
    long_col
        Longitude column
    id_col
        ID column
    master_path
        Path containing all the output from analyzed data
    max_val
        Top geospatial records displayed

    Returns
    -------

    """

    ll = []

    if len(lat_col) == 1 & len(long_col) == 1:
        descriptive_stats_gen(
            df, lat_col[0], long_col[0], None, id_col, master_path, max_val
        )

    else:
        for i in range(0, len(lat_col)):
            descriptive_stats_gen(
                df, lat_col[i], long_col[i], None, id_col, master_path, max_val
            )


def geohash_col_stats_gen(df, geohash_col, id_col, master_path, max_val):

    """

    This function helps to produce descriptive stats for the geohash columns


    Parameters
    ----------

    df
        Analysis DataFrame

    geohash_col
        Geohash column
    id_col
        ID column
    master_path
        Path containing all the output from analyzed data
    max_val
        Top geospatial records displayed

    Returns
    -------

    """

    ll = []

    if len(geohash_col) == 1:
        descriptive_stats_gen(
            df, None, None, geohash_col[0], id_col, master_path, max_val
        )
    else:
        for i in range(0, len(geohash_col)):
            descriptive_stats_gen(
                df, None, None, geohash_col[i], id_col, master_path, max_val
            )


def stats_gen_lat_long_geo(
    df, lat_col, long_col, geohash_col, id_col, master_path, max_val
):

    """

    This function helps to produce descriptive stats for the analyzed geospatial fields


    Parameters
    ----------

    df
        Analysis DataFrame

    lat_col
        Latitude column
    long_col
        Longitude column
    geohash_col
        Geohash column
    id_col
        ID column
    master_path
        Path containing all the output from analyzed data
    max_val
        Top geospatial records displayed

    Returns
    -------

    """

    if lat_col:
        len_lat = len(lat_col)
        ll_stats = lat_long_col_stats_gen(
            df, lat_col, long_col, id_col, master_path, max_val
        )

    else:
        len_lat = 0

    if geohash_col:
        len_geohash_col = len(geohash_col)
        geohash_stats = geohash_col_stats_gen(
            df, geohash_col, id_col, master_path, max_val
        )

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


def geo_cluster_analysis(
    df,
    lat_col,
    long_col,
    max_cluster,
    eps,
    min_samples,
    master_path,
    col_name,
    global_map_box_val,
):

    """

    This function helps to generate cluster analysis stats for the identified geospatial fields

    Parameters
    ----------

    df
        Analysis DataFrame

    lat_col
        Latitude column
    long_col
        Longitude column
    max_cluster
        Maximum number of iterations to decide on the optimum cluster
    eps
        Epsilon value range (Min EPS, Max EPS, Interval) used for DBSCAN clustering
    min_samples
        Minimum Sample Size range (Min Sample Size, Max Sample Size, Interval) used for DBSCAN clustering
    master_path
        Path containing all the output from analyzed data
    col_name
        Analysis column
    global_map_box_val
        Geospatial Chart Theme Index
    Returns
    -------

    """

    df_ = df[[lat_col, long_col]]
    max_k = int(max_cluster)
    ## iterations
    distortions = []
    for i in range(2, max_k + 1):
        if len(df_) >= i:
            model = MiniBatchKMeans(
                n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
            )
            model.fit(df_)
            distortions.append(model.inertia_)
    ## best k: the lowest derivative
    k = [i * 100 for i in np.diff(distortions, 2)].index(
        min([i * 100 for i in np.diff(distortions, 2)])
    )
    ## plot
    f1 = go.Figure()
    f1.add_trace(
        go.Scatter(
            x=list(range(1, len(distortions) + 1)),
            y=distortions,
            mode="lines+markers",
            name="lines+markers",
            line=dict(color=global_theme[2], width=2, dash="dash"),
            marker=dict(size=10),
        )
    )
    f1.update_yaxes(
        title="Distortion",
        showgrid=True,
        gridwidth=1,
        gridcolor=px.colors.sequential.gray[10],
    )
    f1.update_xaxes(title="Values of K")
    f1.add_vline(x=k, line_width=3, line_dash="dash", line_color=global_theme[4])
    f1.update_layout(
        title_text="Elbow Curve Showing the Optimal Number of Clusters [K : "
        + str(k)
        + "] <br><sup>Algorithm Used : KMeans</sup>"
    )
    f1.layout.plot_bgcolor = global_plot_bg_color
    f1.layout.paper_bgcolor = global_paper_bg_color

    f1.write_json(ends_with(master_path) + "cluster_plot_1_elbow_" + col_name)

    model = MiniBatchKMeans(
        n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    df_["cluster"] = model.fit_predict(df_)
    df_.to_csv(
        ends_with(master_path) + "cluster_output_kmeans_" + col_name + ".csv",
        index=False,
    )

    # Use `hole` to create a donut-like pie chart
    cluster_dtls = df_.groupby(["cluster"]).size().reset_index(name="counts")
    # zoom, center = zoom_center(lons=df_[long_col].values.tolist(),lats=df_[lat_col].values.tolist())

    f2 = go.Figure(
        go.Pie(
            labels=list(cluster_dtls.cluster.values),
            values=list(cluster_dtls.counts.values),
            hole=0.3,
            marker_colors=global_theme,
            text=list(cluster_dtls.cluster.values),
        )
    )
    f2.update_layout(
        title_text="Distribution of Clusters"
        + "<br><sup>Algorithm Used : K-Means (Distance : Euclidean) </sup>",
        legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center"),
    )
    f2.write_json(ends_with(master_path) + "cluster_plot_2_kmeans_" + col_name)

    f3 = px.scatter_mapbox(
        df_,
        lat=lat_col,
        lon=long_col,
        color="cluster",
        color_continuous_scale=global_theme,
        mapbox_style=mapbox_list[global_map_box_val],
    )

    f3.update_geos(fitbounds="locations")
    f3.update_layout(mapbox_style=mapbox_list[global_map_box_val])
    f3.update_layout(
        title_text="Cluster Wise Geospatial Datapoints "
        + "<br><sup>Algorithm Used : K-Means</sup>"
    )
    f3.update_layout(coloraxis_showscale=False, autosize=False, width=1200, height=900)
    f3.write_json(ends_with(master_path) + "cluster_plot_3_kmeans_" + col_name)

    # Reading in 2D Feature Space
    df_ = df[[lat_col, long_col]]

    # DBSCAN model with parameters
    eps = eps.split(",")
    min_samples = min_samples.split(",")

    for i in range(3):
        eps[i] = float(eps[i])
        min_samples[i] = float(min_samples[i])

    DBSCAN_params = list(
        product(
            np.arange(eps[0], eps[1], eps[2]),
            np.arange(min_samples[0], min_samples[1], min_samples[2]),
        )
    )

    no_of_clusters = []
    sil_score = []
    for p in DBSCAN_params:
        try:
            DBS_clustering = DBSCAN(eps=p[0], min_samples=p[1], metric="haversine").fit(
                df_
            )
            sil_score.append(silhouette_score(df_, DBS_clustering.labels_))
        except:
            sil_score.append(0)

    tmp = pd.DataFrame.from_records(DBSCAN_params, columns=["Eps", "Min_samples"])
    tmp["Sil_score"] = sil_score

    eps_, min_samples_ = list(tmp.sort_values("Sil_score", ascending=False).values[0])[
        0:2
    ]
    DBS_clustering = DBSCAN(eps=eps_, min_samples=min_samples_, metric="haversine").fit(
        df_
    )
    DBSCAN_clustered = df_.copy()
    DBSCAN_clustered.loc[:, "Cluster"] = DBS_clustering.labels_
    DBSCAN_clustered.to_csv(
        ends_with(master_path) + "cluster_output_dbscan_" + col_name + ".csv",
        index=False,
    )

    pivot_1 = pd.pivot_table(
        tmp, values="Sil_score", index="Min_samples", columns="Eps"
    )

    # def df_to_plotly(df):
    #     return {'z': df.values.tolist(),
    #             'x': df.columns.tolist(),
    #             'y': df.index.tolist()}

    # f1_ = go.Figure(data=go.Heatmap(df_to_plotly(pivot_1),colorscale=global_theme_r,texttemplate="%{text}", textfont={"size":12}))
    # f1_ = go.Figure(data=go.Heatmap(z=df_to_plotly(pivot_1)['z'],x = df_to_plotly(pivot_1)['x'], y = df_to_plotly(pivot_1)['y'] ,colorscale=global_theme_r))
    f1_ = px.imshow(
        pivot_1.values,
        text_auto=".3f",
        color_continuous_scale=global_theme,
        aspect="auto",
        y=list(pivot_1.index),
        x=list(pivot_1.columns),
    )
    f1_.update_xaxes(title="Eps")
    f1_.update_yaxes(title="Min_samples")
    f1_.update_traces(
        text=np.around(pivot_1.values, decimals=3), texttemplate="%{text}"
    )
    f1_.update_layout(
        title_text="Distribution of Silhouette Scores Across Different Parameters "
        + "<br><sup>Algorithm Used : DBSCAN</sup>"
    )
    f1_.layout.plot_bgcolor = global_plot_bg_color
    f1_.layout.paper_bgcolor = global_paper_bg_color
    f1_.write_json(ends_with(master_path) + "cluster_plot_1_silhoutte_" + col_name)

    DBSCAN_clustered.loc[DBSCAN_clustered["Cluster"] == -1, "Cluster"] = 999
    cluster_dtls_ = (
        DBSCAN_clustered.groupby(["Cluster"]).size().reset_index(name="counts")
    )

    f2_ = go.Figure(
        go.Pie(
            labels=list(cluster_dtls_.Cluster.values),
            values=list(cluster_dtls_.counts.values),
            hole=0.3,
            marker_colors=global_theme,
            text=list(cluster_dtls_.Cluster.values),
        )
    )

    f2_.update_layout(
        title_text="Distribution of Clusters"
        + "<br><sup>Algorithm Used : DBSCAN (Distance : Haversine) </sup>",
        legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center"),
    )
    f2_.write_json(ends_with(master_path) + "cluster_plot_2_dbscan_" + col_name)

    f3_ = px.scatter_mapbox(
        DBSCAN_clustered,
        lat=lat_col,
        lon=long_col,
        color="Cluster",
        color_continuous_scale=global_theme,
        mapbox_style=mapbox_list[global_map_box_val],
    )

    f3_.update_geos(fitbounds="locations")
    f3_.update_layout(mapbox_style=mapbox_list[global_map_box_val])
    f3_.update_layout(
        title_text="Cluster Wise Geospatial Datapoints "
        + "<br><sup>Algorithm Used : DBSCAN</sup>"
    )
    # f3_.update_layout(titlecoloraxis_showscale=False, autosize=False, width=1200, height=900)
    f3_.update_layout(autosize=False, width=1200, height=900)
    f3_.update_coloraxes(showscale=False)
    f3_.write_json(ends_with(master_path) + "cluster_plot_3_dbscan_" + col_name)

    try:
        DBSCAN_clustered_ = df_.copy()
        df_outlier = DBSCAN(eps=eps_, min_samples=min_samples_).fit(DBSCAN_clustered_)
        DBSCAN_clustered_.loc[:, "Cluster"] = df_outlier.labels_
        DBSCAN_clustered_ = DBSCAN_clustered_[DBSCAN_clustered_.Cluster.values == -1]
        DBSCAN_clustered_["outlier"] = 1

        f4 = go.Figure(
            go.Scatter(
                mode="markers",
                x=DBSCAN_clustered_[long_col],
                y=DBSCAN_clustered_[lat_col],
                marker_symbol="x-thin",
                marker_line_color="black",
                marker_color="black",
                marker_line_width=2,
                marker_size=20,
            )
        )
        f4.layout.plot_bgcolor = global_plot_bg_color
        f4.layout.paper_bgcolor = global_paper_bg_color
        f4.update_xaxes(title_text="longitude")
        f4.update_yaxes(title_text="latitude")
        f4.update_layout(autosize=False, width=1200, height=900)
        f4.update_layout(
            title_text="Outlier Points Captured By Cluster Analysis"
            + "<br><sup>Algorithm Used : DBSCAN (Distance : Euclidean)</sup>"
        )
        f4.write_json(ends_with(master_path) + "cluster_plot_4_dbscan_1_" + col_name)

    except:
        f4 = blank_chart
        f4.update_layout(
            title_text="No Outliers Were Found Using DBSCAN (Distance : Euclidean)"
        )
        f4.write_json(ends_with(master_path) + "cluster_plot_4_dbscan_1_" + col_name)

    try:
        df_outlier_ = DBSCAN_clustered[DBSCAN_clustered.Cluster.values == 999]

        f4_ = go.Figure(
            go.Scatter(
                mode="markers",
                x=df_outlier_[long_col],
                y=df_outlier_[lat_col],
                marker_symbol="x-thin",
                marker_line_color="black",
                marker_color="black",
                marker_line_width=2,
                marker_size=20,
            )
        )
        f4_.layout.plot_bgcolor = global_plot_bg_color
        f4_.layout.paper_bgcolor = global_paper_bg_color
        f4_.update_xaxes(title_text="longitude")
        f4_.update_yaxes(title_text="latitude")
        f4_.update_layout(autosize=False, width=1200, height=900)
        f4_.update_layout(
            title_text="Outlier Points Captured By Cluster Analysis"
            + "<br><sup>Algorithm Used : DBSCAN (Distance : Haversine)</sup>"
        )
        f4_.write_json(ends_with(master_path) + "cluster_plot_4_dbscan_2_" + col_name)

    except:
        f4_ = blank_chart
        f4_.update_layout(
            title_text="No Outliers Were Found Using DBSCAN (Distance : Haversine)"
        )
        f4_.write_json(ends_with(master_path) + "cluster_plot_4_dbscan_2_" + col_name)


def geo_cluster_generator(
    df,
    lat_col_list,
    long_col_list,
    geo_col_list,
    max_cluster,
    eps,
    min_samples,
    master_path,
    global_map_box_val,
    max_records,
):

    """

    This function helps to trigger cluster analysis stats for the identified geospatial fields

    Parameters
    ----------

    df
        Analysis DataFrame

    lat_col_list
        Latitude columns identified in the data
    long_col_list
        Longitude columns identified in the data
    geo_col_list
        Geohash columns identified in the data
    max_cluster
        Maximum number of iterations to decide on the optimum cluster
    eps
        Epsilon value range (Min EPS, Max EPS, Interval) used for DBSCAN clustering
    min_samples
        Minimum Sample Size range (Min Sample Size, Max Sample Size, Interval) used for DBSCAN clustering
    master_path
        Path containing all the output from analyzed data
    global_map_box_val
        Geospatial Chart Theme Index
    max_records
        Maximum geospatial points analyzed

    Returns
    -------

    """
    if isinstance(df, pd.DataFrame):
        pass
    else:
        cnt_records = df.count()
        frac_sample = float(max_records) / float(cnt_records)
        if frac_sample > 1:
            frac_sample_ = 1.0
        else:
            frac_sample_ = float(frac_sample)

        df = df.select(*[lat_col_list + long_col_list + geo_col_list]).dropna()

        if frac_sample_ == 1.0:

            df = df.toPandas()
        else:
            df = data_sampling.data_sample(
                df, strata_cols="all", fraction=frac_sample_
            ).toPandas()

    try:
        lat_col = lat_col_list
        long_col = long_col_list
    except:
        lat_col = []
    try:
        geohash_col = geo_col_list
    except:
        geohash_col = []

    if len(lat_col) >= 1:
        for idx, i in enumerate(lat_col):
            col_name = lat_col[idx] + "_" + long_col[idx]
            geo_cluster_analysis(
                df,
                lat_col[idx],
                long_col[idx],
                max_cluster,
                eps,
                min_samples,
                master_path,
                col_name,
                global_map_box_val,
            )

    if len(geohash_col) >= 1:
        for idx, i in enumerate(geohash_col):
            col_name = geohash_col[idx]
            df_ = df
            df_["latitude"] = df_.apply(
                lambda x: geo_to_latlong(x[col_name], 0), axis=1
            )
            df_["longitude"] = df_.apply(
                lambda x: geo_to_latlong(x[col_name], 1), axis=1
            )

            geo_cluster_analysis(
                df_,
                "latitude",
                "longitude",
                max_cluster,
                eps,
                min_samples,
                master_path,
                col_name,
                global_map_box_val,
            )


def generate_loc_charts_processor(
    df, lat_col, long_col, geohash_col, max_val, id_col, global_map_box_val, master_path
):

    """

    This function helps to generate the output of location charts for the analyzed geospatial fields

    Parameters
    ----------

    df
        Analysis DataFrame

    lat_col
        Latitude columns identified in the data
    long_col
        Longitude columns identified in the data
    geohash_col
        Geohash columns identified in the data
    max_val
        Maximum geospatial points analyzed
    id_col
        ID column
    global_map_box_val
        Geospatial Chart Theme Index
    master_path
        Path containing all the output from analyzed data

    Returns
    -------

    """

    if lat_col:
        cols_to_select = lat_col + long_col + [id_col]
    elif geohash_col:
        cols_to_select = geohash_col + [id_col]

    df = df.select(cols_to_select).dropna()

    if lat_col:

        if len(lat_col) == 1:

            df_ = (
                df.groupBy(lat_col[0], long_col[0])
                .agg(F.countDistinct(id_col).alias("count"))
                .orderBy("count", ascending=False)
                .limit(max_val)
                .toPandas()
            )
            # zoom, center = zoom_center(lons=df_[long_col[0]].values.tolist(),lats=df_[lat_col[0]].values.tolist())
            base_map = px.scatter_mapbox(
                df_,
                lat=lat_col[0],
                lon=long_col[0],
                mapbox_style=mapbox_list[global_map_box_val],
                size="count",
                color_discrete_sequence=global_theme,
            )
            base_map.update_geos(fitbounds="locations")
            base_map.update_layout(
                mapbox_style=mapbox_list[global_map_box_val],
                autosize=False,
                width=1200,
                height=900,
            )
            base_map.write_json(
                ends_with(master_path)
                + "loc_charts_ll_"
                + lat_col[0]
                + "_"
                + long_col[0]
            )

        elif len(lat_col) > 1:
            l = []
            for i in range(0, len(lat_col)):
                df_ = (
                    df.groupBy(lat_col[i], long_col[i])
                    .agg(F.countDistinct(id_col).alias("count"))
                    .orderBy("count", ascending=False)
                    .limit(max_val)
                    .toPandas()
                )
                # zoom, center = zoom_center(lons=df_[long_col[i]].values.tolist(),lats=df_[lat_col[i]].values.tolist())
                base_map = px.scatter_mapbox(
                    df_,
                    lat=lat_col[i],
                    lon=long_col[i],
                    mapbox_style=mapbox_list[global_map_box_val],
                    size="count",
                    color_discrete_sequence=global_theme,
                )
                base_map.update_geos(fitbounds="locations")
                base_map.update_layout(
                    mapbox_style=mapbox_list[global_map_box_val],
                    autosize=False,
                    width=1200,
                    height=900,
                )
                base_map.write_json(
                    ends_with(master_path)
                    + "loc_charts_ll_"
                    + lat_col[i]
                    + "_"
                    + long_col[i]
                )

    if geohash_col:

        if len(geohash_col) == 1:

            col_ = geohash_col[0]
            df_ = (
                df.groupBy(col_)
                .agg(F.countDistinct(id_col).alias("count"))
                .orderBy("count", ascending=False)
                .limit(max_val)
                .toPandas()
            )
            df_["latitude"] = df_.apply(lambda x: geo_to_latlong(x[col_], 0), axis=1)
            df_["longitude"] = df_.apply(lambda x: geo_to_latlong(x[col_], 1), axis=1)
            # zoom, center = zoom_center(lons=df_["longitude"].values.tolist(),lats=df_["latitude"].values.tolist())
            base_map = px.scatter_mapbox(
                df_,
                lat="latitude",
                lon="longitude",
                mapbox_style=mapbox_list[global_map_box_val],
                size="count",
                color_discrete_sequence=global_theme,
            )
            base_map.update_geos(fitbounds="locations")
            base_map.update_layout(
                mapbox_style=mapbox_list[global_map_box_val],
                autosize=False,
                width=1200,
                height=900,
            )
            base_map.write_json(ends_with(master_path) + "loc_charts_gh_" + col_)

        elif len(geohash_col) > 1:

            l = []
            for i in range(0, len(geohash_col)):
                col_ = geohash_col[i]
                df_ = (
                    df.groupBy(col_)
                    .agg(F.countDistinct(id_col).alias("count"))
                    .orderBy("count", ascending=False)
                    .limit(max_val)
                    .toPandas()
                )
                df_["latitude"] = df_.apply(
                    lambda x: geo_to_latlong(x[col_], 0), axis=1
                )
                df_["longitude"] = df_.apply(
                    lambda x: geo_to_latlong(x[col_], 1), axis=1
                )
                # zoom, center = zoom_center(lons=df_["longitude"].values.tolist(),lats=df_["latitude"].values.tolist())
                base_map = px.scatter_mapbox(
                    df_,
                    lat="latitude",
                    lon="longitude",
                    mapbox_style=mapbox_list[global_map_box_val],
                    size="count",
                    color_discrete_sequence=global_theme,
                )
                base_map.update_geos(fitbounds="locations")
                base_map.update_layout(
                    mapbox_style=mapbox_list[global_map_box_val],
                    autosize=False,
                    width=1200,
                    height=900,
                )
                base_map.write_json(ends_with(master_path) + "loc_charts_gh_" + col_)


def generate_loc_charts_controller(
    df, id_col, lat_col, long_col, geohash_col, max_val, global_map_box_val, master_path
):

    """

    This function helps to trigger the output generation of location charts for the analyzed geospatial fields

    Parameters
    ----------

    df
        Analysis DataFrame
    id_col
        ID column
    lat_col
        Latitude columns identified in the data
    long_col
        Longitude columns identified in the data
    geohash_col
        Geohash columns identified in the data
    max_val
        Maximum geospatial points analyzed
    global_map_box_val
        Geospatial Chart Theme Index
    master_path
        Path containing all the output from analyzed data

    Returns
    -------

    """

    if lat_col:
        len_lat = len(lat_col)
        ll_plot = generate_loc_charts_processor(
            df,
            lat_col=lat_col,
            long_col=long_col,
            geohash_col=None,
            max_val=max_val,
            id_col=id_col,
            global_map_box_val=global_map_box_val,
            master_path=master_path,
        )

    else:
        len_lat = 0

    if geohash_col:
        len_geohash_col = len(geohash_col)
        geohash_plot = generate_loc_charts_processor(
            df,
            lat_col=None,
            long_col=None,
            geohash_col=geohash_col,
            max_val=max_val,
            id_col=id_col,
            global_map_box_val=global_map_box_val,
            master_path=master_path,
        )

    else:
        len_geohash_col = 0

    if (len_lat + len_geohash_col) == 1:

        if len_lat == 0:
            return geohash_plot
        else:
            return ll_plot

    elif (len_lat + len_geohash_col) > 1:

        if (len_lat > 1) and (len_geohash_col == 0):

            return ll_plot

        elif (len_lat == 0) and (len_geohash_col > 1):

            return geohash_plot

        elif (len_lat >= 1) and (len_geohash_col >= 1):

            return ll_plot, geohash_plot


def geospatial_autodetection(
    df,
    id_col,
    master_path,
    max_records,
    top_geo_records,
    max_cluster,
    eps,
    min_samples,
    global_map_box_val,
    run_type,
    auth_key,
):

    """

    This function helps to trigger the output of intermediate data which is further used for producing the geospatial analyzer report

    Parameters
    ----------

    df
        Analysis DataFrame
    id_col
        ID column
    master_path
        Path containing all the output from analyzed data
    max_records
            Maximum geospatial points analyzed
    top_geo_records
        Top geospatial records displayed
    max_cluster
        Maximum number of iterations to decide on the optimum cluster
    eps
        Epsilon value range (Min EPS, Max EPS, Interval) used for DBSCAN clustering
    min_samples
        Minimum Sample Size range (Min Sample Size, Max Sample Size, Interval) used for DBSCAN clustering
    global_map_box_val
        Geospatial Chart Theme Index
    run_type
        Option to choose between run type "Local" or "EMR" or "Azure" or "ak8s" basis the user flexibility. Default option is set as "Local"
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"

    Returns
    -------

    """

    if run_type == "local":
        local_path = master_path
    elif run_type == "databricks":
        local_path = output_to_local(master_path)
    elif run_type in ("emr", "ak8s"):
        local_path = "report_stats"
    else:
        raise ValueError("Invalid run_type")

    Path(local_path).mkdir(parents=True, exist_ok=True)

    lat_cols, long_cols, gh_cols = ll_gh_cols(df, max_records)

    try:
        len_lat_col = len(lat_cols)

    except:

        len_lat_col = 0

    try:
        len_geohash_col = len(gh_cols)
    except:
        len_geohash_col = 0

    if (len_lat_col > 0) or (len_geohash_col > 0):

        all_cols = lat_cols + long_cols + gh_cols

        df.persist()

        stats_gen_lat_long_geo(
            df, lat_cols, long_cols, gh_cols, id_col, local_path, top_geo_records
        )

        geo_cluster_generator(
            df,
            lat_cols,
            long_cols,
            gh_cols,
            max_cluster,
            eps,
            min_samples,
            local_path,
            global_map_box_val,
            max_records,
        )

        generate_loc_charts_controller(
            df,
            id_col,
            lat_cols,
            long_cols,
            gh_cols,
            max_records,
            global_map_box_val,
            local_path,
        )

        return lat_cols, long_cols, gh_cols

    elif len_lat_col + len_geohash_col == 0:

        return [], [], []

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp --recursive "
            + ends_with(local_path)
            + " "
            + ends_with(master_path)
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])

    if run_type == "ak8s":
        output_path_mod = path_ak8s_modify(master_path)
        bash_cmd = (
            'azcopy cp "'
            + ends_with(local_path)
            + '" "'
            + ends_with(output_path_mod)
            + str(auth_key)
            + '" --recursive=true '
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])
