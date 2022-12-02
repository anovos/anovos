# coding=utf-8

"""This module generates the final report output specific to the intermediate data generated across each of the modules. The final report, however, can be proccessed through the config.yaml file or by generating it through the respective functions.

Below are some of the functions used to process the final output.

- line_chart_gen_stability
- data_analyzer_output
- drift_stability_ind
- chart_gen_list
- executive_summary_gen
- wiki_generator
- descriptive_statistics
- quality_check
- attribute_associations
- data_drift_stability
- plotSeasonalDecompose
- gen_time_series_plots
- list_ts_remove_append
- ts_viz_1_1 — ts_viz_1_3
- ts_viz_2_1 — ts_viz_2_3
- ts_viz_3_1 — ts_viz_3_3
- ts_landscape
- ts_stats
- ts_viz_generate
- overall_stats_gen
- loc_field_stats
- read_stats_ll_geo
- read_cluster_stats_ll_geo
- read_loc_charts
- loc_report_gen
- anovos_report

However, each of the functions have been detailed in the respective sections across the parameters used.

"""

import json
import os
import subprocess
import warnings

import datapane as dp
import dateutil.parser
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from anovos.shared.utils import ends_with, output_to_local, path_ak8s_modify

warnings.filterwarnings("ignore")

global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
global_plot_bg_color = "rgba(0,0,0,0)"
global_paper_bg_color = "rgba(0,0,0,0)"
default_template = (
    dp.HTML(
        """
        <html>
            <img src="https://mobilewalla-anovos.s3.amazonaws.com/anovos.png"
                style="height:100px;display:flex;margin:auto;float:right"
            />
        </html>"""
    ),
    dp.Text("# ML-Anovos Report"),
)


def remove_u_score(col):
    """
    This functions help to remove the "_" present in a specific text
    Parameters
    ----------
    col
        Analysis column containing "_" present gets replaced along with upper case conversion
    Returns
    -------
    String
    """
    col_ = col.split("_")
    bl = []
    for i in col_:
        if i == "nullColumns" or i == "nullRows":
            bl.append("Null")
        else:
            bl.append(i[0].upper() + i[1:])
    return " ".join(bl)


def line_chart_gen_stability(df1, df2, col):

    """
    This function helps to produce charts which are specific to data stability index. It taken into account the stability input along with the analysis column to produce the desired output.
    Parameters
    ----------
    df1
        Analysis dataframe pertaining to summarized stability metrics
    df2
        Analysis dataframe pertaining to historical data
    col
        Analysis column
    Returns
    -------
    DatapaneObject
    """

    def val_cat(val):
        """
        Parameters
        ----------
        val

        Returns
        -------
        String
        """
        if val >= 3.5:
            return "Very Stable"
        elif val >= 3 and val < 3.5:
            return "Stable"
        elif val >= 2 and val < 3:
            return "Marginally Stable"
        elif val >= 1 and val < 2:
            return "Unstable"
        elif val >= 0 and val < 1:
            return "Very Unstable"
        else:
            return "Out of Range"

    val_si = list(df2[df2["attribute"] == col].stability_index.values)[0]
    f1 = go.Figure()
    f1.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=val_si,
            gauge={
                "axis": {"range": [None, 4], "tickwidth": 1, "tickcolor": "black"},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 1], "color": px.colors.sequential.Reds[7]},
                    {"range": [1, 2], "color": px.colors.sequential.Reds[6]},
                    {"range": [2, 3], "color": px.colors.sequential.Oranges[4]},
                    {"range": [3, 3.5], "color": px.colors.sequential.BuGn[7]},
                    {"range": [3.5, 4], "color": px.colors.sequential.BuGn[8]},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 1,
                    "value": val_si,
                },
                "bar": {"color": global_plot_bg_color},
            },
            title={"text": "Order of Stability: " + val_cat(val_si)},
        )
    )
    f1.update_layout(height=400, font={"color": "black", "family": "Arial"})
    f5 = "Stability Index for " + str(col.upper())
    if len(df1.columns) > 0:
        attr_type = df1["type"].tolist()[0]
        if attr_type == "Numerical":
            f2 = px.line(
                df1,
                x="idx",
                y="mean",
                markers=True,
                title="CV of Mean is "
                + str(list(df2[df2["attribute"] == col].mean_cv.values)[0]),
            )
            f2.update_traces(line_color=global_theme[2], marker=dict(size=14))
            f2.layout.plot_bgcolor = global_plot_bg_color
            f2.layout.paper_bgcolor = global_paper_bg_color
            f3 = px.line(
                df1,
                x="idx",
                y="stddev",
                markers=True,
                title="CV of Stddev is "
                + str(list(df2[df2["attribute"] == col].stddev_cv.values)[0]),
            )
            f3.update_traces(line_color=global_theme[6], marker=dict(size=14))
            f3.layout.plot_bgcolor = global_plot_bg_color
            f3.layout.paper_bgcolor = global_paper_bg_color
            f4 = px.line(
                df1,
                x="idx",
                y="kurtosis",
                markers=True,
                title="CV of Kurtosis is "
                + str(list(df2[df2["attribute"] == col].kurtosis_cv.values)[0]),
            )
            f4.update_traces(line_color=global_theme[4], marker=dict(size=14))
            f4.layout.plot_bgcolor = global_plot_bg_color
            f4.layout.paper_bgcolor = global_paper_bg_color
            return dp.Group(
                dp.Text("#"),
                dp.Text(f5),
                dp.Plot(f1),
                dp.Group(dp.Plot(f2), dp.Plot(f3), dp.Plot(f4), columns=3),
                label=col,
            )
        else:
            f2 = px.line(
                df1,
                x="idx",
                y="mean",
                markers=True,
                title="Standard deviation of Mean is "
                + str(list(df2[df2["attribute"] == col].mean_stddev.values)[0]),
            )
            f2.update_traces(line_color=global_theme[2], marker=dict(size=14))
            f2.layout.plot_bgcolor = global_plot_bg_color
            f2.layout.paper_bgcolor = global_paper_bg_color
            return dp.Group(
                dp.Text("#"),
                dp.Text(f5),
                dp.Plot(f1),
                dp.Group(dp.Plot(f2), columns=1),
                label=col,
            )
    else:
        return dp.Group(dp.Text("#"), dp.Text(f5), dp.Plot(f1), label=col)


def data_analyzer_output(master_path, avl_recs_tab, tab_name):

    """
    This section produces output in form of datapane objects which is specific to the different data analyzer modules. It is used by referring to the Master path along with the Available list of metrics & the Tab name.
    Parameters
    ----------
    master_path
        Path containing all the output from analyzed data
    avl_recs_tab
        Available file names from the analysis tab
    tab_name
        Analysis tab from association_evaluator / quality_checker / stats_generator
    Returns
    -------
    DatapaneObject
    """

    df_list = []
    df_plot_list = []
    # @FIXME: unused variables
    plot_list = []
    avl_recs_tab = [x for x in avl_recs_tab if "global_summary" not in x]
    for index, i in enumerate(avl_recs_tab):
        data = pd.read_csv(ends_with(master_path) + str(i) + ".csv")
        if len(data.index) == 0:
            continue
        if tab_name == "quality_checker":
            if i == "duplicate_detection":
                duplicate_recs = pd.read_csv(
                    ends_with(master_path) + str(i) + ".csv"
                ).round(3)
                _unique_rows_count = int(
                    duplicate_recs[
                        duplicate_recs["metric"] == "unique_rows_count"
                    ].value.values
                )
                _rows_count = int(
                    duplicate_recs[
                        duplicate_recs["metric"] == "rows_count"
                    ].value.values
                )
                _duplicate_rows_count = int(
                    duplicate_recs[
                        duplicate_recs["metric"] == "duplicate_rows"
                    ].value.values
                )
                _duplicate_pct = float(
                    duplicate_recs[
                        duplicate_recs["metric"] == "duplicate_pct"
                    ].value.values
                    * 100.0
                )
                unique_rows_count = f" No. Of Unique Rows: **{_unique_rows_count}**"
                # @FIXME: variable names exists in outer scope
                rows_count = f" No. of Rows: **{_rows_count}**"
                duplicate_rows = f" No. of Duplicate Rows: **{_duplicate_rows_count}**"
                duplicate_pct = f" Percentage of Duplicate Rows: **{_duplicate_pct}%**"
                df_list.append(
                    [
                        dp.Text("### " + str(remove_u_score(i))),
                        dp.Group(
                            dp.Text(rows_count),
                            dp.Text(unique_rows_count),
                            dp.Text(duplicate_rows),
                            dp.Text(duplicate_pct),
                        ),
                        dp.Text("#"),
                        dp.Text("#"),
                    ]
                )
            elif i == "outlier_detection":
                df_list.append(
                    [
                        dp.Text("### " + str(remove_u_score(i))),
                        dp.DataTable(
                            pd.read_csv(ends_with(master_path) + str(i) + ".csv").round(
                                3
                            )
                        ),
                        "outlier_charts_placeholder",
                    ]
                )
            else:
                df_list.append(
                    [
                        dp.Text("### " + str(remove_u_score(i))),
                        dp.DataTable(
                            pd.read_csv(ends_with(master_path) + str(i) + ".csv").round(
                                3
                            )
                        ),
                        dp.Text("#"),
                        dp.Text("#"),
                    ]
                )
        elif tab_name == "association_evaluator":
            for j in avl_recs_tab:
                if j == "correlation_matrix":
                    df_list_ = pd.read_csv(
                        ends_with(master_path) + str(j) + ".csv"
                    ).round(3)
                    feats_order = list(df_list_["attribute"].values)
                    df_list_ = df_list_.round(3)
                    fig = px.imshow(
                        df_list_[feats_order],
                        y=feats_order,
                        color_continuous_scale=global_theme,
                        aspect="auto",
                    )
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
                    # fig.update_layout(title_text=str("Correlation Plot "))
                    df_plot_list.append(
                        dp.Group(
                            dp.Text("##"),
                            dp.DataTable(df_list_[["attribute"] + feats_order]),
                            dp.Plot(fig),
                            label=remove_u_score(j),
                        )
                    )
                elif j == "variable_clustering":
                    df_list_ = (
                        pd.read_csv(ends_with(master_path) + str(j) + ".csv")
                        .round(3)
                        .sort_values(by=["Cluster"], ascending=True)
                    )
                    fig = px.sunburst(
                        df_list_,
                        path=["Cluster", "Attribute"],
                        values="RS_Ratio",
                        color_discrete_sequence=global_theme,
                    )
                    # fig.update_layout(title_text=str("Distribution of homogenous variable across Clusters"))
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
                    # fig.update_layout(title_text=str("Variable Clustering Plot "))
                    fig.layout.autosize = True
                    df_plot_list.append(
                        dp.Group(
                            dp.Text("##"),
                            dp.DataTable(df_list_),
                            dp.Plot(fig),
                            label=remove_u_score(j),
                        )
                    )
                else:
                    try:
                        df_list_ = pd.read_csv(
                            ends_with(master_path) + str(j) + ".csv"
                        ).round(3)
                        col_nm = [
                            x for x in list(df_list_.columns) if "attribute" not in x
                        ]
                        df_list_ = df_list_.sort_values(col_nm[0], ascending=True)
                        fig = px.bar(
                            df_list_,
                            x=col_nm[0],
                            y="attribute",
                            orientation="h",
                            color_discrete_sequence=global_theme,
                        )
                        fig.layout.plot_bgcolor = global_plot_bg_color
                        fig.layout.paper_bgcolor = global_paper_bg_color
                        # fig.update_layout(title_text=str("Representation of " + str(remove_u_score(j))))
                        fig.layout.autosize = True
                        df_plot_list.append(
                            dp.Group(
                                dp.Text("##"),
                                dp.DataTable(df_list_),
                                dp.Plot(fig),
                                label=remove_u_score(j),
                            )
                        )
                    except Exception as e:
                        logger.error(f"processing failed, error {e}")
                        pass
            if len(avl_recs_tab) == 1:
                df_plot_list.append(
                    dp.Group(
                        dp.DataTable(
                            pd.DataFrame(columns=[" "], index=range(1)), label=" "
                        ),
                        dp.Plot(blank_chart, label=" "),
                        label=" ",
                    )
                )
            else:
                pass
            return df_plot_list
        else:
            df_list.append(
                dp.DataTable(
                    pd.read_csv(ends_with(master_path) + str(i) + ".csv").round(3),
                    label=remove_u_score(avl_recs_tab[index]),
                )
            )
    if tab_name == "quality_checker" and len(avl_recs_tab) == 1:
        return df_list[0], [dp.Text("#"), dp.Plot(blank_chart)]
    elif tab_name == "stats_generator" and len(avl_recs_tab) == 1:
        return [
            df_list[0],
            dp.DataTable(pd.DataFrame(columns=[" "], index=range(1)), label=" "),
        ]
    else:
        return df_list


def drift_stability_ind(
    missing_recs_drift, drift_tab, missing_recs_stability, stability_tab
):
    """
    This function helps to produce the drift & stability indicator for further processing. Ideally a data with both drift & stability should produce a list of [1,1]
    Parameters
    ----------
    missing_recs_drift
        Missing files from the drift tab
    drift_tab
        "drift_statistics"
    missing_recs_stability
        Missing files from the stability tab
    stability_tab
        "stability_index, stabilityIndex_metrics"

    Returns
    -------
    List
    """
    if len(missing_recs_drift) == len(drift_tab):
        drift_ind = 0
    else:
        drift_ind = 1
    if len(missing_recs_stability) == len(stability_tab):
        stability_ind = 0
    elif ("stabilityIndex_metrics" in missing_recs_stability) and (
        "stability_index" not in missing_recs_stability
    ):
        stability_ind = 0.5
    else:
        stability_ind = 1
    return drift_ind, stability_ind


def chart_gen_list(master_path, chart_type, type_col=None):

    """
    This function helps to produce the charts in a list object form nested by a datapane object.
    Parameters
    ----------
    master_path
        Path containing all the charts same as the other files from data analyzed output
    chart_type
        Files containing only the specific chart names for the specific chart category
    type_col
        None. Default value is kept as None
    Returns
    -------
    DatapaneObject
    """
    plot_list = []
    for i in chart_type:
        col_name = i[i.find("_") + 1 :]
        if type_col == "numerical":
            if col_name in numcols_name.replace(" ", "").split(","):
                plot_list.append(
                    dp.Plot(
                        go.Figure(json.load(open(ends_with(master_path) + i))),
                        label=col_name,
                    )
                )
            else:
                pass
        elif type_col == "categorical":
            if col_name in catcols_name.replace(" ", "").split(","):
                plot_list.append(
                    dp.Plot(
                        go.Figure(json.load(open(ends_with(master_path) + i))),
                        label=col_name,
                    )
                )
            else:
                pass
        else:
            plot_list.append(
                dp.Plot(
                    go.Figure(json.load(open(ends_with(master_path) + i))),
                    label=col_name,
                )
            )
    return plot_list


def executive_summary_gen(
    master_path,
    label_col,
    ds_ind,
    id_col,
    iv_threshold,
    corr_threshold,
    print_report=False,
):
    """
    This function helps to produce output specific to the Executive Summary Tab.
    Parameters
    ----------
    master_path
        Path containing the input files.
    label_col
        Label column.
    ds_ind
        Drift stability indicator in list form.
    id_col
        ID column.
    iv_threshold
        IV threshold beyond which attributes can be called as significant.
    corr_threshold
        Correlation threshold beyond which attributes can be categorized under correlated.
    print_report
        Printing option flexibility. Default value is kept as False.
    Returns
    -------
    DatapaneObject / Output[HTML]
    """
    try:
        obj_dtls = json.load(
            open(ends_with(master_path) + "freqDist_" + str(label_col))
        )
        # @FIXME: never used local variable
        text_val = list(list(obj_dtls.values())[0][0].items())[8][1]
        x_val = list(list(obj_dtls.values())[0][0].items())[10][1]
        y_val = list(list(obj_dtls.values())[0][0].items())[12][1]
        label_fig_ = go.Figure(
            data=[
                go.Pie(
                    labels=x_val,
                    values=y_val,
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    pull=[0, 0.1],
                    marker_colors=global_theme,
                )
            ]
        )
        label_fig_.update_traces(textposition="inside", textinfo="percent+label")
        label_fig_.update_layout(
            legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center")
        )
        label_fig_.layout.plot_bgcolor = global_plot_bg_color
        label_fig_.layout.paper_bgcolor = global_paper_bg_color
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        label_fig_ = None
    a1 = (
        "The dataset contains  **"
        + str(f"{rows_count:,d}")
        + "** records and **"
        + str(numcols_count + catcols_count)
        + "** attributes (**"
        + str(numcols_count)
        + "** numerical + **"
        + str(catcols_count)
        + "** categorical)."
    )
    if label_col is None:
        a2 = dp.Group(
            dp.Text("- There is **no** target variable in the dataset"),
            dp.Text("- Data Diagnosis:"),
        )
    else:
        if label_fig_ is None:
            a2 = dp.Group(
                dp.Text("- Target variable is **" + str(label_col) + "** "),
                dp.Text("- Data Diagnosis:"),
            )
        else:
            a2 = dp.Group(
                dp.Text("- Target variable is **" + str(label_col) + "** "),
                dp.Plot(label_fig_),
                dp.Text("- Data Diagnosis:"),
            )
    try:
        x1 = list(
            pd.read_csv(ends_with(master_path) + "measures_of_dispersion.csv")
            .query("`cov`>1")
            .attribute.values
        )
        if len(x1) > 0:
            x1_1 = ["High Variance", x1]
        else:
            x1_1 = ["High Variance", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x1_1 = ["High Variance", None]
    try:
        x2 = list(
            pd.read_csv(ends_with(master_path) + "measures_of_shape.csv")
            .query("`skewness`>0")
            .attribute.values
        )
        if len(x2) > 0:
            x2_1 = ["Positive Skewness", x2]
        else:
            x2_1 = ["Positive Skewness", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x2_1 = ["Positive Skewness", None]
    try:
        x3 = list(
            pd.read_csv(ends_with(master_path) + "measures_of_shape.csv")
            .query("`skewness`<0")
            .attribute.values
        )
        if len(x3) > 0:
            x3_1 = ["Negative Skewness", x3]
        else:
            x3_1 = ["Negative Skewness", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x3_1 = ["Negative Skewness", None]
    try:
        x4 = list(
            pd.read_csv(ends_with(master_path) + "measures_of_shape.csv")
            .query("`kurtosis`>0")
            .attribute.values
        )
        if len(x4) > 0:
            x4_1 = ["High Kurtosis", x4]
        else:
            x4_1 = ["High Kurtosis", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x4_1 = ["High Kurtosis", None]
    try:
        x5 = list(
            pd.read_csv(ends_with(master_path) + "measures_of_shape.csv")
            .query("`kurtosis`<0")
            .attribute.values
        )
        if len(x5) > 0:
            x5_1 = ["Low Kurtosis", x5]
        else:
            x5_1 = ["Low Kurtosis", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x5_1 = ["Low Kurtosis", None]
    try:
        x6 = list(
            pd.read_csv(ends_with(master_path) + "measures_of_counts.csv")
            .query("`fill_pct`<0.7")
            .attribute.values
        )
        if len(x6) > 0:
            x6_1 = ["Low Fill Rates", x6]
        else:
            x6_1 = ["Low Fill Rates", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x6_1 = ["Low Fill Rates", None]
    try:
        biasedness_df = pd.read_csv(ends_with(master_path) + "biasedness_detection.csv")
        if "treated" in biasedness_df:
            x7 = list(biasedness_df.query("`treated`>0").attribute.values)
        else:
            x7 = list(biasedness_df.query("`flagged`>0").attribute.values)
        if len(x7) > 0:
            x7_1 = ["High Biasedness", x7]
        else:
            x7_1 = ["High Biasedness", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x7_1 = ["High Biasedness", None]
    try:
        x8 = list(
            pd.read_csv(
                ends_with(master_path) + "outlier_detection.csv"
            ).attribute.values
        )
        if len(x8) > 0:
            x8_1 = ["Outliers", x8]
        else:
            x8_1 = ["Outliers", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x8_1 = ["Outliers", None]
    try:
        corr_matrx = pd.read_csv(ends_with(master_path) + "correlation_matrix.csv")
        corr_matrx = corr_matrx[list(corr_matrx.attribute.values)]
        corr_matrx = corr_matrx.where(
            np.triu(np.ones(corr_matrx.shape), k=1).astype(np.bool)
        )
        to_drop = [
            column
            for column in corr_matrx.columns
            if any(corr_matrx[column] > corr_threshold)
        ]
        if len(to_drop) > 0:
            x9_1 = ["High Correlation", to_drop]
        else:
            x9_1 = ["High Correlation", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x9_1 = ["High Correlation", None]
    try:
        x10 = list(
            pd.read_csv(ends_with(master_path) + "IV_calculation.csv")
            .query("`iv`>" + str(iv_threshold))
            .attribute.values
        )
        if len(x10) > 0:
            x10_1 = ["Significant Attributes", x10]
        else:
            x10_1 = ["Significant Attributes", None]
    except Exception as e:
        logger.error(f"processing failed, error {e}")
        x10_1 = ["Significant Attributes", None]
    blank_list_df = []
    for i in [x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1, x10_1]:
        try:
            for j in i[1]:
                blank_list_df.append([i[0], j])
        except Exception as e:
            logger.error(f"processing failed, error {e}")
            blank_list_df.append([i[0], "NA"])
    list_n = []
    x1 = pd.DataFrame(blank_list_df, columns=["Metric", "Attribute"])
    x1["Value"] = "✔"
    all_cols = (
        catcols_name.replace(" ", "") + "," + numcols_name.replace(" ", "")
    ).split(",")
    remainder_cols = list(set(all_cols) - set(x1.Attribute.values))
    total_metrics = set(list(x1.Metric.values))
    for i in remainder_cols:
        for j in total_metrics:
            list_n.append([j, i])
    x2 = pd.DataFrame(list_n, columns=["Metric", "Attribute"])
    x2["Value"] = "✘"
    x = x1.append(x2, ignore_index=True)
    x = (
        x.drop_duplicates()
        .pivot(index="Attribute", columns="Metric", values="Value")
        .fillna("✘")
        .reset_index()[
            [
                "Attribute",
                "Outliers",
                "Significant Attributes",
                "Positive Skewness",
                "Negative Skewness",
                "High Variance",
                "High Correlation",
                "High Kurtosis",
                "Low Kurtosis",
            ]
        ]
    )
    x = x[
        ~(
            (x["Attribute"].isnull())
            | (x.Attribute.values == "NA")
            | (x["Attribute"] == " ")
        )
    ]
    if ds_ind[0] == 1 and ds_ind[1] >= 0.5:
        a5 = "Data Health based on Drift Metrics & Stability Index : "
        report = dp.Group(
            dp.Text("# "),
            dp.Text("**Key Report Highlights**"),
            dp.Text("- " + a1),
            a2,
            dp.DataTable(x),
            dp.Text("- " + a5),
            dp.Group(
                dp.BigNumber(
                    heading="# Drifted Attributes",
                    value=str(str(drifted_feats) + " out of " + str(len_feats)),
                ),
                dp.BigNumber(
                    heading="% Drifted Attributes",
                    value=str(np.round((100 * drifted_feats / len_feats), 2)) + "%",
                ),
                dp.BigNumber(
                    heading="# Unstable Attributes",
                    value=str(len(unstable_attr))
                    + " out of "
                    + str(len(total_unstable_attr)),
                    change="numerical",
                    is_upward_change=True,
                ),
                dp.BigNumber(
                    heading="% Unstable Attributes",
                    value=str(
                        np.round(100 * len(unstable_attr) / len(total_unstable_attr), 2)
                    )
                    + "%",
                ),
                columns=4,
            ),
            dp.Text("# "),
            dp.Text("# "),
            label="Executive Summary",
        )
    if ds_ind[0] == 0 and ds_ind[1] >= 0.5:
        a5 = "Data Health based on Stability Index : "
        report = dp.Group(
            dp.Text("# "),
            dp.Text("**Key Report Highlights**"),
            dp.Text("# "),
            dp.Text("- " + a1),
            a2,
            dp.DataTable(x),
            dp.Text("- " + a5),
            dp.Group(
                dp.BigNumber(
                    heading="# Unstable Attributes",
                    value=str(len(unstable_attr))
                    + " out of "
                    + str(len(total_unstable_attr)),
                    change="numerical",
                    is_upward_change=True,
                ),
                dp.BigNumber(
                    heading="% Unstable Attributes",
                    value=str(
                        np.round(100 * len(unstable_attr) / len(total_unstable_attr), 2)
                    )
                    + "%",
                ),
                columns=2,
            ),
            dp.Text("# "),
            dp.Text("# "),
            label="Executive Summary",
        )
    if ds_ind[0] == 1 and ds_ind[1] == 0:
        a5 = "Data Health based on Drift Metrics : "
        report = dp.Group(
            dp.Text("# "),
            dp.Text("**Key Report Highlights**"),
            dp.Text("# "),
            dp.Text("- " + a1),
            a2,
            dp.DataTable(x),
            dp.Text("- " + a5),
            dp.Group(
                dp.BigNumber(
                    heading="# Drifted Attributes",
                    value=str(str(drifted_feats) + " out of " + str(len_feats)),
                ),
                dp.BigNumber(
                    heading="% Drifted Attributes",
                    value=str(np.round((100 * drifted_feats / len_feats), 2)) + "%",
                ),
                columns=2,
            ),
            dp.Text("# "),
            dp.Text("# "),
            label="Executive Summary",
        )
    if ds_ind[0] == 0 and ds_ind[1] == 0:
        report = dp.Group(
            dp.Text("# "),
            dp.Text("**Key Report Highlights**"),
            dp.Text("# "),
            dp.Text("- " + a1),
            a2,
            dp.DataTable(x),
            dp.Text("# "),
            label="Executive Summary",
        )
    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "executive_summary.html", open=True
        )
    return report


# @FIXME: rename variables with their corresponding within the config files
def wiki_generator(
    master_path, dataDict_path=None, metricDict_path=None, print_report=False
):
    """
    This function helps to produce output specific to the Wiki Tab.
    Parameters
    ----------
    master_path
        Path containing the input files.
    dataDict_path
        Data dictionary path. Default value is kept as None.
    metricDict_path
        Metric dictionary path. Default value is kept as None.
    print_report
        Printing option flexibility. Default value is kept as False.
    Returns
    -------
    DatapaneObject / Output[HTML]
    """
    try:
        datatype_df = pd.read_csv(ends_with(master_path) + "data_type.csv")
    except FileNotFoundError:
        logger.error(
            f"file {master_path}/data_type.csv doesn't exist, cannot read datatypes"
        )
    except Exception:
        logger.info("generate an empty dataframe with columns attribute and data_type ")
        datatype_df = pd.DataFrame(columns=["attribute", "data_type"], index=range(1))
    try:
        data_dict = pd.read_csv(dataDict_path).merge(
            datatype_df, how="outer", on="attribute"
        )
    except FileNotFoundError:
        logger.error(f"file {dataDict_path} doesn't exist, cannot read data dict")
    except Exception:
        data_dict = datatype_df
    try:
        metric_dict = pd.read_csv(metricDict_path)
    except FileNotFoundError:
        logger.error(f"file {metricDict_path} doesn't exist, cannot read metrics dict")
    except Exception:
        metric_dict = pd.DataFrame(
            columns=[
                "Section Category",
                "Section Name",
                "Metric Name",
                "Metric Definitions",
            ],
            index=range(1),
        )
    report = dp.Group(
        dp.Text("# "),
        dp.Text(
            """
            *A quick reference to the attributes from the dataset (Data Dictionary)
            and the metrics computed in the report (Metric Dictionary).*
            """
        ),
        dp.Text("# "),
        dp.Text("# "),
        dp.Select(
            blocks=[
                dp.Group(
                    dp.Group(dp.Text("## "), dp.DataTable(data_dict)),
                    label="Data Dictionary",
                ),
                dp.Group(
                    dp.Text("##"), dp.DataTable(metric_dict), label="Metric Dictionary"
                ),
            ],
            type=dp.SelectType.TABS,
        ),
        dp.Text("# "),
        dp.Text("# "),
        dp.Text("# "),
        dp.Text("# "),
        label="Wiki",
    )
    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "wiki_generator.html", open=True
        )
    return report


def descriptive_statistics(
    master_path,
    SG_tabs,
    avl_recs_SG,
    missing_recs_SG,
    all_charts_num_1_,
    all_charts_cat_1_,
    print_report=False,
):
    """
    This function helps to produce output specific to the Descriptive Stats Tab.
    Parameters
    ----------
    master_path
        Path containing the input files.
    SG_tabs
        measures_of_counts','measures_of_centralTendency','measures_of_cardinality','measures_of_percentiles','measures_of_dispersion','measures_of_shape','global_summary'
    avl_recs_SG
        Available files from the SG_tabs (Stats Generator tabs)
    missing_recs_SG
        Missing files from the SG_tabs (Stats Generator tabs)
    all_charts_num_1_
        Numerical charts (histogram) all collated in a list format supported as per datapane objects
    all_charts_cat_1_
        Categorical charts (barplot) all collated in a list format supported as per datapane objects
    print_report
        Printing option flexibility. Default value is kept as False.
    Returns
    -------
    DatapaneObject / Output[HTML]
    """
    if "global_summary" in avl_recs_SG:
        cnt = 0
    else:
        cnt = 1
    if len(missing_recs_SG) + cnt == len(SG_tabs):
        return "null_report"
    else:
        if "global_summary" in avl_recs_SG:
            l1 = dp.Group(
                dp.Text("# "),
                dp.Text(
                    "*This section summarizes the dataset with key statistical metrics and distribution plots.*"
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Global Summary"),
                dp.Group(
                    dp.Text(" Total Number of Records: **" + f"{rows_count:,}" + "**"),
                    dp.Text(
                        " Total Number of Attributes: **" + str(columns_count) + "**"
                    ),
                    dp.Text(
                        " Number of Numerical Attributes : **"
                        + str(numcols_count)
                        + "**"
                    ),
                    dp.Text(
                        " Numerical Attributes Name : **" + str(numcols_name) + "**"
                    ),
                    dp.Text(
                        " Number of Categorical Attributes : **"
                        + str(catcols_count)
                        + "**"
                    ),
                    dp.Text(
                        " Categorical Attributes Name : **" + str(catcols_name) + "**"
                    ),
                ),
            )
        else:
            l1 = dp.Text("# ")
        if len(data_analyzer_output(master_path, avl_recs_SG, "stats_generator")) > 0:
            l2 = dp.Text("### Statistics by Metric Type")
            l3 = dp.Group(
                dp.Select(
                    blocks=data_analyzer_output(
                        master_path, avl_recs_SG, "stats_generator"
                    ),
                    type=dp.SelectType.TABS,
                ),
                dp.Text("# "),
            )
        else:
            l2 = dp.Text("# ")
            l3 = dp.Text("# ")
        if len(all_charts_num_1_) == 0 and len(all_charts_cat_1_) == 0:
            l4 = 1
        elif len(all_charts_num_1_) == 0 and len(all_charts_cat_1_) > 0:
            l4 = (
                dp.Text("# "),
                dp.Text("### Attribute Visualization"),
                dp.Select(blocks=all_charts_cat_1_, type=dp.SelectType.DROPDOWN),
                dp.Text("# "),
                dp.Text("# "),
            )
        elif len(all_charts_num_1_) > 0 and len(all_charts_cat_1_) == 0:
            l4 = (
                dp.Text("# "),
                dp.Text("### Attribute Visualization"),
                dp.Select(blocks=all_charts_num_1_, type=dp.SelectType.DROPDOWN),
                dp.Text("# "),
                dp.Text("# "),
            )
        else:
            l4 = (
                dp.Text("# "),
                dp.Text("### Attribute Visualization"),
                dp.Group(
                    dp.Select(
                        blocks=[
                            dp.Group(
                                dp.Select(
                                    blocks=all_charts_num_1_,
                                    type=dp.SelectType.DROPDOWN,
                                ),
                                label="Numerical",
                            ),
                            dp.Group(
                                dp.Select(
                                    blocks=all_charts_cat_1_,
                                    type=dp.SelectType.DROPDOWN,
                                ),
                                label="Categorical",
                            ),
                        ],
                        type=dp.SelectType.TABS,
                    )
                ),
                dp.Text("# "),
                dp.Text("# "),
            )
    if l4 == 1:
        report = dp.Group(
            l1,
            dp.Text("# "),
            l2,
            l3,
            dp.Text("# "),
            dp.Text("# "),
            label="Descriptive Statistics",
        )
    else:
        report = dp.Group(
            l1,
            dp.Text("# "),
            l2,
            l3,
            *l4,
            dp.Text("# "),
            dp.Text("# "),
            label="Descriptive Statistics",
        )
    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "descriptive_statistics.html", open=True
        )
    return report


def quality_check(
    master_path,
    QC_tabs,
    avl_recs_QC,
    missing_recs_QC,
    all_charts_num_3_,
    print_report=False,
):
    """
    This function helps to produce output specific to the Quality Checker Tab.
    Parameters
    ----------
    master_path
        Path containing the input files.
    QC_tabs
        nullColumns_detection','IDness_detection','biasedness_detection','invalidEntries_detection','duplicate_detection','nullRows_detection','outlier_detection'
    avl_recs_QC
        Available files from the QC_tabs (Quality Checker tabs)
    missing_recs_QC
        Missing files from the QC_tabs (Quality Checker tabs)
    all_charts_num_3_
        Numerical charts (outlier charts) all collated in a list format supported as per datapane objects
    print_report
        Printing option flexibility. Default value is kept as False.
    Returns
    -------
    DatapaneObject / Output[HTML]
    """
    c_ = []
    r_ = []
    if len(missing_recs_QC) == len(QC_tabs):
        return "null_report"
    else:
        row_wise = ["duplicate_detection", "nullRows_detection"]
        col_wise = [
            "nullColumns_detection",
            "IDness_detection",
            "biasedness_detection",
            "invalidEntries_detection",
            "outlier_detection",
        ]
        row_wise_ = [p for p in row_wise if p in avl_recs_QC]
        col_wise_ = [p for p in col_wise if p in avl_recs_QC]
        len_row_wise = len([p for p in row_wise if p in avl_recs_QC])
        len_col_wise = len([p for p in col_wise if p in avl_recs_QC])
        if len_row_wise == 0:
            c = data_analyzer_output(master_path, col_wise_, "quality_checker")
            for i in c:
                for j in i:
                    if j == "outlier_charts_placeholder" and len(all_charts_num_3_) > 1:
                        c_.append(
                            dp.Select(
                                blocks=all_charts_num_3_, type=dp.SelectType.DROPDOWN
                            )
                        )
                    elif (
                        j == "outlier_charts_placeholder"
                        and len(all_charts_num_3_) == 0
                    ):
                        c_.append(dp.Plot(blank_chart))
                    else:
                        c_.append(j)
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    "*This section identifies the data quality issues at both row and column level.*"
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Group(*c_),
                dp.Text("# "),
                dp.Text("# "),
                label="Quality Check",
            )
        elif len_col_wise == 0:
            r = data_analyzer_output(master_path, row_wise_, "quality_checker")
            for i in r:
                for j in i:
                    r_.append(j)
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    "*This section identifies the data quality issues at both row and column level.*"
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Group(*r_),
                dp.Text("# "),
                dp.Text("# "),
                label="Quality Check",
            )
        else:
            c = data_analyzer_output(master_path, col_wise_, "quality_checker")
            for i in c:
                for j in i:
                    if j == "outlier_charts_placeholder" and len(all_charts_num_3_) > 1:
                        c_.append(
                            dp.Select(
                                blocks=all_charts_num_3_, type=dp.SelectType.DROPDOWN
                            )
                        )
                    elif (
                        j == "outlier_charts_placeholder"
                        and len(all_charts_num_3_) == 0
                    ):
                        c_.append(dp.Plot(blank_chart))
                    else:
                        c_.append(j)
            r = data_analyzer_output(master_path, row_wise_, "quality_checker")
            for i in r:
                for j in i:
                    r_.append(j)
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    "*This section identifies the data quality issues at both row and column level.*"
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Select(
                    blocks=[
                        dp.Group(dp.Text("# "), dp.Group(*c_), label="Column Level"),
                        dp.Group(dp.Text("# "), dp.Group(*r_), label="Row Level"),
                    ],
                    type=dp.SelectType.TABS,
                ),
                dp.Text("# "),
                dp.Text("# "),
                label="Quality Check",
            )
    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "quality_check.html", open=True
        )
    return report


def attribute_associations(
    master_path,
    AE_tabs,
    avl_recs_AE,
    missing_recs_AE,
    label_col,
    all_charts_num_2_,
    all_charts_cat_2_,
    print_report=False,
):
    """
    This function helps to produce output specific to the Attribute Association Tab.
    Parameters
    ----------
    master_path
        Path containing the input files.
    AE_tabs
        correlation_matrix','IV_calculation','IG_calculation','variable_clustering'
    avl_recs_AE
        Available files from the AE_tabs (Association Evaluator tabs)
    missing_recs_AE
        Missing files from the AE_tabs (Association Evaluator tabs)
    label_col
        label column
    all_charts_num_2_
        Numerical charts (histogram) all collated in a list format supported as per datapane objects
    all_charts_cat_2_
        Categorical charts (barplot) all collated in a list format supported as per datapane objects
    print_report
        Printing option flexibility. Default value is kept as False.
    Returns
    -------
    DatapaneObject / Output[HTML]
    """
    if (len(missing_recs_AE) == len(AE_tabs)) and (
        (len(all_charts_num_2_) + len(all_charts_cat_2_)) == 0
    ):
        return "null_report"
    else:
        if len(all_charts_num_2_) == 0 and len(all_charts_cat_2_) == 0:
            target_association_rep = dp.Text("##")
        else:
            if len(all_charts_num_2_) > 0 and len(all_charts_cat_2_) == 0:
                target_association_rep = dp.Group(
                    dp.Text("### Attribute to Target Association"),
                    dp.Text(
                        """
                        *Bivariate Distribution considering the event captured across different
                        attribute splits (or categories)*
                        """
                    ),
                    dp.Select(blocks=all_charts_num_2_, type=dp.SelectType.DROPDOWN),
                    label="Numerical",
                )
            elif len(all_charts_num_2_) == 0 and len(all_charts_cat_2_) > 0:
                target_association_rep = dp.Group(
                    dp.Text("### Attribute to Target Association"),
                    dp.Text(
                        """
                        *Bivariate Distribution considering the event captured across different
                        attribute splits (or categories)*
                        """
                    ),
                    dp.Select(blocks=all_charts_cat_2_, type=dp.SelectType.DROPDOWN),
                    label="Categorical",
                )
            else:
                target_association_rep = dp.Group(
                    dp.Text("### Attribute to Target Association"),
                    dp.Select(
                        blocks=[
                            dp.Group(
                                dp.Select(
                                    blocks=all_charts_num_2_,
                                    type=dp.SelectType.DROPDOWN,
                                ),
                                label="Numerical",
                            ),
                            dp.Group(
                                dp.Select(
                                    blocks=all_charts_cat_2_,
                                    type=dp.SelectType.DROPDOWN,
                                ),
                                label="Categorical",
                            ),
                        ],
                        type=dp.SelectType.TABS,
                    ),
                    dp.Text(
                        """
                        *Event Rate is defined as % of event label (i.e. label 1) in a bin or a categorical
                        value of an attribute.*
                        """
                    ),
                    dp.Text("# "),
                )
    if len(missing_recs_AE) == len(AE_tabs):
        report = dp.Group(
            dp.Text("# "),
            dp.Text(
                """
                *This section analyzes the interaction between different attributes and/or the relationship between
                an attribute & the binary target variable.*
                """
            ),
            dp.Text("## "),
            target_association_rep,
            dp.Text("## "),
            dp.Text("## "),
            label="Attribute Associations",
        )
    else:
        report = dp.Group(
            dp.Text("# "),
            dp.Text(
                """
                *This section analyzes the interaction between different attributes and/or the relationship between
                 an attribute & the binary target variable.*
                 """
            ),
            dp.Text("# "),
            dp.Text("# "),
            dp.Text("### Association Matrix & Plot"),
            dp.Select(
                blocks=data_analyzer_output(
                    master_path, avl_recs_AE, tab_name="association_evaluator"
                ),
                type=dp.SelectType.DROPDOWN,
            ),
            dp.Text("### "),
            dp.Text("## "),
            target_association_rep,
            dp.Text("## "),
            dp.Text("## "),
            label="Attribute Associations",
        )
    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "attribute_associations.html", open=True
        )
    return report


def data_drift_stability(
    master_path,
    ds_ind,
    id_col,
    drift_threshold_model,
    all_drift_charts_,
    print_report=False,
):
    """
    This function helps to produce output specific to the Data Drift & Stability Tab.
    Parameters
    ----------
    master_path
        Path containing the input files.
    ds_ind
        Drift stability indicator in list form.
    id_col
        ID column
    drift_threshold_model
        threshold which the user is specifying for tagging an attribute to be drifted or not
    all_drift_charts_
        Charts (histogram/barplot) all collated in a list format supported as per datapane objects
    print_report
        Printing option flexibility. Default value is kept as False.
    Returns
    -------
    DatapaneObject / Output[HTML]
    """
    line_chart_list = []
    if ds_ind[0] > 0:
        fig_metric_drift = go.Figure()
        fig_metric_drift.add_trace(
            go.Scatter(
                x=list(drift_df[drift_df.flagged.values == 1][metric_drift[0]].values),
                y=list(drift_df[drift_df.flagged.values == 1].attribute.values),
                marker=dict(color=global_theme[1], size=14),
                mode="markers",
                name=metric_drift[0],
            )
        )
        fig_metric_drift.add_trace(
            go.Scatter(
                x=list(drift_df[drift_df.flagged.values == 1][metric_drift[1]].values),
                y=list(drift_df[drift_df.flagged.values == 1].attribute.values),
                marker=dict(color=global_theme[3], size=14),
                mode="markers",
                name=metric_drift[1],
            )
        )
        fig_metric_drift.add_trace(
            go.Scatter(
                x=list(drift_df[drift_df.flagged.values == 1][metric_drift[2]].values),
                y=list(drift_df[drift_df.flagged.values == 1].attribute.values),
                marker=dict(color=global_theme[5], size=14),
                mode="markers",
                name=metric_drift[2],
            )
        )
        fig_metric_drift.add_trace(
            go.Scatter(
                x=list(drift_df[drift_df.flagged.values == 1][metric_drift[3]].values),
                y=list(drift_df[drift_df.flagged.values == 1].attribute.values),
                marker=dict(color=global_theme[7], size=14),
                mode="markers",
                name=metric_drift[3],
            )
        )
        fig_metric_drift.add_vrect(
            x0=0,
            x1=drift_threshold_model,
            fillcolor=global_theme[7],
            opacity=0.1,
            layer="below",
            line_width=1,
        ),
        fig_metric_drift.update_layout(
            legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center")
        )
        fig_metric_drift.layout.plot_bgcolor = global_plot_bg_color
        fig_metric_drift.layout.paper_bgcolor = global_paper_bg_color
        fig_metric_drift.update_xaxes(
            showline=True, linewidth=2, gridcolor=px.colors.sequential.Greys[1]
        )
        fig_metric_drift.update_yaxes(
            showline=True, linewidth=2, gridcolor=px.colors.sequential.Greys[2]
        )
        #     Drift Chart - 2
        fig_gauge_drift = go.Figure(
            go.Indicator(
                domain={"x": [0, 1], "y": [0, 1]},
                value=drifted_feats,
                mode="gauge+number",
                title={"text": ""},
                gauge={
                    "axis": {"range": [None, len_feats]},
                    "bar": {"color": px.colors.sequential.Reds[7]},
                    "steps": [
                        {
                            "range": [0, drifted_feats],
                            "color": px.colors.sequential.Reds[8],
                        },
                        {
                            "range": [drifted_feats, len_feats],
                            "color": px.colors.sequential.Greens[8],
                        },
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 1,
                        "value": len_feats,
                    },
                },
            )
        )
        fig_gauge_drift.update_layout(font={"color": "black", "family": "Arial"})

        def drift_text_gen(drifted_feats, len_feats):
            """
            Parameters
            ----------
            drifted_feats
                count of attributes drifted
            len_feats
                count of attributes passed for analysis
            Returns
            -------
            String
            """
            if drifted_feats == 0:
                text = """
                *Drift barometer does not indicate any drift in the underlying data. Please refer to the metric
                values as displayed in the above table & comparison plot for better understanding*
                """
            elif drifted_feats == 1:
                text = (
                    "*Drift barometer indicates that "
                    + str(drifted_feats)
                    + " out of "
                    + str(len_feats)
                    + " ("
                    + str(np.round((100 * drifted_feats / len_feats), 2))
                    + "%) attributes has been drifted from its source behaviour.*"
                )
            elif drifted_feats > 1:
                text = (
                    "*Drift barometer indicates that "
                    + str(drifted_feats)
                    + " out of "
                    + str(len_feats)
                    + " ("
                    + str(np.round((100 * drifted_feats / len_feats), 2))
                    + "%) attributes have been drifted from its source behaviour.*"
                )
            else:
                text = ""
            return text

    else:
        pass
    if ds_ind[0] == 0 and ds_ind[1] == 0:
        return "null_report"
    elif ds_ind[0] == 0 and ds_ind[1] > 0.5:
        for i in total_unstable_attr:
            if len(total_unstable_attr) > 1:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
            else:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
                line_chart_list.append(dp.Plot(blank_chart, label=" "))

        report = dp.Group(
            dp.Text("# "),
            dp.Text(
                """
                *This section examines the dataset stability wrt the baseline dataset (via computing drift
                statistics) and/or wrt the historical datasets (via computing stability index).*
                """
            ),
            dp.Text("# "),
            dp.Text("# "),
            dp.Text("### Data Stability Analysis"),
            dp.DataTable(df_si),
            dp.Select(blocks=line_chart_list, type=dp.SelectType.DROPDOWN),
            dp.Group(
                dp.Text("**Stability Index Interpretation:**"),
                dp.Plot(plot_index_stability),
            ),
            label="Drift & Stability",
        )
    elif ds_ind[0] == 1 and ds_ind[1] == 0:
        if len(all_drift_charts_) > 0:
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    """
                    *This section examines the dataset stability wrt the baseline dataset (via computing drift
                    statistics) and/or wrt the historical datasets (via computing stability index).*
                    """
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Data Drift Analysis"),
                dp.DataTable(drift_df),
                dp.Text(
                    "*An attribute is flagged as drifted if any drift metric is found to be above the threshold of "
                    + str(drift_threshold_model)
                    + ".*"
                ),
                dp.Text("##"),
                dp.Select(blocks=all_drift_charts_, type=dp.SelectType.DROPDOWN),
                dp.Text(
                    """
                    *Source & Target datasets were compared to see the % deviation at decile level for numerical
                    attributes and at individual category level for categorical attributes*
                    """
                ),
                dp.Text("###  "),
                dp.Text("###  "),
                dp.Text("### Data Health"),
                dp.Group(
                    dp.Plot(fig_metric_drift), dp.Plot(fig_gauge_drift), columns=2
                ),
                dp.Group(
                    dp.Text(
                        "*Representation of attributes across different computed Drift Metrics*"
                    ),
                    dp.Text(drift_text_gen(drifted_feats, len_feats)),
                    columns=2,
                ),
                label="Drift & Stability",
            )
        else:
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    """
                    *This section examines the dataset stability wrt the baseline dataset (via computing drift
                    statistics) and/or wrt the historical datasets (via computing stability index).*
                    """
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Data Drift Analysis"),
                dp.DataTable(drift_df),
                dp.Text(
                    "*An attribute is flagged as drifted if any drift metric is found to be above the threshold of "
                    + str(drift_threshold_model)
                    + ".*"
                ),
                dp.Text("##"),
                dp.Text("###  "),
                dp.Text("### Data Health"),
                dp.Group(
                    dp.Plot(fig_metric_drift), dp.Plot(fig_gauge_drift), columns=2
                ),
                dp.Group(
                    dp.Text(
                        "*Representation of attributes across different computed Drift Metrics*"
                    ),
                    dp.Text(drift_text_gen(drifted_feats, len_feats)),
                    columns=2,
                ),
                label="Drift & Stability",
            )
    elif ds_ind[0] == 1 and ds_ind[1] >= 0.5:
        for i in total_unstable_attr:
            if len(total_unstable_attr) > 1:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
            else:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
                line_chart_list.append(dp.Plot(blank_chart, label=" "))
        if len(all_drift_charts_) > 0:
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    """
                    *This section examines the dataset stability wrt the baseline dataset (via computing drift
                    statistics) and/or wrt the historical datasets (via computing stability index).*
                    """
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Data Drift Analysis"),
                dp.DataTable(drift_df),
                dp.Text(
                    "*An attribute is flagged as drifted if any drift metric is found to be above the threshold of "
                    + str(drift_threshold_model)
                    + ".*"
                ),
                dp.Text("##"),
                dp.Select(blocks=all_drift_charts_, type=dp.SelectType.DROPDOWN),
                dp.Text(
                    """
                    *Source & Target datasets were compared to see the % deviation at decile level for numerical
                    attributes and at individual category level for categorical attributes*
                    """
                ),
                dp.Text("###  "),
                dp.Text("###  "),
                dp.Text("### Data Health"),
                dp.Group(
                    dp.Plot(fig_metric_drift), dp.Plot(fig_gauge_drift), columns=2
                ),
                dp.Group(
                    dp.Text(
                        "*Representation of attributes across different computed Drift Metrics*"
                    ),
                    dp.Text(drift_text_gen(drifted_feats, len_feats)),
                    columns=2,
                ),
                dp.Text("## "),
                dp.Text("## "),
                dp.Text("### Data Stability Analysis"),
                dp.DataTable(df_si),
                dp.Select(blocks=line_chart_list, type=dp.SelectType.DROPDOWN),
                dp.Group(
                    dp.Text("**Stability Index Interpretation:**"),
                    dp.Plot(plot_index_stability),
                ),
                label="Drift & Stability",
            )
        else:
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    """
                    *This section examines the dataset stability wrt the baseline dataset (via computing drift
                    statistics) and/or wrt the historical datasets (via computing stability index).*
                    """
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Data Drift Analysis"),
                dp.DataTable(drift_df),
                dp.Text(
                    "*An attribute is flagged as drifted if any drift metric is found to be above the threshold of "
                    + str(drift_threshold_model)
                    + ".*"
                ),
                dp.Text("##"),
                dp.Text("### Data Health"),
                dp.Group(
                    dp.Plot(fig_metric_drift), dp.Plot(fig_gauge_drift), columns=2
                ),
                dp.Group(
                    dp.Text(
                        "*Representation of attributes across different computed Drift Metrics*"
                    ),
                    dp.Text(drift_text_gen(drifted_feats, len_feats)),
                    columns=2,
                ),
                dp.Text("## "),
                dp.Text("## "),
                dp.Text("### Data Stability Analysis"),
                dp.DataTable(df_si),
                dp.Select(blocks=line_chart_list, type=dp.SelectType.DROPDOWN),
                dp.Group(
                    dp.Text("**Stability Index Interpretation:**"),
                    dp.Plot(plot_index_stability),
                ),
                label="Drift & Stability",
            )
    elif ds_ind[0] == 0 and ds_ind[1] >= 0.5:
        for i in total_unstable_attr:
            if len(total_unstable_attr) > 1:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
            else:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
                line_chart_list.append(dp.Plot(blank_chart, label=" "))
        report = dp.Group(
            dp.Text("# "),
            dp.Text(
                """
                *This section examines the dataset stability wrt the baseline dataset (via computing drift statistics)
                and/or wrt the historical datasets (via computing stability index).*
                """
            ),
            dp.Text("# "),
            dp.Text("# "),
            dp.Text("### Data Stability Analysis"),
            dp.DataTable(df_si),
            dp.Select(blocks=line_chart_list, type=dp.SelectType.DROPDOWN),
            dp.Group(
                dp.Text("**Stability Index Interpretation:**"),
                dp.Plot(plot_index_stability),
            ),
            label="Drift & Stability",
        )
    else:
        for i in total_unstable_attr:
            if len(total_unstable_attr) > 1:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
            else:
                line_chart_list.append(
                    line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
                )
                line_chart_list.append(dp.Plot(blank_chart, label=" "))
        if len(all_drift_charts_) > 0:
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    """
                    *This section examines the dataset stability wrt the baseline dataset (via computing drift
                    statistics) and/or wrt the historical datasets (via computing stability index).*
                    """
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Data Drift Analysis"),
                dp.DataTable(drift_df),
                dp.Text(
                    "*An attribute is flagged as drifted if any drift metric is found to be above the threshold of "
                    + str(drift_threshold_model)
                    + ".*"
                ),
                dp.Text("##"),
                dp.Select(blocks=all_drift_charts_, type=dp.SelectType.DROPDOWN),
                dp.Text(
                    """
                    *Source & Target datasets were compared to see the % deviation at decile level for numerical
                    attributes and at individual category level for categorical attributes*
                    """
                ),
                dp.Text("###  "),
                dp.Text("###  "),
                dp.Text("### Data Health"),
                dp.Group(
                    dp.Plot(fig_metric_drift), dp.Plot(fig_gauge_drift), columns=2
                ),
                dp.Group(
                    dp.Text(
                        "*Representation of attributes across different computed Drift Metrics*"
                    ),
                    dp.Text(drift_text_gen(drifted_feats, len_feats)),
                    columns=2,
                ),
                dp.Text("## "),
                dp.Text("## "),
                dp.Text("### Data Stability Analysis"),
                dp.DataTable(df_si),
                dp.Select(blocks=line_chart_list, type=dp.SelectType.DROPDOWN),
                dp.Group(
                    dp.Text("**Stability Index Interpretation:**"),
                    dp.Plot(plot_index_stability),
                ),
                label="Drift & Stability",
            )
        else:
            report = dp.Group(
                dp.Text("# "),
                dp.Text(
                    """
                    *This section examines the dataset stability wrt the baseline dataset (via computing drift
                    statistics) and/or wrt the historical datasets (via computing stability index).*
                    """
                ),
                dp.Text("# "),
                dp.Text("# "),
                dp.Text("### Data Drift Analysis"),
                dp.DataTable(drift_df),
                dp.Text(
                    "*An attribute is flagged as drifted if any drift metric is found to be above the threshold of "
                    + str(drift_threshold_model)
                    + ".*"
                ),
                dp.Text("##"),
                dp.Text("### Data Health"),
                dp.Group(
                    dp.Plot(fig_metric_drift), dp.Plot(fig_gauge_drift), columns=2
                ),
                dp.Group(
                    dp.Text(
                        "*Representation of attributes across different computed Drift Metrics*"
                    ),
                    dp.Text(drift_text_gen(drifted_feats, len_feats)),
                    columns=2,
                ),
                dp.Text("## "),
                dp.Text("## "),
                dp.Text("### Data Stability Analysis"),
                dp.DataTable(df_si),
                dp.Select(blocks=line_chart_list, type=dp.SelectType.DROPDOWN),
                dp.Group(
                    dp.Text("**Stability Index Interpretation:**"),
                    dp.Plot(plot_index_stability),
                ),
                label="Drift & Stability",
            )
    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "data_drift_stability.html", open=True
        )
    return report


def plotSeasonalDecompose(
    base_path, x_col, y_col, metric_col="median", title="Seasonal Decomposition"
):

    """
    This function helps to produce output specific to the Seasonal Decomposition of Time Series. Ideally it's expected to source a data containing atleast 2 cycles or 24 months as the most.

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    x_col
        Timestamp / date column name
    y_col
        Numerical column names
    metric_col
        Metric of aggregation. Options can be between "Median", "Mean", "Min", "Max"
    title
        "Title Description"
    Returns
    -------
    Plot
    """
    df = pd.read_csv(ends_with(base_path) + x_col + "_" + y_col + "_daily.csv").dropna()

    df[x_col] = pd.to_datetime(df[x_col], format="%Y-%m-%d %H:%M:%S.%f")
    df = df.set_index(x_col)

    if len([x for x in df.columns if "min" in x]) == 0:

        #         result = seasonal_decompose(df[metric_col],model="additive")
        pass

    else:

        result = seasonal_decompose(df[metric_col], model="additive", period=12)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        #         fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=result.observed,
                name="Observed",
                mode="lines+markers",
                line=dict(color=global_theme[0]),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=result.trend,
                name="Trend",
                mode="lines+markers",
                line=dict(color=global_theme[2]),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=result.seasonal,
                name="Seasonal",
                mode="lines+markers",
                line=dict(color=global_theme[4]),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=result.resid,
                name="Residuals",
                mode="lines+markers",
                line=dict(color=global_theme[6]),
            ),
            row=2,
            col=2,
        )

        #         fig.add_trace(go.Scatter(x=df.index, y=result.observed, name ="Observed", mode='lines+markers',line=dict(color=global_theme[0])))

        #         fig.add_trace(go.Scatter(x=df.index, y=result.trend, name ="Trend", mode='lines+markers',line=dict(color=global_theme[2])))

        #         fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, name ="Seasonal", mode='lines+markers',line=dict(color=global_theme[4])))

        #         fig.add_trace(go.Scatter(x=df.index, y=result.resid, name ="Residuals", mode='lines+markers',line=dict(color=global_theme[6])))

        fig.layout.plot_bgcolor = global_plot_bg_color
        fig.layout.paper_bgcolor = global_paper_bg_color
        fig.update_xaxes(gridcolor=px.colors.sequential.Greys[1])
        fig.update_yaxes(gridcolor=px.colors.sequential.Greys[1])
        fig.update_layout(autosize=True, width=2000, height=800)
        fig.update_layout(
            legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center")
        )

        return fig


def gen_time_series_plots(base_path, x_col, y_col, time_cat):

    """

    This function helps to produce Time Series Plots by sourcing the aggregated data as Daily/Hourly/Weekly level.

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    x_col
        Timestamp / date column name
    y_col
        Numerical column names
    time_cat
        Time category of analysis which can be between "Daily", "Hourly", "Weekly"

    Returns
    -------
    Plot

    """

    df = pd.read_csv(
        ends_with(base_path) + x_col + "_" + y_col + "_" + time_cat + ".csv"
    ).dropna()

    if len([x for x in df.columns if "min" in x]) == 0:

        if time_cat == "daily":

            # x_col = x_col + "_ts"

            fig = px.line(
                df,
                x=x_col,
                y="count",
                color=y_col,
                color_discrete_sequence=global_theme,
            )

            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list(
                            [
                                dict(
                                    count=1,
                                    label="1m",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=3,
                                    label="3m",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=6,
                                    label="6m",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=1, label="YTD", step="year", stepmode="todate"
                                ),
                                dict(
                                    count=1,
                                    label="1y",
                                    step="year",
                                    stepmode="backward",
                                ),
                                dict(step="all"),
                            ]
                        )
                    ),
                    rangeslider=dict(visible=True),
                    type="date",
                )
            )

        elif time_cat == "weekly":

            fig = px.bar(
                df,
                x="dow",
                y="count",
                color=y_col,
                color_discrete_sequence=global_theme,
            )
        #             fig.update_layout(barmode='stack')

        elif time_cat == "hourly":

            fig = px.bar(
                df,
                x="daypart_cat",
                y="count",
                color=y_col,
                color_discrete_sequence=global_theme,
            )
        #             fig.update_layout(barmode='stack')

        else:
            pass

    else:

        if time_cat == "daily":

            # x_col = x_col + "_ts"
            f1 = go.Scatter(
                x=list(df[x_col]),
                y=list(df["min"]),
                name="Min",
                line=dict(color=global_theme[6]),
            )
            f2 = go.Scatter(
                x=list(df[x_col]),
                y=list(df["max"]),
                name="Max",
                line=dict(color=global_theme[4]),
            )
            f3 = go.Scatter(
                x=list(df[x_col]),
                y=list(df["mean"]),
                name="Mean",
                line=dict(color=global_theme[2]),
            )
            f4 = go.Scatter(
                x=list(df[x_col]),
                y=list(df["median"]),
                name="Median",
                line=dict(color=global_theme[0]),
            )

            fig = go.Figure(data=[f1, f2, f3, f4])

            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list(
                            [
                                dict(
                                    count=1,
                                    label="1m",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=3,
                                    label="3m",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=6,
                                    label="6m",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=1, label="YTD", step="year", stepmode="todate"
                                ),
                                dict(
                                    count=1,
                                    label="1y",
                                    step="year",
                                    stepmode="backward",
                                ),
                                dict(step="all"),
                            ]
                        )
                    ),
                    rangeslider=dict(visible=True),
                    type="date",
                )
            )

        elif time_cat == "weekly":

            f1 = go.Bar(
                x=list(df["dow"]),
                y=list(df["min"]),
                marker_color=global_theme[6],
                name="Min",
            )
            f2 = go.Bar(
                x=list(df["dow"]),
                y=list(df["max"]),
                marker_color=global_theme[4],
                name="Max",
            )
            f3 = go.Bar(
                x=list(df["dow"]),
                y=list(df["mean"]),
                marker_color=global_theme[2],
                name="Mean",
            )
            f4 = go.Bar(
                x=list(df["dow"]),
                y=list(df["median"]),
                marker_color=global_theme[0],
                name="Median",
            )

            fig = go.Figure(data=[f1, f2, f3, f4])
            fig.update_layout(barmode="group")

        elif time_cat == "hourly":

            f1 = go.Bar(
                x=list(df["daypart_cat"]),
                y=list(df["min"]),
                marker_color=global_theme[6],
                name="Min",
            )
            f2 = go.Bar(
                x=list(df["daypart_cat"]),
                y=list(df["max"]),
                marker_color=global_theme[4],
                name="Max",
            )
            f3 = go.Bar(
                x=list(df["daypart_cat"]),
                y=list(df["mean"]),
                marker_color=global_theme[2],
                name="Mean",
            )
            f4 = go.Bar(
                x=list(df["daypart_cat"]),
                y=list(df["median"]),
                marker_color=global_theme[0],
                name="Median",
            )

            fig = go.Figure(data=[f1, f2, f3, f4])
            fig.update_layout(barmode="group")

        else:
            pass

    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    fig.update_xaxes(gridcolor=px.colors.sequential.Greys[1])
    fig.update_yaxes(gridcolor=px.colors.sequential.Greys[1])
    fig.update_layout(
        legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center")
    )

    return fig


def list_ts_remove_append(l, opt):

    """

    This function helps to remove or append "_ts" from any list.


    Parameters
    ----------
    l
        List containing column name
    opt
        Option to choose between 1 & Others to enable the functionality of removing or appending "_ts" within the elements of a list

    Returns
    -------
    List

    """

    ll = []
    if opt == 1:
        for i in l:
            if i[-3:] == "_ts":
                ll.append(i[0:-3:])
            else:
                ll.append(i)
        return ll
    else:
        for i in l:
            if i[-3:] == "_ts":
                ll.append(i)
            else:
                ll.append(i + "_ts")
        return ll


def ts_viz_1_1(base_path, x_col, y_col, output_type):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    x_col
        Timestamp / date column name
    y_col
        Numerical column names
    output_type
        Time category of analysis which can be between "Daily", "Hourly", "Weekly"

    Returns
    -------
    Plot
    """

    ts_fig = gen_time_series_plots(base_path, x_col, y_col, output_type)

    return ts_fig


def ts_viz_1_2(base_path, ts_col, col_list, output_type):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    col_list
        Numerical / Categorical column names
    output_type
        Time category of analysis which can be between "Daily", "Hourly", "Weekly"

    Returns
    -------
    DatapaneObject
    """

    bl = []

    for i in col_list:
        if len(col_list) > 1:
            bl.append(dp.Group(ts_viz_1_1(base_path, ts_col, i, output_type), label=i))
        else:
            bl.append(dp.Group(ts_viz_1_1(base_path, ts_col, i, output_type), label=i))
            bl.append(dp.Plot(blank_chart, label="_"))

    return dp.Select(blocks=bl, type=dp.SelectType.DROPDOWN)


def ts_viz_1_3(base_path, ts_col, num_cols, cat_cols, output_type):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    num_cols
        Numerical column names
    cat_cols
        Categorical column names
    output_type
        Time category of analysis which can be between "Daily", "Hourly", "Weekly"

    Returns
    -------
    DatapaneObject
    """

    ts_v = []
    # print(num_cols)
    # print(cat_cols)
    if len(num_cols) == 0:
        for i in ts_col:
            if len(ts_col) > 1:
                ts_v.append(
                    dp.Group(ts_viz_1_2(base_path, i, cat_cols, output_type), label=i)
                )
            else:
                ts_v.append(
                    dp.Group(ts_viz_1_2(base_path, i, cat_cols, output_type), label=i)
                )
                ts_v.append(dp.Plot(blank_chart, label="_"))

    elif len(cat_cols) == 0:
        for i in ts_col:
            if len(ts_col) > 1:
                ts_v.append(
                    dp.Group(ts_viz_1_2(base_path, i, num_cols, output_type), label=i)
                )
            else:
                ts_v.append(
                    dp.Group(ts_viz_1_2(base_path, i, num_cols, output_type), label=i)
                )
                ts_v.append(dp.Plot(blank_chart, label="_"))

    elif (len(num_cols) >= 1) & (len(cat_cols) >= 1):

        for i in ts_col:
            if len(ts_col) > 1:
                ts_v.append(
                    dp.Group(
                        dp.Select(
                            blocks=[
                                dp.Group(
                                    ts_viz_1_2(base_path, i, num_cols, output_type),
                                    label="Numerical",
                                ),
                                dp.Group(
                                    ts_viz_1_2(base_path, i, cat_cols, output_type),
                                    label="Categorical",
                                ),
                            ],
                            type=dp.SelectType.TABS,
                        ),
                        label=i,
                    )
                )
            else:
                ts_v.append(
                    dp.Group(
                        dp.Select(
                            blocks=[
                                dp.Group(
                                    ts_viz_1_2(base_path, i, num_cols, output_type),
                                    label="Numerical",
                                ),
                                dp.Group(
                                    ts_viz_1_2(base_path, i, cat_cols, output_type),
                                    label="Categorical",
                                ),
                            ],
                            type=dp.SelectType.TABS,
                        ),
                        label=i,
                    )
                )
                ts_v.append(dp.Plot(blank_chart, label="_"))

    return dp.Select(blocks=ts_v, type=dp.SelectType.DROPDOWN)


def ts_viz_2_1(base_path, x_col, y_col):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    x_col
        Timestamp / date column name
    y_col
        Numerical column names

    Returns
    -------
    DatapaneObject
    """

    ts_fig = []

    for i in ["mean", "median", "min", "max"]:

        ts_fig.append(
            dp.Plot(
                plotSeasonalDecompose(base_path, x_col, y_col, metric_col=i),
                label=i.title(),
            )
        )

    return dp.Select(blocks=ts_fig, type=dp.SelectType.TABS)


def ts_viz_2_2(base_path, ts_col, col_list):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    col_list
        Numerical column names

    Returns
    -------
    DatapaneObject
    """

    bl = []

    for i in col_list:
        if len(col_list) > 1:
            bl.append(dp.Group(ts_viz_2_1(base_path, ts_col, i), label=i))
        else:
            bl.append(dp.Group(ts_viz_2_1(base_path, ts_col, i), label=i))
            bl.append(dp.Group(dp.Plot(blank_chart, label=" "), label=" "))

    return dp.Select(blocks=bl, type=dp.SelectType.DROPDOWN)


def ts_viz_2_3(base_path, ts_col, num_cols):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    num_cols
        Numerical column names

    Returns
    -------
    DatapaneObject
    """

    ts_v = []

    if len(ts_col) > 1:

        for i in ts_col:

            f = list(
                pd.read_csv(
                    ends_with(base_path) + "stats_" + i + "_2.csv"
                ).count_unique_dates.values
            )[0]

            if f >= 24:

                ts_v.append(dp.Group(ts_viz_2_2(base_path, i, num_cols), label=i))
            else:
                ts_v.append(
                    dp.Group(
                        dp.Text(
                            "The plots couldn't be displayed as x must have 2 complete cycles requires 24 observations. x only has "
                            + str(f)
                            + " observation(s)"
                        ),
                        label=i,
                    )
                )

    else:

        for i in ts_col:

            f = list(
                pd.read_csv(
                    ends_with(base_path) + "stats_" + i + "_2.csv"
                ).count_unique_dates.values
            )[0]

            if f >= 24:

                ts_v.append(dp.Group(ts_viz_2_2(base_path, i, num_cols), label=i))
                ts_v.append(dp.Plot(blank_chart, label="_"))

            else:

                ts_v.append(
                    dp.Group(
                        dp.Text(
                            "The plots couldn't be displayed as x must have 2 complete cycles requires 24 observations. x only has "
                            + str(f)
                            + " observation(s)"
                        ),
                        label=i,
                    )
                )
                ts_v.append(dp.Plot(blank_chart, label="_"))

    return dp.Select(blocks=ts_v, type=dp.SelectType.DROPDOWN)


def ts_landscape(base_path, ts_cols, id_col):

    """

    This function helps to produce a basic landscaping view of the data by picking up the base path for reading the aggregated data and specified by the timestamp / date column & the ID column.

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    id_col
        ID Column

    Returns
    -------
    DatapaneObject
    """

    if ts_cols is None:

        return dp.Text("#")
    else:
        df_stats_ts = []
        for i in ts_cols:
            if len(ts_cols) > 1:
                df_stats_ts.append(
                    dp.Group(
                        dp.Group(
                            dp.Text("#   "),
                            dp.Text("*ID considered here is : " + str(id_col) + "*"),
                            dp.Text("#   "),
                            dp.Text("#### Consistency Analysis Of Dates"),
                            dp.DataTable(
                                pd.read_csv(
                                    ends_with(base_path) + "stats_" + i + "_1.csv"
                                )
                                .set_index("attribute")
                                .T,
                                label=i,
                            ),
                        ),
                        dp.Group(
                            dp.Text(
                                "*The Percentile distribution across different bins of ID-Date / Date-ID combination should be in a considerable range to determine the regularity of Time series. In an ideal scenario the proportion of dates within each ID should be same. Also, the count of IDs across unique dates should be consistent for a balanced distribution*"
                            ),
                            dp.Text("#   "),
                            dp.Text("#### Vital Statistics"),
                            dp.DataTable(
                                pd.read_csv(
                                    ends_with(base_path) + "stats_" + i + "_2.csv"
                                ).T.rename(columns={0: ""}),
                                label=i,
                            ),
                        ),
                        label=i,
                    )
                )

            else:
                df_stats_ts.append(
                    dp.Group(
                        dp.Group(
                            dp.Text("#   "),
                            dp.Text("*ID considered here is : " + str(id_col) + "*"),
                            dp.Text("#### Consistency Analysis Of Dates"),
                            dp.Text("#   "),
                            dp.DataTable(
                                pd.read_csv(
                                    ends_with(base_path) + "stats_" + i + "_1.csv"
                                )
                                .set_index("attribute")
                                .T,
                                label=i,
                            ),
                        ),
                        dp.Group(
                            dp.Text("#   "),
                            dp.Text("#### Vital Statistics"),
                            dp.DataTable(
                                pd.read_csv(
                                    ends_with(base_path) + "stats_" + i + "_2.csv"
                                ).T.rename(columns={0: ""}),
                                label=i,
                            ),
                        ),
                        label=i,
                    )
                )
                df_stats_ts.append(dp.Plot(blank_chart, label="_"))

        return dp.Group(
            dp.Text("### Time Stamp Data Diagnosis"),
            dp.Select(blocks=df_stats_ts, type=dp.SelectType.DROPDOWN),
        )


def lambda_cat(val):

    """

    Parameters
    ----------

    val
        Value of Box Cox Test which translates into the transformation to be applied.

    Returns
    -------
    String
    """

    if val < -1:
        return "Reciprocal Square Transform"
    elif val >= -1 and val < -0.5:
        return "Reciprocal Transform"
    elif val >= -0.5 and val < 0:
        return "Receiprocal Square Root Transform"
    elif val >= 0 and val < 0.5:
        return "Log Transform"
    elif val >= 0.5 and val < 1:
        return "Square Root Transform"
    elif val >= 1 and val < 2:
        return "No Transform"
    elif val >= 2:
        return "Square Transform"
    else:
        return "ValueOutOfRange"


def ts_viz_3_1(base_path, x_col, y_col):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    x_col
        Timestamp / date column name
    y_col
        Numerical column names

    Returns
    -------
    DatapaneObject
    """

    ts_fig = []

    df = pd.read_csv(ends_with(base_path) + x_col + "_" + y_col + "_daily.csv").dropna()
    df[x_col] = pd.to_datetime(df[x_col], format="%Y-%m-%d %H:%M:%S.%f")
    df = df.set_index(x_col)

    for metric_col in ["mean", "median", "min", "max"]:

        try:
            adf_test = (
                round(adfuller(df[metric_col])[0], 3),
                round(adfuller(df[metric_col])[1], 3),
            )
            if adf_test[1] < 0.05:
                adf_flag = True
            else:
                adf_flag = False
        except:
            adf_test = ("nan", "nan")
            adf_flag = False

        try:
            kpss_test = (
                round(kpss(df[metric_col], regression="ct")[0], 3),
                round(kpss(df[metric_col], regression="ct")[1], 3),
            )
            if kpss_test[1] < 0.05:
                kpss_flag = True
            else:
                kpss_flag = False
        except:
            kpss_test = ("nan", "nan")
            kpss_flag = False

        #         df[metric_col] = df[metric_col].apply(lambda x: boxcox1p(x,0.25))
        #         lambda_box_cox = round(boxcox(df[metric_col])[1],5)
        fit = PowerTransformer(method="yeo-johnson")

        try:
            lambda_box_cox = round(
                fit.fit(np.array(df[metric_col]).reshape(-1, 1)).lambdas_[0], 3
            )
            cnt = 0
        except:
            cnt = 1

        if cnt == 0:

            #         df[metric_col+"_transformed"] = boxcox(df[metric_col],lmbda=lambda_box_cox)
            df[metric_col + "_transformed"] = fit.transform(
                np.array(df[metric_col]).reshape(-1, 1)
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Pre-Transformation", "Post-Transformation"],
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[metric_col],
                    mode="lines+markers",
                    name=metric_col,
                    line=dict(color=global_theme[1]),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[metric_col + "_transformed"],
                    mode="lines+markers",
                    name=metric_col + "_transformed",
                    line=dict(color=global_theme[7]),
                ),
                row=1,
                col=2,
            )
            fig.layout.plot_bgcolor = global_plot_bg_color
            fig.layout.paper_bgcolor = global_paper_bg_color
            fig.update_xaxes(gridcolor=px.colors.sequential.Greys[1])
            fig.update_yaxes(gridcolor=px.colors.sequential.Greys[1])
            fig.update_layout(autosize=True, width=2000, height=400)
            fig.update_layout(
                legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center")
            )

            ts_fig.append(
                dp.Group(
                    dp.Group(
                        dp.BigNumber(
                            heading="ADF Test Statistic",
                            value=adf_test[0],
                            change=adf_test[1],
                            is_upward_change=adf_flag,
                        ),
                        dp.BigNumber(
                            heading="KPSS Test Statistic",
                            value=kpss_test[0],
                            change=kpss_test[1],
                            is_upward_change=kpss_flag,
                        ),
                        dp.BigNumber(
                            heading="Box-Cox Transformation",
                            value=lambda_box_cox,
                            change=str(lambda_cat(lambda_box_cox)),
                            is_upward_change=True,
                        ),
                        columns=3,
                    ),
                    dp.Text("#### Transformation View"),
                    dp.Text(
                        "Below Transformation is basis the inferencing from the Box Cox Transformation. The Lambda value of "
                        + str(lambda_box_cox)
                        + " indicates a "
                        + str(lambda_cat(lambda_box_cox))
                        + ". A Pre-Post Transformation Visualization is done for better clarity. "
                    ),
                    dp.Plot(fig),
                    dp.Text("**Guidelines :** "),
                    dp.Text(
                        "**ADF** : *The more negative the statistic, the more likely we are to reject the null hypothesis. If the p-value is less than the significance level of 0.05, we can reject the null hypothesis and take that the series is stationary*"
                    ),
                    dp.Text(
                        "**KPSS** : *If the p-value is high, we cannot reject the null hypothesis. So the series is stationary.*"
                    ),
                    label=metric_col.title(),
                )
            )
        else:

            ts_fig.append(
                dp.Group(
                    dp.Group(
                        dp.BigNumber(
                            heading="ADF Test Statistic",
                            value=adf_test[0],
                            change=adf_test[1],
                            is_upward_change=adf_flag,
                        ),
                        dp.BigNumber(
                            heading="KPSS Test Statistic",
                            value=kpss_test[0],
                            change=kpss_test[1],
                            is_upward_change=kpss_flag,
                        ),
                        dp.BigNumber(
                            heading="Box-Cox Transformation",
                            value="ValueOutOfRange",
                            change="ValueOutOfRange",
                            is_upward_change=True,
                        ),
                        columns=3,
                    ),
                    dp.Text("**Guidelines :** "),
                    dp.Text(
                        "**ADF** : *The more negative the statistic, the more likely we are to reject the null hypothesis. If the p-value is less than the significance level of 0.05, we can reject the null hypothesis and take that the series is stationary*"
                    ),
                    dp.Text(
                        "**KPSS** : *If the p-value is high, we cannot reject the null hypothesis. So the series is stationary.*"
                    ),
                    label=metric_col.title(),
                )
            )

    return dp.Select(blocks=ts_fig, type=dp.SelectType.TABS)


def ts_viz_3_2(base_path, ts_col, col_list):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    col_list
        Numerical column names

    Returns
    -------
    DatapaneObject
    """

    bl = []

    for i in col_list:
        if len(num_cols) > 1:
            bl.append(dp.Group(ts_viz_3_1(base_path, ts_col, i), label=i))
        else:
            bl.append(dp.Group(ts_viz_3_1(base_path, ts_col, i), label=i))
            bl.append(dp.Group(dp.Plot(blank_chart, label=" "), label=" "))

    return dp.Select(blocks=bl, type=dp.SelectType.DROPDOWN)


def ts_viz_3_3(base_path, ts_col, num_cols):

    """

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    ts_col
        Timestamp / date column name
    num_cols
        Numerical column names

    Returns
    -------
    DatapaneObject
    """

    #     f = list(pd.read_csv(ends_with(base_path) + "stats_" + i + "_2.csv").count_unique_dates.values)[0]

    # if f >= 6:
    if len(ts_col) > 1:
        ts_v = []
        for i in ts_col:
            f = list(
                pd.read_csv(
                    ends_with(base_path) + "stats_" + i + "_2.csv"
                ).count_unique_dates.values
            )[0]
            if f >= 6:
                ts_v.append(dp.Group(ts_viz_3_2(base_path, i, num_cols), label=i))
            else:
                ts_v.append(
                    dp.Group(
                        dp.Text(
                            "The data contains insufficient data points for the desired transformation analysis. Please ensure the number of unique dates is sufficient."
                        ),
                        label=i,
                    )
                )

    else:
        ts_v = []
        for i in ts_col:
            f = list(
                pd.read_csv(
                    ends_with(base_path) + "stats_" + i + "_2.csv"
                ).count_unique_dates.values
            )[0]
            if f >= 6:
                ts_v.append(dp.Group(ts_viz_3_2(base_path, i, num_cols), label=i))
                ts_v.append(dp.Plot(blank_chart, label="_"))
            else:
                ts_v.append(
                    dp.Group(
                        dp.Text(
                            "The data contains insufficient data points for the desired transformation analysis. Please ensure the number of unique dates is sufficient."
                        ),
                        label=i,
                    )
                )
                ts_v.append(dp.Plot(blank_chart, label="_"))

    return dp.Select(blocks=ts_v, type=dp.SelectType.DROPDOWN)


def ts_stats(base_path):

    """

    This function helps to read the base data containing desired input and produces output specific to the `ts_cols_stats.csv` file

    Parameters
    ----------
    base_path
        Base path which is the same as Master path where the aggregated data resides.
    Returns
    -------
    List
    """

    df = pd.read_csv(base_path + "ts_cols_stats.csv")

    all_stats = []
    for i in range(0, 7):
        try:
            all_stats.append(df[df.index.values == i].values[0][0].split(","))
        except:
            all_stats.append([])

    c0 = pd.DataFrame(all_stats[0], columns=["attributes"])
    c1 = pd.DataFrame(list_ts_remove_append(all_stats[1], 1), columns=["attributes"])
    c1["Analyzed Attributes"] = "✔"
    c2 = pd.DataFrame(list_ts_remove_append(all_stats[2], 1), columns=["attributes"])
    c2["Attributes Identified"] = "✔"
    c3 = pd.DataFrame(list_ts_remove_append(all_stats[3], 1), columns=["attributes"])
    c3["Attributes Pre-Existed"] = "✔"
    c4 = pd.DataFrame(list_ts_remove_append(all_stats[4], 1), columns=["attributes"])
    c4["Overall TimeStamp Attributes"] = "✔"

    c5 = list_ts_remove_append(all_stats[5], 1)
    c6 = list_ts_remove_append(all_stats[6], 1)

    return c0, c1, c2, c3, c4, c5, c6


def ts_viz_generate(master_path, id_col, print_report=False, output_type=None):

    """

    This function helps to produce the output in the nested / recursive function supported by datapane. Eventually this is populated at the final report.


    Parameters
    ----------
    master_path
        Master path where the aggregated data resides.
    id_col
        ID Column
    print_report
        Option to specify whether the Report needs to be saved or not. True / False can be used to specify the needful.
    output_type
        Time category of analysis which can be between "Daily", "Hourly", "Weekly"

    Returns
    -------
    DatapaneObject / Output[HTML]
    """

    master_path = ends_with(master_path)

    try:
        c0, c1, c2, c3, c4, c5, c6 = ts_stats(master_path)

    except:
        return "null_report"

    stats_df = (
        c0.merge(c1, on="attributes", how="left")
        .merge(c2, on="attributes", how="left")
        .merge(c3, on="attributes", how="left")
        .merge(c4, on="attributes", how="left")
        .fillna("✘")
    )

    global num_cols
    global cat_cols

    num_cols, cat_cols = c5, c6

    final_ts_cols = list(ts_stats(master_path)[4].attributes.values)

    if output_type == "daily":

        report = dp.Group(
            dp.Text("# "),
            dp.Text(
                "*This section summarizes the information about timestamp features and how they are interactive with other attributes. An exhaustive diagnosis is done by looking at different time series components, how they could be useful in deriving insights for further downstream applications*"
            ),
            dp.Text("# "),
            dp.Text("# "),
            dp.Text("### Basic Landscaping"),
            dp.Text(
                "Out of **"
                + str(len(list(ts_stats(master_path)[1].attributes.values)))
                + "** potential attributes in the data, the module could locate **"
                + str(len(final_ts_cols))
                + "** attributes as Timestamp"
            ),
            dp.DataTable(stats_df),
            ts_landscape(master_path, final_ts_cols, id_col),
            dp.Text(
                "*Lower the **CoV** (Coefficient Of Variation), Higher the Consistency between the consecutive dates. Similarly the Mean & Variance should be consistent over time*"
            ),
            dp.Text("### Visualization across the Shortlisted Timestamp Attributes"),
            ts_viz_1_3(master_path, final_ts_cols, num_cols, cat_cols, output_type),
            dp.Text("### Decomposed View"),
            ts_viz_2_3(master_path, final_ts_cols, num_cols),
            dp.Text("### Stationarity & Transformations"),
            ts_viz_3_3(master_path, final_ts_cols, num_cols),
            dp.Text("#"),
            dp.Text("#"),
            label="Time Series Analyzer",
        )

    elif output_type is None:
        report = "null_report"

    else:

        report = dp.Group(
            dp.Text("# "),
            dp.Text(
                "*This section summarizes the information about timestamp features and how they are interactive with other attributes. An exhaustive diagnosis is done by looking at different time series components, how they could be useful in deriving insights for further downstream applications*"
            ),
            dp.Text("# "),
            dp.Text("# "),
            dp.Text("### Basic Landscaping"),
            dp.Text(
                "Out of **"
                + str(len(list(ts_stats(master_path)[1].attributes.values)))
                + "** potential attributes in the data, the module could locate **"
                + str(len(final_ts_cols))
                + "** attributes as Timestamp"
            ),
            dp.DataTable(stats_df),
            ts_landscape(master_path, final_ts_cols, id_col),
            dp.Text(
                "*Lower the **CoV** (Coefficient Of Variation), Higher the Consistency between the consecutive dates. Similarly the Mean & Variance should be consistent over time*"
            ),
            dp.Text("### Visualization across the Shortlisted Timestamp Attributes"),
            ts_viz_1_3(master_path, final_ts_cols, num_cols, cat_cols, output_type),
            dp.Text("#"),
            dp.Text("#"),
            label="Time Series Analyzer",
        )

    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "time_series_analyzer.html", open=True
        )

    return report


def overall_stats_gen(lat_col_list, long_col_list, geohash_col_list):

    """

    This function helps to produce a basic summary of all the geospatial fields auto-detected in a dictionary along with the length of lat-lon & geohash cols identified.

    Parameters
    ----------
    lat_col_list
        List of latitude columns identified
    long_col_list
        List of longitude columns identified
    geohash_col_list
        List of geohash columns identified

    Returns
    -------
    Dictionary,Integer,Integer
    """

    d = {}
    ll = []

    col_list = ["Latitude Col", "Longitude Col", "Geohash Col"]
    #     for idx,i in enumerate([lat_col_list,long_col_list,geohash_col_list,polygon_col_list]):
    for idx, i in enumerate([lat_col_list, long_col_list, geohash_col_list]):
        if i is None:
            ll = []
        elif i is not None:
            ll = []
            for j in i:
                ll.append(j)
        d[col_list[idx]] = ",".join(ll)

    l1 = len(lat_col_list)
    l2 = len(geohash_col_list)

    return d, l1, l2


def loc_field_stats(lat_col_list, long_col_list, geohash_col_list, max_records):

    """

    This function helps to produce a basic summary of all the geospatial fields auto-detected

    Parameters
    ----------
    lat_col_list
        List of latitude columns identified
    long_col_list
        List of longitude columns identified
    geohash_col_list
        List of geohash columns identified
    max_records
        Maximum geospatial points analyzed

    Returns
    -------
    DatapaneObject
    """

    loc_cnt = (
        overall_stats_gen(lat_col_list, long_col_list, geohash_col_list)[1] * 2
    ) + (overall_stats_gen(lat_col_list, long_col_list, geohash_col_list)[2])
    loc_var_stats = overall_stats_gen(lat_col_list, long_col_list, geohash_col_list)[0]

    x = "#"

    t0 = dp.Text(x)
    t1 = dp.Text(
        "There are **"
        + str(loc_cnt)
        + "** location fields captured in the data containing "
        + str(overall_stats_gen(lat_col_list, long_col_list, geohash_col_list)[1])
        + " pair(s) of **Lat,Long** & "
        + str(overall_stats_gen(lat_col_list, long_col_list, geohash_col_list)[2])
        + " **Geohash** field(s)"
    )
    t2 = dp.DataTable(
        pd.DataFrame(pd.Series(loc_var_stats, index=loc_var_stats.keys())).rename(
            columns={0: ""}
        )
    )

    return dp.Group(t0, t1, t2)


def read_stats_ll_geo(lat_col, long_col, geohash_col, master_path, top_geo_records):

    """

    This function helps to read all the basis stats output for the lat-lon & geohash field produced from the analyzer module

    Parameters
    ----------
    lat_col
        Latitude column identified
    long_col
        Longitude column identified
    geohash_col
        Geohash column identified
    master_path
        Master path where the aggregated data resides
    top_geo_records
        Top geospatial records displayed

    Returns
    -------
    DatapaneObject

    """

    try:
        len_lat_col = len(lat_col)

    except:

        len_lat_col = 0

    try:
        len_geohash_col = len(geohash_col)
    except:
        len_geohash_col = 0

    ll_stats, geohash_stats = [], []

    if len_lat_col > 0:

        if len_lat_col == 1:
            for idx, i in enumerate(lat_col):
                ll_stats.append(
                    dp.Group(
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Overall_Summary_1_"
                                        + lat_col[idx]
                                        + "_"
                                        + long_col[idx]
                                        + ".csv"
                                    ),
                                    label="Overall Summary",
                                ),
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Top_"
                                        + str(top_geo_records)
                                        + "_Lat_Long_1_"
                                        + lat_col[idx]
                                        + "_"
                                        + long_col[idx]
                                        + ".csv"
                                    ),
                                    label="Top " + str(top_geo_records) + " Lat Long",
                                ),
                            ],
                            type=dp.SelectType.TABS,
                        ),
                        label=lat_col[idx] + "_" + long_col[idx],
                    )
                )
                ll_stats.append(
                    dp.Group(
                        dp.DataTable(
                            pd.DataFrame(columns=[" "], index=range(1)), label=" "
                        ),
                        label=" ",
                    )
                )

        elif len_lat_col > 1:
            for idx, i in enumerate(lat_col):
                ll_stats.append(
                    dp.Group(
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Overall_Summary_1_"
                                        + lat_col[idx]
                                        + "_"
                                        + long_col[idx]
                                        + ".csv"
                                    ),
                                    label="Overall Summary",
                                ),
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Top_"
                                        + str(top_geo_records)
                                        + "_Lat_Long_1_"
                                        + lat_col[idx]
                                        + "_"
                                        + long_col[idx]
                                        + ".csv"
                                    ),
                                    label="Top " + str(top_geo_records) + " Lat Long",
                                ),
                            ],
                            type=dp.SelectType.TABS,
                        ),
                        label=lat_col[idx] + "_" + long_col[idx],
                    )
                )

        ll_stats = dp.Select(blocks=ll_stats, type=dp.SelectType.DROPDOWN)

    if len_geohash_col > 0:

        if len_geohash_col == 1:
            for idx, i in enumerate(geohash_col):
                geohash_stats.append(
                    dp.Group(
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Overall_Summary_2_"
                                        + geohash_col[idx]
                                        + ".csv"
                                    ),
                                    label="Overall Summary",
                                ),
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Top_"
                                        + str(top_geo_records)
                                        + "_Geohash_Distribution_2_"
                                        + geohash_col[idx]
                                        + ".csv"
                                    ),
                                    label="Top "
                                    + str(top_geo_records)
                                    + "  Geohash Distribution",
                                ),
                            ],
                            type=dp.SelectType.TABS,
                        ),
                        label=geohash_col[idx],
                    )
                )
                geohash_stats.append(
                    dp.Group(
                        dp.DataTable(
                            pd.DataFrame(columns=[" "], index=range(1)), label=" "
                        ),
                        label=" ",
                    )
                )

        elif len_geohash_col > 1:
            for idx, i in enumerate(geohash_col):
                geohash_stats.append(
                    dp.Group(
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Overall_Summary_2_"
                                        + geohash_col[idx]
                                        + ".csv"
                                    ),
                                    label="Overall Summary",
                                ),
                                dp.DataTable(
                                    pd.read_csv(
                                        ends_with(master_path)
                                        + "Top_"
                                        + str(top_geo_records)
                                        + "_Geohash_Distribution_2_"
                                        + geohash_col[idx]
                                        + ".csv"
                                    ),
                                    label="Top "
                                    + str(top_geo_records)
                                    + "  Geohash Distribution",
                                ),
                            ],
                            type=dp.SelectType.TABS,
                        ),
                        label=geohash_col[idx],
                    )
                )

        geohash_stats = dp.Select(blocks=geohash_stats, type=dp.SelectType.DROPDOWN)

    if (len_lat_col + len_geohash_col) == 1:

        if len_lat_col == 0:

            return geohash_stats

        else:
            return ll_stats

    elif (len_lat_col + len_geohash_col) > 1:

        if (len_lat_col > 1) and (len_geohash_col == 0):

            return ll_stats

        elif (len_lat_col == 0) and (len_geohash_col > 1):

            return geohash_stats

        elif (len_lat_col >= 1) and (len_geohash_col >= 1):

            return dp.Select(
                blocks=[
                    dp.Group(ll_stats, label="Lat-Long-Stats"),
                    dp.Group(geohash_stats, label="Geohash-Stats"),
                ],
                type=dp.SelectType.TABS,
            )


def read_cluster_stats_ll_geo(lat_col, long_col, geohash_col, master_path):

    """

    This function helps to read all the cluster analysis output for the lat-lon & geohash field produced from the analyzer module

    Parameters
    ----------
    lat_col
        Latitude column identified
    long_col
        Longitude column identified
    geohash_col
        Geohash column identified
    master_path
        Master path where the aggregated data resides

    Returns
    -------
    DatapaneObject
    """

    ll_col, plot_ll, all_geo_cols = [], [], []

    try:
        len_lat_col = len(lat_col)

    except:

        len_lat_col = 0

    try:
        len_geohash_col = len(geohash_col)
    except:
        len_geohash_col = 0

    if (len_lat_col > 0) or (len_geohash_col > 0):

        try:
            for idx, i in enumerate(lat_col):

                ll_col.append(lat_col[idx] + "_" + long_col[idx])
        except:
            pass
        all_geo_cols = ll_col + geohash_col

    if len(all_geo_cols) > 0:

        for i in all_geo_cols:

            if len(all_geo_cols) == 1:

                p1 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path) + "cluster_plot_1_elbow_" + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_1_silhoutte_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Cluster Identification",
                )

                p2 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_2_kmeans_"
                                    + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_2_dbscan_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Cluster Distribution",
                )

                p3 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_3_kmeans_"
                                    + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_3_dbscan_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Visualization",
                )

                p4 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_4_dbscan_1_"
                                    + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_4_dbscan_2_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Outlier Points",
                )

                plot_ll.append(
                    dp.Group(
                        dp.Select(blocks=[p1, p2, p3, p4], type=dp.SelectType.TABS),
                        label=i,
                    )
                )

                plot_ll.append(dp.Plot(blank_chart, label=" "))

            elif len(all_geo_cols) > 1:

                p1 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path) + "cluster_plot_1_elbow_" + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_1_silhoutte_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Cluster Identification",
                )

                p2 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_2_kmeans_"
                                    + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_2_dbscan_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Cluster Distribution",
                )

                p3 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_3_kmeans_"
                                    + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_3_dbscan_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Visualization",
                )

                p4 = dp.Group(
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_4_dbscan_1_"
                                    + i
                                )
                            )
                        )
                    ),
                    dp.Plot(
                        go.Figure(
                            json.load(
                                open(
                                    ends_with(master_path)
                                    + "cluster_plot_4_dbscan_2_"
                                    + i
                                )
                            )
                        )
                    ),
                    label="Outlier Points",
                )

                plot_ll.append(
                    dp.Group(
                        dp.Select(blocks=[p1, p2, p3, p4], type=dp.SelectType.TABS),
                        label=i,
                    )
                )

        return dp.Select(blocks=plot_ll, type=dp.SelectType.DROPDOWN)


def read_loc_charts(master_path):

    """

    This function helps to read all the geospatial charts from the master path and populate in the report

    Parameters
    ----------

    master_path
        Master path where the aggregated data resides

    Returns
    -------
    DatapaneObject
    """

    ll_charts_nm = [x for x in os.listdir(master_path) if "loc_charts_ll" in x]
    geo_charts_nm = [x for x in os.listdir(master_path) if "loc_charts_gh" in x]

    ll_col_charts, geo_col_charts = [], []

    if len(ll_charts_nm) > 0:

        if len(ll_charts_nm) == 1:
            for i1 in ll_charts_nm:
                col_name = i1.replace("loc_charts_ll_", "")
                ll_col_charts.append(
                    dp.Plot(
                        go.Figure(json.load(open(ends_with(master_path) + i1))),
                        label=col_name,
                    )
                )
                ll_col_charts.append(dp.Plot(blank_chart, label=" "))
        elif len(ll_charts_nm) > 1:
            for i1 in ll_charts_nm:
                col_name = i1.replace("loc_charts_ll_", "")
                ll_col_charts.append(
                    dp.Plot(
                        go.Figure(json.load(open(ends_with(master_path) + i1))),
                        label=col_name,
                    )
                )

        ll_col_charts = dp.Select(blocks=ll_col_charts, type=dp.SelectType.DROPDOWN)

    if len(geo_charts_nm) > 0:

        if len(geo_charts_nm) == 1:
            for i2 in geo_charts_nm:
                col_name = i2.replace("loc_charts_gh_", "")
                geo_col_charts.append(
                    dp.Plot(
                        go.Figure(json.load(open(ends_with(master_path) + i2))),
                        label=col_name,
                    )
                )
                geo_col_charts.append(dp.Plot(blank_chart, label=" "))

        elif len(geo_charts_nm) > 1:
            for i2 in geo_charts_nm:
                col_name = i2.replace("loc_charts_gh_", "")
                geo_col_charts.append(
                    dp.Plot(
                        go.Figure(json.load(open(ends_with(master_path) + i2))),
                        label=col_name,
                    )
                )

        geo_col_charts = dp.Select(blocks=geo_col_charts, type=dp.SelectType.DROPDOWN)

    if (len(ll_charts_nm) > 0) and (len(geo_charts_nm) == 0):

        return ll_col_charts

    elif (len(ll_charts_nm) == 0) and (len(geo_charts_nm) > 0):

        return geo_col_charts

    elif (len(ll_charts_nm) > 0) and (len(geo_charts_nm) > 0):

        return dp.Select(
            blocks=[
                dp.Group(ll_col_charts, label="Lat-Long-Plot"),
                dp.Group(geo_col_charts, label="Geohash-Plot"),
            ],
            type=dp.SelectType.TABS,
        )


def loc_report_gen(
    lat_cols,
    long_cols,
    geohash_cols,
    master_path,
    max_records,
    top_geo_records,
    print_report=False,
):

    """

    This function helps to read all the lat,long & geohash columns as input alongside few input parameters to produce the geospatial analysis report tab

    Parameters
    ----------
    lat_cols
        Latitude columns identified in the data
    long_cols
        Longitude columns identified in the data
    geohash_cols
        Geohash columns identified in the data
    master_path
        Master path where the aggregated data resides
    max_records
        Maximum geospatial points analyzed
    top_geo_records
        Top geospatial records displayed
    print_report
        Option to specify whether the Report needs to be saved or not. True / False can be used to specify the needful

    Returns
    -------
    DatapaneObject
    """

    _ = dp.Text("#")
    dp1 = dp.Group(
        _,
        dp.Text(
            "*This section summarizes the information about the geospatial features identified in the data and their landscaping view*"
        ),
        loc_field_stats(lat_cols, long_cols, geohash_cols, max_records),
    )

    if (len(lat_cols) + len(geohash_cols)) > 0:

        dp2 = dp.Group(
            _,
            dp.Text("## Descriptive Analysis by Location Attributes"),
            read_stats_ll_geo(
                lat_cols, long_cols, geohash_cols, master_path, top_geo_records
            ),
            _,
        )
        dp3 = dp.Group(
            _,
            dp.Text("## Clustering Geospatial Field"),
            read_cluster_stats_ll_geo(lat_cols, long_cols, geohash_cols, master_path),
            _,
        )
        dp4 = dp.Group(
            _,
            dp.Text("## Visualization by Geospatial Fields"),
            read_loc_charts(master_path),
            _,
        )

        report = dp.Group(dp1, dp2, dp3, dp4, label="Geospatial Analyzer")

    elif (len(lat_cols) + len(geohash_cols)) == 0:

        report = "null_report"

    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "geospatial_analyzer.html", open=True
        )

    return report


def anovos_report(
    master_path,
    id_col=None,
    label_col=None,
    corr_threshold=0.4,
    iv_threshold=0.02,
    drift_threshold_model=0.1,
    dataDict_path=".",
    metricDict_path=".",
    run_type="local",
    final_report_path=".",
    output_type=None,
    mlflow_config=None,
    lat_cols=[],
    long_cols=[],
    gh_cols=[],
    max_records=100000,
    top_geo_records=100,
    auth_key="NA",
):
    """

    This function actually helps to produce the final report by scanning through the output processed from the data analyzer module.

    Parameters
    ----------
    master_path
        Path containing the input files.
    id_col
        ID column (Default value = "")
    label_col
        label column (Default value = "")
    corr_threshold
        Correlation threshold beyond which attributes can be categorized under correlated. (Default value = 0.4)
    iv_threshold
        IV threshold beyond which attributes can be called as significant. (Default value = 0.02)
    drift_threshold_model
        threshold which the user is specifying for tagging an attribute to be drifted or not (Default value = 0.1)
    dataDict_path
        Data dictionary path. Default value is kept as None.
    metricDict_path
        Metric dictionary path. Default value is kept as None.
    run_type
        local or emr or databricks or ak8s option. Default is kept as local
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type.
    final_report_path
        Path where the report will be saved. (Default value = ".")
    output_type
        Time category of analysis which can be between "Daily", "Hourly", "Weekly"
    mlflow_config
        MLflow configuration. If None, all MLflow features are disabled.
    lat_cols
        Latitude columns identified in the data
    long_cols
        Longitude columns identified in the data
    gh_cols
        Geohash columns identified in the data
    max_records
        Maximum geospatial points analyzed
    top_geo_records
        Top geospatial records displayed


    Returns
    -------
    Output[HTML]
    """

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp --recursive "
            + ends_with(master_path)
            + " "
            + ends_with("report_stats")
        )
        master_path = "report_stats"
        subprocess.check_output(["bash", "-c", bash_cmd])

    if run_type == "databricks":
        master_path = output_to_local(master_path)
        dataDict_path = output_to_local(dataDict_path)
        metricDict_path = output_to_local(metricDict_path)
        final_report_path = output_to_local(final_report_path)

    if run_type == "ak8s":
        output_path_mod = path_ak8s_modify(master_path)
        bash_cmd = (
            'azcopy cp "'
            + ends_with(output_path_mod)
            + str(auth_key)
            + '" "'
            + ends_with("report_stats")
            + '" --recursive=true'
        )
        master_path = "report_stats"
        subprocess.check_output(["bash", "-c", bash_cmd])

    if "global_summary.csv" not in os.listdir(master_path):
        print(
            "Minimum supporting data is unavailable, hence the Report could not be generated."
        )
        return None

    global global_summary_df
    global numcols_name
    global catcols_name
    global rows_count
    global columns_count
    global numcols_count
    global catcols_count
    global blank_chart
    global df_si_
    global df_si
    global unstable_attr
    global total_unstable_attr
    global drift_df
    global metric_drift
    global drift_df
    global len_feats
    global drift_df_stats
    global drifted_feats
    global df_stability
    global n_df_stability
    global stability_interpretation_table
    global plot_index_stability

    SG_tabs = [
        "measures_of_counts",
        "measures_of_centralTendency",
        "measures_of_cardinality",
        "measures_of_percentiles",
        "measures_of_dispersion",
        "measures_of_shape",
        "global_summary",
    ]
    QC_tabs = [
        "nullColumns_detection",
        "IDness_detection",
        "biasedness_detection",
        "invalidEntries_detection",
        "duplicate_detection",
        "nullRows_detection",
        "outlier_detection",
    ]
    AE_tabs = [
        "correlation_matrix",
        "IV_calculation",
        "IG_calculation",
        "variable_clustering",
    ]
    drift_tab = ["drift_statistics"]
    stability_tab = ["stability_index", "stabilityIndex_metrics"]
    avl_SG, avl_QC, avl_AE = [], [], []

    stability_interpretation_table = pd.DataFrame(
        [
            ["0-1", "Very Unstable"],
            ["1-2", "Unstable"],
            ["2-3", "Marginally Stable"],
            ["3-3.5", "Stable"],
            ["3.5-4", "Very Stable"],
        ],
        columns=["StabilityIndex", "StabilityOrder"],
    )
    plot_index_stability = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(stability_interpretation_table.columns),
                    fill_color=px.colors.sequential.Greys[2],
                    align="center",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[
                        stability_interpretation_table.StabilityIndex,
                        stability_interpretation_table.StabilityOrder,
                    ],
                    line_color=px.colors.sequential.Greys[2],
                    fill_color="white",
                    align="center",
                    height=25,
                ),
                columnwidth=[2, 10],
            )
        ]
    )

    plot_index_stability.update_layout(margin=dict(l=20, r=700, t=20, b=20))

    blank_chart = go.Figure()
    blank_chart.update_layout(autosize=False, width=10, height=10)
    blank_chart.layout.plot_bgcolor = global_plot_bg_color
    blank_chart.layout.paper_bgcolor = global_paper_bg_color
    blank_chart.update_xaxes(visible=False)
    blank_chart.update_yaxes(visible=False)

    global_summary_df = pd.read_csv(ends_with(master_path) + "global_summary.csv")
    rows_count = int(
        global_summary_df[global_summary_df.metric.values == "rows_count"].value.values[
            0
        ]
    )
    catcols_count = int(
        global_summary_df[
            global_summary_df.metric.values == "catcols_count"
        ].value.values[0]
    )
    numcols_count = int(
        global_summary_df[
            global_summary_df.metric.values == "numcols_count"
        ].value.values[0]
    )
    columns_count = int(
        global_summary_df[
            global_summary_df.metric.values == "columns_count"
        ].value.values[0]
    )
    if catcols_count > 0:
        catcols_name = ",".join(
            list(
                global_summary_df[
                    global_summary_df.metric.values == "catcols_name"
                ].value.values
            )
        )
    else:
        catcols_name = ""
    if numcols_count > 0:
        numcols_name = ",".join(
            list(
                global_summary_df[
                    global_summary_df.metric.values == "numcols_name"
                ].value.values
            )
        )
    else:
        numcols_name = ""
    all_files = os.listdir(master_path)
    eventDist_charts = [x for x in all_files if "eventDist" in x]
    stats_files = [x for x in all_files if ".csv" in x]
    freq_charts = [x for x in all_files if "freqDist" in x]
    outlier_charts = [x for x in all_files if "outlier" in x]
    drift_charts = [x for x in all_files if "drift" in x and ".csv" not in x]
    all_charts_num_1_ = chart_gen_list(
        master_path, chart_type=freq_charts, type_col="numerical"
    )
    all_charts_num_2_ = chart_gen_list(
        master_path, chart_type=eventDist_charts, type_col="numerical"
    )
    all_charts_num_3_ = chart_gen_list(
        master_path, chart_type=outlier_charts, type_col="numerical"
    )
    all_charts_cat_1_ = chart_gen_list(
        master_path, chart_type=freq_charts, type_col="categorical"
    )
    all_charts_cat_2_ = chart_gen_list(
        master_path, chart_type=eventDist_charts, type_col="categorical"
    )
    all_drift_charts_ = chart_gen_list(master_path, chart_type=drift_charts)
    for x in [
        all_charts_num_1_,
        all_charts_num_2_,
        all_charts_num_3_,
        all_charts_cat_1_,
        all_charts_cat_2_,
        all_drift_charts_,
    ]:
        if len(x) == 1:
            x.append(dp.Plot(blank_chart, label=" "))
        else:
            x
    mapping_tab_list = []
    for i in stats_files:
        if i.split(".csv")[0] in SG_tabs:
            mapping_tab_list.append([i.split(".csv")[0], "Descriptive Statistics"])
        elif i.split(".csv")[0] in QC_tabs:
            mapping_tab_list.append([i.split(".csv")[0], "Quality Check"])
        elif i.split(".csv")[0] in AE_tabs:
            mapping_tab_list.append([i.split(".csv")[0], "Attribute Associations"])
        elif i.split(".csv")[0] in drift_tab or i.split(".csv")[0] in stability_tab:
            mapping_tab_list.append([i.split(".csv")[0], "Data Drift & Data Stability"])
        else:
            mapping_tab_list.append([i.split(".csv")[0], "null"])
    xx = pd.DataFrame(mapping_tab_list, columns=["file_name", "tab_name"])
    xx_avl = list(set(xx.file_name.values))
    for i in SG_tabs:
        if i in xx_avl:
            avl_SG.append(i)
    for j in QC_tabs:
        if j in xx_avl:
            avl_QC.append(j)
    for k in AE_tabs:
        if k in xx_avl:
            avl_AE.append(k)
    missing_SG = list(set(SG_tabs) - set(avl_SG))
    missing_QC = list(set(QC_tabs) - set(avl_QC))
    missing_AE = list(set(AE_tabs) - set(avl_AE))
    missing_drift = list(
        set(drift_tab)
        - set(xx[xx.tab_name.values == "Data Drift & Data Stability"].file_name.values)
    )
    missing_stability = list(
        set(stability_tab)
        - set(xx[xx.tab_name.values == "Data Drift & Data Stability"].file_name.values)
    )
    ds_ind = drift_stability_ind(
        missing_drift, drift_tab, missing_stability, stability_tab
    )
    if ds_ind[0] > 0:
        drift_df = pd.read_csv(
            ends_with(master_path) + "drift_statistics.csv"
        ).sort_values(by=["flagged"], ascending=False)
        metric_drift = list(drift_df.drop(["attribute", "flagged"], 1).columns)
        drift_df = drift_df[drift_df.attribute.values != id_col]
        len_feats = drift_df.shape[0]
        drift_df_stats = (
            drift_df[drift_df.flagged.values == 1]
            .melt(id_vars="attribute", value_vars=metric_drift)
            .sort_values(by=["variable", "value"], ascending=False)
        )
        drifted_feats = drift_df[drift_df.flagged.values == 1].shape[0]
    if ds_ind[1] > 0.5:
        df_stability = pd.read_csv(
            ends_with(master_path) + "stabilityIndex_metrics.csv"
        )
        df_stability["idx"] = df_stability["idx"].astype(str).apply(lambda x: "df" + x)
        n_df_stability = str(df_stability["idx"].nunique())
        df_si_ = pd.read_csv(ends_with(master_path) + "stability_index.csv")
        df_si = df_si_[
            [
                "attribute",
                "stability_index",
                "mean_si",
                "stddev_si",
                "kurtosis_si",
                "flagged",
            ]
        ]
        unstable_attr = list(df_si_[df_si_.flagged.values == 1].attribute.values)
        total_unstable_attr = list(df_si_.attribute.values)
    elif ds_ind[1] == 0.5:
        df_si_ = pd.read_csv(ends_with(master_path) + "stability_index.csv")
        df_si = df_si_[
            [
                "attribute",
                "stability_index",
                "mean_si",
                "stddev_si",
                "kurtosis_si",
                "flagged",
            ]
        ]
        unstable_attr = list(df_si_[df_si_.flagged.values == 1].attribute.values)
        total_unstable_attr = list(df_si_.attribute.values)
        df_stability = pd.DataFrame()
        n_df_stability = "the"
    else:
        pass
    tab1 = executive_summary_gen(
        master_path, label_col, ds_ind, id_col, iv_threshold, corr_threshold
    )
    tab2 = wiki_generator(
        master_path, dataDict_path=dataDict_path, metricDict_path=metricDict_path
    )
    tab3 = descriptive_statistics(
        master_path, SG_tabs, avl_SG, missing_SG, all_charts_num_1_, all_charts_cat_1_
    )
    tab4 = quality_check(master_path, QC_tabs, avl_QC, missing_QC, all_charts_num_3_)
    tab5 = attribute_associations(
        master_path,
        AE_tabs,
        avl_AE,
        missing_AE,
        label_col,
        all_charts_num_2_,
        all_charts_cat_2_,
    )
    tab6 = data_drift_stability(
        master_path, ds_ind, id_col, drift_threshold_model, all_drift_charts_
    )

    tab7 = ts_viz_generate(master_path, id_col, False, output_type)

    tab8 = loc_report_gen(
        lat_cols, long_cols, gh_cols, master_path, max_records, top_geo_records, False
    )

    final_tabs_list = []
    for i in [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8]:
        if i == "null_report":
            pass
        else:
            final_tabs_list.append(i)
    if run_type in ("local", "databricks"):
        run_id = (
            mlflow_config["run_id"]
            if mlflow_config is not None and mlflow_config["track_reports"]
            else ""
        )

        report_run_path = ends_with(final_report_path) + run_id + "/"
        dp.Report(
            default_template[0],
            default_template[1],
            dp.Select(blocks=final_tabs_list, type=dp.SelectType.TABS),
        ).save(report_run_path + "ml_anovos_report.html", open=True)
        if mlflow_config is not None:
            mlflow.log_artifact(report_run_path)
    elif run_type == "emr":
        dp.Report(
            default_template[0],
            default_template[1],
            dp.Select(blocks=final_tabs_list, type=dp.SelectType.TABS),
        ).save("ml_anovos_report.html", open=True)
        bash_cmd = "aws s3 cp ml_anovos_report.html " + ends_with(final_report_path)
        subprocess.check_output(["bash", "-c", bash_cmd])
    elif run_type == "ak8s":
        dp.Report(
            default_template[0],
            default_template[1],
            dp.Select(blocks=final_tabs_list, type=dp.SelectType.TABS),
        ).save("ml_anovos_report.html", open=True)
        bash_cmd = (
            'azcopy cp "ml_anovos_report.html" '
            + ends_with(path_ak8s_modify(final_report_path))
            + str(auth_key)
        )
        subprocess.check_output(["bash", "-c", bash_cmd])
    else:
        raise ValueError("Invalid run_type")
    print("Report generated successfully at the specified location")
