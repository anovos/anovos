import json
import os
import subprocess

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

from anovos.shared.utils import ends_with

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
    :param col: Analysis column containing "_" present gets replaced along with upper case conversion
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
    :param df1: Analysis dataframe pertaining to summarized stability metrics
    :param df2: Analysis dataframe pertaining to historical data
    :param col: Analysis column
    """

    def val_cat(val):
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
        df1 = df1[df1["attribute"] == col]

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
            rows=4,
            label=col,
        )

    else:
        return dp.Group(dp.Text("#"), dp.Text(f5), dp.Plot(f1), rows=3, label=col)


def data_analyzer_output(master_path, avl_recs_tab, tab_name):
    """
    :param master_path: Path containing all the output from analyzed data
    :param avl_recs_tab: Available file names from the analysis tab
    :param tab_name: Analysis tab from association_evaluator / quality_checker / stats_generator

    """
    df_list = []
    df_plot_list = []
    # @FIXME: unused variables
    txt_list = []
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

                unique_rows_count = f" No. Of Unique Rows: ** {_unique_rows_count}, **"
                # @FIXME: variable names exists in outer scope
                rows_count = f" No. of Rows: ** {_rows_count}, **"
                duplicate_rows = (
                    f" No. of Duplicate Rows: ** {_duplicate_rows_count}, **"
                )
                duplicate_pct = f" Percentage of Duplicate Rows: ** {_duplicate_pct}%**"

                df_list.append(
                    [
                        dp.Text("### " + str(remove_u_score(i))),
                        dp.Group(
                            dp.Text(rows_count),
                            dp.Text(unique_rows_count),
                            dp.Text(duplicate_rows),
                            dp.Text(duplicate_pct),
                            rows=4,
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
                            rows=3,
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
                            rows=3,
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
                                rows=3,
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
        missing_recs_drift: Missing files from the drift tab
        drift_tab: "drift_statistics"
    missing_recs_stability: Missing files from the stability tab
        stability_tab:"stabilityIndex_computation, stabilityIndex_metrics"

    """

    if len(missing_recs_drift) == len(drift_tab):
        drift_ind = 0
    else:
        drift_ind = 1

    if len(missing_recs_stability) == len(stability_tab):
        stability_ind = 0
    elif ("stabilityIndex_metrics" in missing_recs_stability) and (
        "stabilityIndex_computation" not in missing_recs_stability
    ):

        stability_ind = 0.5
    else:
        stability_ind = 1

    return drift_ind, stability_ind


def chart_gen_list(master_path, chart_type, type_col=None):
    """
    :param master_path: Path containing all the charts same as the other files from data analyzed output
    :param chart_type: Files containing only the specific chart names for the specific chart category
    :param type_col=None. Default value is kept as None

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
    :param master_path: Path containing the input files.
    :param label_col: Label column.
    :param ds_ind: Drift stability indicator in list form.
    :param id_col: ID column.
    :param iv_threshold: IV threshold beyond which attributes can be called as significant.
    :param corr_threshold: Correlation threshold beyond which attributes can be categorized under correlated.
    :param print_report: Printing option flexibility. Default value is kept as False.

    """

    try:
        obj_dtls = json.load(
            open(ends_with(master_path) + "freqDist_" + str(label_col))
        )
        # @FIXME: never used local variable
        text_val = list(list(obj_dtls.values())[0][0].items())[8][1]
        x_val = list(list(obj_dtls.values())[0][0].items())[11][1]
        y_val = list(list(obj_dtls.values())[0][0].items())[13][1]
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
            rows=2,
        )
    else:
        if label_fig_ is None:

            a2 = dp.Group(
                dp.Text("- Target variable is **" + str(label_col) + "** "),
                dp.Text("- Data Diagnosis:"),
                rows=2,
            )
        else:
            a2 = dp.Group(
                dp.Text("- Target variable is **" + str(label_col) + "** "),
                dp.Plot(label_fig_),
                dp.Text("- Data Diagnosis:"),
                rows=3,
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

    x = x[x.Attribute.values != "NA"]

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
    :param master_path: Path containing the input files.
    :param dataDict_path: Data dictionary path. Default value is kept as None.
    :param metricDict_path: Metric dictionary path. Default value is kept as None.
    :param print_report: Printing option flexibility. Default value is kept as False.
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
    :param master_path: Path containing the input files.
    :param SG_tabs: 'measures_of_counts','measures_of_centralTendency','measures_of_cardinality','measures_of_percentiles','measures_of_dispersion','measures_of_shape','global_summary'
    :param avl_recs_SG: Available files from the SG_tabs (Stats Generator tabs)
    :param missing_recs_SG: Missing files from the SG_tabs (Stats Generator tabs)
    :param all_charts_num_1_: Numerical charts (histogram) all collated in a list format supported as per datapane objects
    :param all_charts_cat_1_: Categorical charts (barplot) all collated in a list format supported as per datapane objects
    :param print_report: Printing option flexibility. Default value is kept as False.
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
                    rows=6,
                ),
                rows=8,
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
    :param master_path: Path containing the input files.
    :param QC_tabs: 'nullColumns_detection','IDness_detection','biasedness_detection','invalidEntries_detection','duplicate_detection','nullRows_detection','outlier_detection'
    :param avl_recs_QC: Available files from the QC_tabs (Quality Checker tabs)
    :param missing_recs_QC: Missing files from the QC_tabs (Quality Checker tabs)
    :param all_charts_num_3_: Numerical charts (outlier charts) all collated in a list format supported as per datapane objects
    :param print_report: Printing option flexibility. Default value is kept as False.
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
                rows=8,
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
                rows=8,
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
                        dp.Group(
                            dp.Text("# "), dp.Group(*c_), rows=2, label="Column Level"
                        ),
                        dp.Group(
                            dp.Text("# "), dp.Group(*r_), rows=2, label="Row Level"
                        ),
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
    :param master_path: Path containing the input files.
    :param AE_tabs: 'correlation_matrix','IV_calculation','IG_calculation','variable_clustering'
    :param avl_recs_AE: Available files from the AE_tabs (Association Evaluator tabs)
    :param missing_recs_AE: Missing files from the AE_tabs (Association Evaluator tabs)
    :param label_col: label column
    :param all_charts_num_2_: Numerical charts (histogram) all collated in a list format supported as per datapane objects
    :param all_charts_cat_2_: Categorical charts (barplot) all collated in a list format supported as per datapane objects
    :param print_report: Printing option flexibility. Default value is kept as False.
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
    :param master_path: Path containing the input files.
    :param ds_ind: Drift stability indicator in list form.
    :param id_col: ID column
    :param drift_threshold_model: threshold which the user is specifying for tagging an attribute to be drifted or not
    :param all_drift_charts_: Charts (histogram/barplot) all collated in a list format supported as per datapane objects
    :param print_report: Printing option flexibility. Default value is kept as False.
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
            legend=dict(orientation="h", x=0.5, yanchor="top", xanchor="center")
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
            :param drifted_feats: count of attributes drifted
            :param len_feats: count of attributes passed for analysis
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
            line_chart_list.append(
                line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
            )

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
                    rows=2,
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
            line_chart_list.append(
                line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
            )

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
                    rows=2,
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
                    rows=2,
                ),
                label="Drift & Stability",
            )

    elif ds_ind[0] == 0 and ds_ind[1] >= 0.5:

        for i in total_unstable_attr:
            line_chart_list.append(
                line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
            )

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
                rows=2,
            ),
            label="Drift & Stability",
        )

    else:

        for i in total_unstable_attr:
            line_chart_list.append(
                line_chart_gen_stability(df1=df_stability, df2=df_si_, col=i)
            )

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
                    rows=2,
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
                    rows=2,
                ),
                label="Drift & Stability",
            )

    if print_report:
        dp.Report(default_template[0], default_template[1], report).save(
            ends_with(master_path) + "data_drift_stability.html", open=True
        )

    return report


def anovos_report(
    master_path,
    id_col="",
    label_col="",
    corr_threshold=0.4,
    iv_threshold=0.02,
    drift_threshold_model=0.1,
    dataDict_path=".",
    metricDict_path=".",
    run_type="local",
    final_report_path=".",
):
    """
    :param master_path: Path containing the input files.
    :param id_col: ID column
    :param label_col: label column
    :param corr_threshold: Correlation threshold beyond which attributes can be categorized under correlated.
    :param iv_threshold: IV threshold beyond which attributes can be called as significant.
    :param drift_threshold_model: threshold which the user is specifying for tagging an attribute to be drifted or not
    :param dataDict_path: Data dictionary path. Default value is kept as None.
    :param metricDict_path: Metric dictionary path. Default value is kept as None.
    :param run_type: local or emr or databricks option. Default is kept as local
    :param final_report_path: Path where the report will be saved.
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
    stability_tab = ["stabilityIndex_computation", "stabilityIndex_metrics"]
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
        df_si_ = pd.read_csv(ends_with(master_path) + "stabilityIndex_computation.csv")
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
        df_si_ = pd.read_csv(ends_with(master_path) + "stabilityIndex_computation.csv")
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

    final_tabs_list = []

    for i in [tab1, tab2, tab3, tab4, tab5, tab6]:
        if i == "null_report":
            pass
        else:
            final_tabs_list.append(i)

    if run_type == "local" or "databricks":

        dp.Report(
            default_template[0],
            default_template[1],
            dp.Select(blocks=final_tabs_list, type=dp.SelectType.TABS),
        ).save(ends_with(final_report_path) + "ml_anovos_report.html", open=True)

    elif run_type == "emr":

        dp.Report(
            default_template[0],
            default_template[1],
            dp.Select(blocks=final_tabs_list, type=dp.SelectType.TABS),
        ).save("ml_anovos_report.html", open=True)

        bash_cmd = "aws s3 cp ml_anovos_report.html " + ends_with(final_report_path)
        subprocess.check_output(["bash", "-c", bash_cmd])
    else:
        raise ValueError("Invalid run_type")

    print("Report generated successfully at the specified location")
