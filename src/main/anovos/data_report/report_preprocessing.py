import subprocess
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyspark
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from anovos.data_analyzer.stats_generator import uniqueCount_computation
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.transformers import (
    attribute_binning,
    imputation_MMM,
    outlier_categories,
)
from anovos.shared.utils import (
    attributeType_segregation,
    ends_with,
    output_to_local,
    path_ak8s_modify,
)

warnings.filterwarnings("ignore")

global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
global_plot_bg_color = "rgba(0,0,0,0)"
global_paper_bg_color = "rgba(0,0,0,0)"
num_cols = []
cat_cols = []


def save_stats(
    spark,
    idf,
    master_path,
    function_name,
    reread=False,
    run_type="local",
    mlflow_config=None,
    auth_key="NA",
):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        input dataframe
    master_path
        Path to master folder under which all statistics will be saved in a csv file format.
    function_name
        Function Name for which statistics need to be saved. file name will be saved as csv
    reread
        option to reread. Default value is kept as False
    run_type
        local or emr or databricks or ak8s based on the mode of execution. Default value is kept as local
    mlflow_config
        MLflow configuration. If None, all MLflow features are disabled.
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

    local_path = (
        local_path + "/" + mlflow_config["run_id"]
        if mlflow_config is not None and mlflow_config.get("track_reports", False)
        else local_path
    )

    Path(local_path).mkdir(parents=True, exist_ok=True)

    idf.toPandas().to_csv(ends_with(local_path) + function_name + ".csv", index=False)

    if mlflow_config is not None:
        mlflow.log_artifact(local_path)

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp "
            + ends_with(local_path)
            + function_name
            + ".csv "
            + ends_with(master_path)
        )

        subprocess.check_output(["bash", "-c", bash_cmd])

    if run_type == "ak8s":
        output_path_mod = path_ak8s_modify(master_path)
        bash_cmd = (
            'azcopy cp "'
            + ends_with(local_path)
            + function_name
            + '.csv" "'
            + ends_with(output_path_mod)
            + str(auth_key)
            + '"'
        )
        subprocess.check_output(["bash", "-c", bash_cmd])

    if reread:
        odf = spark.read.csv(
            ends_with(master_path) + function_name + ".csv",
            header=True,
            inferSchema=True,
        )
        return odf


def edit_binRange(col):
    """

    Parameters
    ----------
    col
        The column which is passed as input and needs to be treated.
        The generated output will not contain any range whose value at either side is the same.

    Returns
    -------

    """
    try:
        list_col = col.split("-")
        deduped_col = list(set(list_col))
        if len(list_col) != len(deduped_col):
            return deduped_col[0]
        else:
            return col
    except Exception as e:
        logger.error(f"processing failed during edit_binRange, error {e}")
        pass


f_edit_binRange = F.udf(edit_binRange, T.StringType())


def binRange_to_binIdx(spark, col, cutoffs_path):
    """

    Parameters
    ----------
    spark
        Spark Session
    col
        The input column which is needed to by mapped with respective index
    cutoffs_path
        paths containing the range cutoffs applicable for each index

    Returns
    -------

    """
    bin_cutoffs = (
        spark.read.parquet(cutoffs_path)
        .where(F.col("attribute") == col)
        .select("parameters")
        .rdd.flatMap(lambda x: x)
        .collect()[0]
    )
    bin_ranges = []
    max_cat = len(bin_cutoffs) + 1
    for idx in range(0, max_cat):
        if idx == 0:
            bin_ranges.append("<= " + str(round(bin_cutoffs[idx], 4)))
        elif idx < (max_cat - 1):
            bin_ranges.append(
                str(round(bin_cutoffs[idx - 1], 4))
                + "-"
                + str(round(bin_cutoffs[idx], 4))
            )
        else:
            bin_ranges.append("> " + str(round(bin_cutoffs[idx - 1], 4)))
    mapping = spark.createDataFrame(
        zip(range(1, max_cat + 1), bin_ranges), schema=["bin_idx", col]
    )
    return mapping


def plot_frequency(spark, idf, col, cutoffs_path):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input dataframe which would be referred for producing the frequency charts in form of
        bar plots / histograms
    col
        Analysis column
    cutoffs_path
        Path containing the range cut offs details for the analysis column

    Returns
    -------

    """
    odf = (
        idf.groupBy(col)
        .count()
        .withColumn(
            "count_%",
            100 * (F.col("count") / F.sum("count").over(Window.partitionBy())),
        )
        .withColumn(col, f_edit_binRange(col))
    )

    if col in cat_cols:
        odf_pd = odf.orderBy("count", ascending=False).toPandas().fillna("Missing")
        odf_pd.loc[odf_pd[col] == "others", col] = "others*"

    if col in num_cols:
        mapping = binRange_to_binIdx(spark, col, cutoffs_path)
        odf_pd = (
            odf.join(mapping, col, "left_outer")
            .orderBy("bin_idx")
            .toPandas()
            .fillna("Missing")
        )

    fig = px.bar(
        odf_pd,
        x=col,
        y="count",
        text=odf_pd["count_%"].apply(lambda x: "{0:1.2f}%".format(x)),
        color_discrete_sequence=global_theme,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(title_text=str("Frequency Distribution for " + str(col.upper())))
    fig.update_xaxes(type="category")
    # fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    # plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}bar_graph.html")

    return fig


def plot_outlier(spark, idf, col, split_var=None, sample_size=500000):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input dataframe which would be referred for capturing the outliers in form of violin charts
    col
        Analysis column
    split_var
        Column which is needed. Default value is kept as None
    sample_size
        Maximum Sample size. Default value is kept as 500000

    Returns
    -------

    """
    idf_sample = idf.select(col).sample(
        False, min(1.0, float(sample_size) / idf.count()), 0
    )
    idf_sample.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    idf_imputed = imputation_MMM(spark, idf_sample)
    idf_pd = idf_imputed.toPandas()
    fig = px.violin(
        idf_pd,
        y=col,
        color=split_var,
        box=True,
        points="outliers",
        color_discrete_sequence=[global_theme_r[8], global_theme_r[4]],
    )
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    fig.update_layout(
        legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center")
    )

    return fig


def plot_eventRate(spark, idf, col, label_col, event_label, cutoffs_path):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input dataframe which would be referred for producing the frequency charts in form of bar plots / histogram
    col
        Analysis column
    label_col
        Label column
    event_label
        Event label
    cutoffs_path
        Path containing the range cut offs details for the analysis column

    Returns
    -------

    """

    odf = (
        idf.withColumn(
            label_col, F.when(F.col(label_col) == event_label, 1).otherwise(0)
        )
        .groupBy(col)
        .pivot(label_col)
        .count()
        .fillna(0, subset=["0", "1"])
        .withColumn("event_rate", 100 * (F.col("1") / (F.col("0") + F.col("1"))))
        .withColumn("attribute_name", F.lit(col))
        .withColumn(col, f_edit_binRange(col))
    )

    if col in cat_cols:
        odf_pd = odf.orderBy("event_rate", ascending=False).toPandas()
        odf_pd.loc[odf_pd[col] == "others", col] = "others*"

    if col in num_cols:
        mapping = binRange_to_binIdx(spark, col, cutoffs_path)
        odf_pd = odf.join(mapping, col, "left_outer").orderBy("bin_idx").toPandas()

    fig = px.bar(
        odf_pd,
        x=col,
        y="event_rate",
        text=odf_pd["event_rate"].apply(lambda x: "{0:1.2f}%".format(x)),
        color_discrete_sequence=global_theme,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title_text=str(
            "Event Rate Distribution for "
            + str(col.upper())
            + str(" [Target Variable : " + str(event_label) + str("]"))
        )
    )
    fig.update_xaxes(type="category")
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    # plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}feat_analysis_label.html")

    return fig


def plot_comparative_drift(spark, idf, source, col, cutoffs_path):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        Target dataframe which would be referred for producing the frequency charts in form of bar plots / histogram
    source
        Source dataframe of comparison
    col
        Analysis column
    cutoffs_path
        Path containing the range cut offs details for the analysis column

    Returns
    -------

    """
    odf = (
        idf.groupBy(col)
        .agg((F.count(col) / idf.count()).alias("countpct_target"))
        .fillna(np.nan, subset=[col])
    )

    if col in cat_cols:
        odf_pd = (
            odf.join(
                source.withColumnRenamed("p", "countpct_source").fillna(
                    np.nan, subset=[col]
                ),
                col,
                "full_outer",
            )
            .orderBy("countpct_target", ascending=False)
            .toPandas()
        )

    if col in num_cols:
        mapping = binRange_to_binIdx(spark, col, cutoffs_path)
        odf_pd = (
            odf.join(mapping, col, "left_outer")
            .fillna(np.nan, subset=["bin_idx"])
            .join(
                source.fillna(np.nan, subset=[col]).select(
                    F.col(col).alias("bin_idx"), F.col("p").alias("countpct_source")
                ),
                "bin_idx",
                "full_outer",
            )
            .orderBy("bin_idx")
            .toPandas()
        )

    odf_pd.fillna(
        {col: "Missing", "countpct_source": 0, "countpct_target": 0}, inplace=True
    )
    odf_pd["%_diff"] = (
        (odf_pd["countpct_target"] / odf_pd["countpct_source"]) - 1
    ) * 100
    fig = go.Figure()
    fig.add_bar(
        y=list(odf_pd.countpct_source.values),
        x=odf_pd[col],
        name="source",
        marker=dict(color=global_theme),
    )
    fig.update_traces(overwrite=True, marker={"opacity": 0.7})
    fig.add_bar(
        y=list(odf_pd.countpct_target.values),
        x=odf_pd[col],
        name="target",
        text=odf_pd["%_diff"].apply(lambda x: "{0:0.2f}%".format(x)),
        marker=dict(color=global_theme),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor=global_paper_bg_color,
        plot_bgcolor=global_plot_bg_color,
        showlegend=False,
    )
    fig.update_layout(
        title_text=str(
            "Drift Comparison for " + col + "<br><sup>(L->R : Source->Target)</sup>"
        )
    )
    fig.update_traces(marker=dict(color=global_theme))
    fig.update_xaxes(type="category")
    # fig.add_trace(go.Scatter(x=odf_pd[col], y=odf_pd.countpct_target.values, mode='lines+markers',
    #                        line=dict(color=px.colors.qualitative.Antique[10], width=3, dash='dot')))
    fig.update_layout(
        xaxis_tickfont_size=14,
        yaxis=dict(title="frequency", titlefont_size=16, tickfont_size=14),
    )

    return fig


def charts_to_objects(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    label_col=None,
    event_label=1,
    bin_method="equal_range",
    bin_size=10,
    coverage=1.0,
    drift_detector=False,
    outlier_charts=False,
    source_path="NA",
    master_path=".",
    stats_unique={},
    run_type="local",
    auth_key="NA",
):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input dataframe
    list_of_cols
        List of columns passed for analysis (Default value = "all")
    drop_cols
        List of columns dropped from analysis (Default value = [])
    label_col
        Label column (Default value = None)
    event_label
        Event label (Default value = 1)
    bin_method
        Binning method equal_range or equal_frequency (Default value = "equal_range")
    bin_size
        Maximum bin size categories. Default value is kept as 10
    coverage
        Maximum coverage of categories. Default value is kept as 1.0 (which is 100%)
    drift_detector
        True or False as per the availability. Default value is kept as False
    source_path
        Source data path. Default value is kept as "NA" to save intermediate data in "intermediate_data/" folder.
    master_path
        Path where the output needs to be saved, ideally the same path where the analyzed data output is also saved (Default value = ".")
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    run_type
        local or emr or databricks or ak8s run type. Default value is kept as local
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"

    Returns
    -------

    """

    global num_cols
    global cat_cols

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    if stats_unique == {}:
        remove_cols = (
            uniqueCount_computation(spark, idf, list_of_cols)
            .where(F.col("unique_values") < 2)
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
    else:
        remove_cols = (
            read_dataset(spark, **stats_unique)
            .where(F.col("unique_values") < 2)
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )

    list_of_cols = list(
        set([e for e in list_of_cols if e not in (drop_cols + remove_cols)])
    )

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")

    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))

    if cat_cols:
        idf_cleaned = outlier_categories(
            spark, idf, list_of_cols=cat_cols, coverage=coverage, max_category=bin_size
        )
    else:
        idf_cleaned = idf

    if source_path == "NA":
        source_path = "intermediate_data"

    if drift_detector:
        encoding_model_exists = True
        binned_cols = (
            spark.read.parquet(source_path + "/drift_statistics/attribute_binning")
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        to_be_binned = [e for e in num_cols if e not in binned_cols]
    else:
        encoding_model_exists = False
        binned_cols = []
        to_be_binned = num_cols

    if to_be_binned:
        idf_encoded = attribute_binning(
            spark,
            idf_cleaned,
            list_of_cols=to_be_binned,
            method_type=bin_method,
            bin_size=bin_size,
            bin_dtype="categorical",
            pre_existing_model=False,
            model_path=source_path + "/charts_to_objects",
            output_mode="append",
        )
    else:
        idf_encoded = idf_cleaned

    if binned_cols:
        idf_encoded = attribute_binning(
            spark,
            idf_encoded,
            list_of_cols=binned_cols,
            method_type=bin_method,
            bin_size=bin_size,
            bin_dtype="categorical",
            pre_existing_model=True,
            model_path=source_path + "/drift_statistics",
            output_mode="append",
        )

    cutoffs_path1 = source_path + "/charts_to_objects/attribute_binning"
    cutoffs_path2 = source_path + "/drift_statistics/attribute_binning"

    idf_encoded.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    if run_type == "local":
        local_path = master_path
    elif run_type == "databricks":
        local_path = output_to_local(master_path)
    elif run_type in ("emr", "ak8s"):
        local_path = "report_stats"
    else:
        raise ValueError("Invalid run_type")

    Path(local_path).mkdir(parents=True, exist_ok=True)

    for idx, col in enumerate(list_of_cols):

        if col in binned_cols:
            cutoffs_path = cutoffs_path2
        else:
            cutoffs_path = cutoffs_path1

        if col in cat_cols:
            f = plot_frequency(spark, idf_encoded, col, cutoffs_path)
            f.write_json(ends_with(local_path) + "freqDist_" + col)

            if label_col:
                if col != label_col:
                    f = plot_eventRate(
                        spark, idf_encoded, col, label_col, event_label, cutoffs_path
                    )
                    f.write_json(ends_with(local_path) + "eventDist_" + col)

            if drift_detector:
                try:
                    frequency_path = (
                        source_path + "/drift_statistics/frequency_counts/" + col
                    )
                    idf_source = spark.read.csv(
                        frequency_path, header=True, inferSchema=True
                    )
                    f = plot_comparative_drift(
                        spark, idf_encoded, idf_source, col, cutoffs_path
                    )
                    f.write_json(ends_with(local_path) + "drift_" + col)
                except Exception as e:
                    logger.error(f"processing failed during drift detection, error {e}")
                    pass

        if col in num_cols:
            if outlier_charts:
                f = plot_outlier(spark, idf, col, split_var=None)
                f.write_json(ends_with(local_path) + "outlier_" + col)
            f = plot_frequency(
                spark,
                idf_encoded.drop(col).withColumnRenamed(col + "_binned", col),
                col,
                cutoffs_path,
            )
            f.write_json(ends_with(local_path) + "freqDist_" + col)

            if label_col:
                if col != label_col:
                    f = plot_eventRate(
                        spark,
                        idf_encoded.drop(col).withColumnRenamed(col + "_binned", col),
                        col,
                        label_col,
                        event_label,
                        cutoffs_path,
                    )
                    f.write_json(ends_with(local_path) + "eventDist_" + col)

            if drift_detector:
                try:
                    frequency_path = (
                        source_path + "/drift_statistics/frequency_counts/" + col
                    )
                    idf_source = spark.read.csv(
                        frequency_path, header=True, inferSchema=True
                    )
                    f = plot_comparative_drift(
                        spark,
                        idf_encoded.drop(col).withColumnRenamed(col + "_binned", col),
                        idf_source,
                        col,
                        cutoffs_path,
                    )
                    f.write_json(ends_with(local_path) + "drift_" + col)
                except Exception as e:
                    logger.error(f"processing failed during drift detection, error {e}")
                    pass

    pd.DataFrame(idf.dtypes, columns=["attribute", "data_type"]).to_csv(
        ends_with(local_path) + "data_type.csv", index=False
    )

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp --recursive "
            + ends_with(local_path)
            + " "
            + ends_with(master_path)
        )
        subprocess.check_output(["bash", "-c", bash_cmd])

    if run_type == "ak8s":
        output_path_mod = path_ak8s_modify(master_path)
        bash_cmd = (
            'azcopy cp "'
            + ends_with(local_path)
            + '" "'
            + ends_with(output_path_mod)
            + str(auth_key)
            + '" --recursive=true'
        )
        subprocess.check_output(["bash", "-c", bash_cmd])
