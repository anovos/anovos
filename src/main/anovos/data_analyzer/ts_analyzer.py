# coding=utf-8

"""This module generates the intermediate output specific to the inspection of Time series analysis.

As a part of generation of final output, there are various functions created such as -

- ts_processed_feats
- ts_eligiblity_check
- ts_viz_data
- ts_analyzer
- daypart_cat

Respective functions have sections containing the detailed definition of the parameters used for computing.

"""

import calendar
from anovos.shared.utils import (
    attributeType_segregation,
    ends_with,
    output_to_local,
    path_ak8s_modify,
)
from anovos.data_analyzer.stats_generator import measures_of_percentiles
from anovos.data_ingest.ts_auto_detection import ts_preprocess
from anovos.data_transformer.datetime import (
    timeUnits_extraction,
    unix_to_timestamp,
    lagged_ts,
)

import csv
import datetime
import io
import os
import re
import subprocess
import warnings
from pathlib import Path

import dateutil.parser
import numpy as np
import pandas as pd
import pyspark
from loguru import logger
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from statsmodels.tsa.seasonal import seasonal_decompose


def daypart_cat(column):

    """
    This functioin helps to convert the input hour part into the respective day parts. The different dayparts are Early Hours, Work Hours, Late Hours, Commuting Hours, Other Hours based on the hour value.

    Parameters
    ----------

    column
        Reads the column containing the hour part and converts into respective day part

    Returns
    -------
    String
    """

    # calculate hour buckets after adding local timezone

    if column is None:
        return "Missing_NA"
    elif (column >= 4) and (column < 7):
        return "early_hours"
    elif (column >= 10) and (column < 17):
        return "work_hours"
    elif (column >= 23) or (column < 4):
        return "late_hours"
    elif ((column >= 7) and (column < 10)) or ((column >= 17) and (column < 20)):
        return "commuting_hours"
    else:
        return "other_hours"


f_daypart_cat = F.udf(daypart_cat, T.StringType())


def ts_processed_feats(idf, col, id_col, tz, cnt_row, cnt_unique_id):

    """
    This function helps to extract time units from the input dataframe on a processed column being timestamp / date.

    Parameters
    ----------

    idf
        Input dataframe
    col
        Column belonging to timestamp / date
    id_col
        ID column
    tz
        Timezone offset
    cnt_row
        Count of rows present in the Input dataframe
    cnt_unique_id
        Count of unique records present in the Input dataframe

    Returns
    -------
    DataFrame
    """

    if cnt_row == cnt_unique_id:

        odf = (
            timeUnits_extraction(
                idf,
                col,
                "all",
                output_mode="append",
            )
            .withColumn("yyyymmdd_col", F.to_date(col))
            .orderBy("yyyymmdd_col")
            .withColumn("daypart_cat", f_daypart_cat(F.col(col + "_hour")))
            .withColumn(
                "week_cat",
                F.when(F.col(col + "_dayofweek") > 5, F.lit("weekend")).otherwise(
                    "weekday"
                ),
            )
            .withColumnRenamed(col + "_dayofweek", "dow")
        )

        return odf

    else:

        odf = (
            timeUnits_extraction(
                idf,
                col,
                "all",
                output_mode="append",
            )
            .withColumn("yyyymmdd_col", F.to_date(col))
            .orderBy(id_col, "yyyymmdd_col")
            .withColumn("daypart_cat", f_daypart_cat(F.col(col + "_hour")))
            .withColumn(
                "week_cat",
                F.when(F.col(col + "_dayofweek") > 5, F.lit("weekend")).otherwise(
                    "weekday"
                ),
            )
            .withColumnRenamed(col + "_dayofweek", "dow")
        )

        return odf


def ts_eligiblity_check(spark, idf, id_col, opt=1, tz_offset="local"):

    """
    This function helps to extract various metrics which can help to understand the nature of timestamp / date column for a given dataset.

    Parameters
    ----------

    spark
        Spark session
    idf
        Input dataframe
    id_col
        ID Column
    opt
        Option to choose between [1,2]. 1 is kept as default. Based on the user input, the specific aggregation of data will happen.
    tz_offset
        Timezone offset (Option to chose between options like Local, GMT, UTC, etc.). Default option is set as "Local".

    Returns
    -------
    DataFrame
    """

    lagged_df = lagged_ts(
        idf.select("yyyymmdd_col").distinct().orderBy("yyyymmdd_col"),
        "yyyymmdd_col",
        lag=1,
        tsdiff_unit="days",
        output_mode="append",
    ).orderBy("yyyymmdd_col")

    diff_lagged_df = list(
        np.around(
            lagged_df.withColumn(
                "daydiff", F.datediff("yyyymmdd_col", "yyyymmdd_col_lag1")
            )
            .where(F.col("daydiff").isNotNull())
            .groupBy()
            .agg(
                F.mean("daydiff").alias("mean"),
                F.variance("daydiff").alias("variance"),
                F.stddev("daydiff").alias("stdev"),
            )
            .withColumn("coef_of_var_lag", F.col("stdev") / F.col("mean"))
            .rdd.flatMap(lambda x: x)
            .collect(),
            3,
        )
    )

    p1 = measures_of_percentiles(
        spark,
        idf.groupBy(id_col).agg(F.countDistinct("yyyymmdd_col").alias("id_date_pair")),
        list_of_cols="id_date_pair",
    )
    p2 = measures_of_percentiles(
        spark,
        idf.groupBy("yyyymmdd_col").agg(F.countDistinct(id_col).alias("date_id_pair")),
        list_of_cols="date_id_pair",
    )

    if opt == 1:

        odf = p1.union(p2).toPandas()

        return odf

    else:

        odf = idf
        m = (
            odf.groupBy("yyyymmdd_col")
            .count()
            .orderBy("count", ascending=False)
            .collect()
        )
        mode = str(m[0][0]) + " [" + str(m[0][1]) + "]"
        missing_vals = odf.where(F.col("yyyymmdd_col").isNull()).count()
        odf = (
            odf.groupBy()
            .agg(
                F.countDistinct("yyyymmdd_col").alias("count_unique_dates"),
                F.min("yyyymmdd_col").alias("min_date"),
                F.max("yyyymmdd_col").alias("max_date"),
            )
            .withColumn("modal_date", F.lit(mode))
            .withColumn("date_diff", F.datediff("max_date", "min_date"))
            .withColumn("missing_date", F.lit(missing_vals))
            .withColumn("mean", F.lit(diff_lagged_df[0]))
            .withColumn("variance", F.lit(diff_lagged_df[1]))
            .withColumn("stdev", F.lit(diff_lagged_df[2]))
            .withColumn("cov", F.lit(diff_lagged_df[3]))
            .toPandas()
        )

        return odf


def ts_viz_data(
    idf,
    x_col,
    y_col,
    id_col,
    tz_offset="local",
    output_mode="append",
    output_type="daily",
    n_cat=10,
):

    """

    This function helps to produce the processed dataframe with the relevant aggregation at the time frequency chosen for a given column as seen against the timestamp / date column.

    Parameters
    ----------

    idf
        Input Dataframe
    x_col
        Timestamp / Date column as set in the X-Axis
    y_col
        Numerical & Categorical column as set in the Y-Axis
    id_col
        ID Column
    tz_offset
        Timezone offset (Option to chose between options like Local, GMT, UTC, etc.). Default option is set as "Local".
    output_mode
        Option to choose between Append or Replace. If the option Append is selected, the column names are Appended by "_ts" else it's replaced by the original column name
    output_type
        Option to choose between "Daily" or "Weekly" or "Hourly". Daily is chosen as default. If "Daily" is selected as the output type, the daily view is populated ; If it's "Hourly", the view is shown at a Day part level. However, if it's "Weekly", then the display it per individual week days (1-7) as captured.
    n_cat
        For categorical columns whose cardinality is beyond N, the Top N categories are chosen, beyond which the categories are grouped as Others.

    Returns
    -------
    DataFrame
    """

    y_col_org = y_col
    y_col = y_col.replace("-", "_")
    idf = idf.withColumnRenamed(y_col_org, y_col)

    for i in idf.dtypes:

        if y_col == i[0]:
            y_col_dtype = i[1]

    if y_col_dtype == "string":

        top_cat = list(
            idf.groupBy(y_col)
            .count()
            .orderBy("count", ascending=False)
            .limit(int(n_cat))
            .select(y_col)
            .toPandas()[y_col]
            .values
        )
        idf = idf.withColumn(
            y_col,
            F.when(F.col(y_col).isin(top_cat), F.col(y_col)).otherwise(F.lit("Others")),
        )

        if output_type == "daily":

            odf = (
                idf.groupBy(y_col, "yyyymmdd_col")
                .agg(F.count(y_col).alias("count"))
                .orderBy("yyyymmdd_col")
                .withColumnRenamed("yyyymmdd_col", x_col)
                .withColumnRenamed(y_col, y_col_org)
                .toPandas()
            )

        elif output_type == "hourly":

            odf = (
                idf.groupBy(y_col, "daypart_cat")
                .agg(F.count(y_col).alias("count"))
                .orderBy("daypart_cat")
                .withColumnRenamed(y_col, y_col_org)
                .toPandas()
            )

        elif output_type == "weekly":

            odf = (
                idf.groupBy(y_col, "dow")
                .agg(F.count(y_col).alias("count"))
                .orderBy("dow")
                .withColumnRenamed(y_col, y_col_org)
                .toPandas()
            )

        return odf

    else:

        if output_type == "daily":

            odf = (
                idf.groupBy("yyyymmdd_col")
                .agg(
                    F.min(y_col).alias("min"),
                    F.max(y_col).alias("max"),
                    F.mean(y_col).alias("mean"),
                    F.expr("percentile(" + y_col + ", array(0.5))")[0].alias("median"),
                )
                .orderBy("yyyymmdd_col")
                .withColumnRenamed("yyyymmdd_col", x_col)
                .withColumnRenamed(y_col, y_col_org)
                .toPandas()
            )

        elif output_type == "hourly":

            odf = (
                idf.groupBy("daypart_cat")
                .agg(
                    F.min(y_col).alias("min"),
                    F.max(y_col).alias("max"),
                    F.mean(y_col).alias("mean"),
                    F.expr("percentile(" + y_col + ", array(0.5))")[0].alias("median"),
                )
                .orderBy("daypart_cat")
                .withColumnRenamed(y_col, y_col_org)
                .toPandas()
            )

        elif output_type == "weekly":

            odf = (
                idf.groupBy("dow")
                .agg(
                    F.min(y_col).alias("min"),
                    F.max(y_col).alias("max"),
                    F.mean(y_col).alias("mean"),
                    F.expr("percentile(" + y_col + ", array(0.5))")[0].alias("median"),
                )
                .orderBy("dow")
                .withColumnRenamed(y_col, y_col_org)
                .toPandas()
            )

        return odf


def ts_analyzer(
    spark,
    idf,
    id_col,
    max_days,
    output_path,
    output_type="daily",
    tz_offset="local",
    run_type="local",
    auth_key="NA",
):

    """

    This function helps to produce the processed output in an aggregate form considering the input dataframe with processed timestamp / date column. The aggregation happens across Mean, Median, Min & Max for the Numerical / Categorical column.

    Parameters
    ----------

    spark
        Spark session
    idf
        Input Dataframe
    id_col
        ID Column
    max_days
        Max days upto which the data will be aggregated. If we've a dataset containing a timestamp / date field with very high number of unique dates (Let's say beyond 20 years worth of daily data), a maximum days value chosen basis which the latest output is displayed.
    output_path
        Output path where the intermediate data is going to be saved
    output_type
        Option to choose between "Daily" or "Weekly" or "Hourly". Daily is chosen as default. If "Daily" is selected as the output type, the daily view is populated ; If it's "Hourly", the view is shown at a Day part level. However, if it's "Weekly", then the display it per individual week days (1-7) as captured.
    tz_offset
        Timezone offset (Option to chose between options like Local, GMT, UTC, etc.). Default option is set as "Local".
    run_type
        Option to choose between run type "local" or "emr" or "databricks" or "ak8s" basis the user flexibility. Default option is set as "Local".
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"

    Returns
    -------
    Output[CSV]
    """

    if run_type == "local":
        local_path = output_path
    elif run_type == "databricks":
        local_path = output_to_local(output_path)
    elif run_type in ("emr", "ak8s"):
        local_path = "report_stats"
    else:
        raise ValueError("Invalid run_type")

    Path(local_path).mkdir(parents=True, exist_ok=True)

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    num_cols = [x for x in num_cols if x not in [id_col]]
    cat_cols = [x for x in cat_cols if x not in [id_col]]

    ts_loop_cols_post = [x[0] for x in idf.dtypes if x[1] in ["timestamp", "date"]]

    cnt_row = idf.count()
    cnt_unique_id = idf.select(id_col).distinct().count()

    for i in ts_loop_cols_post:

        ts_processed_feat_df = ts_processed_feats(
            idf, i, id_col, tz_offset, cnt_row, cnt_unique_id
        )
        ts_processed_feat_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

        # for j in range(1, 3):
        #     f = ts_eligiblity_check(
        #         spark,
        #         ts_processed_feat_df,
        #         id_col=id_col,
        #         opt=j,
        #         tz_offset=tz_offset,
        #     )
        #     f.to_csv(
        #         ends_with(local_path) + "stats_" + str(i) + "_" + str(j) + ".csv",
        #         index=False,
        #     )

        f1 = ts_eligiblity_check(
            spark, ts_processed_feat_df, id_col=id_col, opt=1, tz_offset=tz_offset
        )
        f1.to_csv(
            ends_with(local_path) + "stats_" + str(i) + "_" + str(1) + ".csv",
            index=False,
        )

        f2 = ts_eligiblity_check(
            spark, ts_processed_feat_df, id_col=id_col, opt=2, tz_offset=tz_offset
        )
        f2.to_csv(
            ends_with(local_path) + "stats_" + str(i) + "_" + str(2) + ".csv",
            index=False,
        )

        for k in [num_cols, cat_cols]:
            for l in k:
                for m in [output_type]:
                    f = (
                        ts_viz_data(
                            ts_processed_feat_df,
                            i,
                            l,
                            id_col=id_col,
                            tz_offset=tz_offset,
                            output_mode="append",
                            output_type=m,
                            n_cat=10,
                        )
                        .tail(int(max_days))
                        .dropna()
                    )
                    f.to_csv(
                        ends_with(local_path) + i + "_" + l + "_" + m + ".csv",
                        index=False,
                    )
        ts_processed_feat_df.unpersist()

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp --recursive "
            + ends_with(local_path)
            + " "
            + ends_with(output_path)
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])

    if run_type == "ak8s":
        output_path_mod = path_ak8s_modify(output_path)
        bash_cmd = (
            'azcopy cp "'
            + ends_with(local_path)
            + '" "'
            + ends_with(output_path_mod)
            + str(auth_key)
            + '" --recursive=true '
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])
