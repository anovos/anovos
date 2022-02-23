import pyspark
import datetime
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from loguru import logger
import calendar
from anovos.shared.utils import attributeType_segregation, ends_with
from anovos.data_analyzer.stats_generator import measures_of_percentiles
from anovos.data_ingest.ts_auto_detection import (
    check_val_ind,
    ts_loop_cols_pre,
    list_ts_remove_append,
    ts_preprocess,
)
from anovos.data_transformer.datetime import (
    timeUnits_extraction,
    unix_to_timestamp,
    lagged_ts,
)

import csv
import io
import os
import re
import warnings
import subprocess
from pathlib import Path
import dateutil.parser
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np


def daypart_cat(column):
    """calculate hour buckets after adding local timezone"""
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


def ts_processed_feats(idf, col, id_col, tz):

    if idf.count() == idf.select(id_col).distinct().count():

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


def ts_eligiblity_check(spark, idf, col, id_col, opt=1, tz_offset="local"):

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


def check_val_ind(val):
    if val is None:
        return 0
    else:
        return val


def ts_loop_cols_pre(idf, id_col):
    lc = []
    ts_col = []
    for i in idf.dtypes:
        if (
            (i[1] in ["string", "object"])
            or (
                i[1] in ["long", "bigint"]
                and (
                    check_val_ind(
                        idf.select(F.max(F.length(i[0])))
                        .rdd.flatMap(lambda x: x)
                        .collect()[0]
                    )
                    > 9
                )
                and (idf.select(i[0]).distinct())
            )
        ) and i[0] != id_col:
            lc.append(i[0])
        else:
            pass

        if i[1] in ["timestamp", "date"] and i[0] != id_col:
            ts_col.append(i[0])
    return lc, ts_col


def list_ts_remove_append(l, opt):
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


def ts_analyzer(
    spark,
    idf,
    id_col,
    output_path,
    output_type="daily",
    tz_offset="local",
    run_type="local",
):

    if run_type == "local":
        local_path = output_path
    else:
        local_path = "report_stats"
    print(local_path)
    Path(local_path).mkdir(parents=True, exist_ok=True)

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    num_cols = [x for x in num_cols if x not in [id_col]]
    cat_cols = [x for x in cat_cols if x not in [id_col]]

    ts_loop_cols_post = ts_loop_cols_pre(idf, id_col)[1]
    print(ts_loop_cols_post)

    for i in ts_loop_cols_post:

        ts_processed_feat_df = ts_processed_feats(idf, i, id_col, tz_offset)
        ts_processed_feat_df.persist()

        for j in range(1, 3):
            print(i, j)
            f = ts_eligiblity_check(
                spark,
                ts_processed_feat_df,
                i,
                id_col=id_col,
                opt=j,
                tz_offset=tz_offset,
            )
            f.to_csv(
                ends_with(local_path) + "stats_" + str(i) + "_" + str(j) + ".csv",
                index=False,
            )

        for k in [num_cols, cat_cols]:
            for l in k:
                for m in [output_type]:
                    print(i, k, l, m)
                    f = ts_viz_data(
                        ts_processed_feat_df,
                        i,
                        l,
                        id_col=id_col,
                        tz_offset=tz_offset,
                        output_mode="append",
                        output_type=m,
                        n_cat=10,
                    )
                    f.to_csv(
                        ends_with(local_path) + i + "_" + l + "_" + m + ".csv",
                        index=False,
                    )

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp --recursive "
            + ends_with(local_path)
            + " "
            + ends_with(output_path)
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])
