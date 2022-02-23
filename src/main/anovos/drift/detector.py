# coding=utf-8
from __future__ import division, print_function

import numpy as np
import pandas as pd
import pyspark
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from scipy.stats import variation
import sympy as sp

from anovos.data_ingest.data_ingest import concatenate_dataset
from anovos.data_transformer.transformers import attribute_binning
from anovos.shared.utils import attributeType_segregation
from .distances import hellinger, psi, js_divergence, ks
from .validations import check_distance_method, check_list_of_columns


@check_distance_method
@check_list_of_columns
def statistics(
    spark: SparkSession,
    idf_target: DataFrame,
    idf_source: DataFrame,
    *,
    list_of_cols: list = "all",
    drop_cols: list = None,
    method_type: str = "PSI",
    bin_method: str = "equal_range",
    bin_size: int = 10,
    threshold: float = 0.1,
    pre_existing_source: bool = False,
    source_path: str = "NA",
    model_directory: str = "drift_statistics",
    print_impact: bool = False,
):
    """
    :param spark: Spark Session
    :param idf_target: Input Dataframe
    :param idf_source: Baseline/Source Dataframe. This argument is ignored if pre_existing_source is True.
    :param list_of_cols: List of columns to check drift e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all (non-array) columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param method_type: "PSI", "JSD", "HD", "KS","all".
                   "all" can be passed to calculate all drift metrics.
                    One or more methods can be passed in a form of list or string where different metrics are separated
                    by pipe delimiter “|” e.g. ["PSI", "JSD"] or "PSI|JSD"
    :param bin_method: "equal_frequency", "equal_range".
                        In "equal_range" method, each bin is of equal size/width and in "equal_frequency", each bin
                        has equal no. of rows, though the width of bins may vary.
    :param bin_size: Number of bins for creating histogram
    :param threshold: A column is flagged if any drift metric is above the threshold.
    :param pre_existing_source: Boolean argument – True or False. True if the drift_statistics folder (binning model &
                                frequency counts for each attribute) exists already, False Otherwise.
    :param source_path: If pre_existing_source is False, this argument can be used for saving the drift_statistics folder.
                        The drift_statistics folder will have attribute_binning (binning model) & frequency_counts sub-folders.
                        If pre_existing_source is True, this argument is path for referring the drift_statistics folder.
                        Default "NA" for temporarily saving source dataset attribute_binning folder.
    :param model_directory: If pre_existing_source is False, this argument can be used for saving the drift stats to folder.
                        The default drift statics directory is drift_statistics folder will have attribute_binning
                        If pre_existing_source is True, this argument is model_directory for referring the drift statistics dir.
                        Default "drift_statistics" for temporarily saving source dataset attribute_binning folder.
    :param print_impact: True, False
    :return: Output Dataframe [attribute, *metric, flagged]
             Number of columns will be dependent on method argument. There will be one column for each drift method/metric.
    """
    drop_cols = drop_cols or []
    num_cols = attributeType_segregation(idf_target.select(list_of_cols))[0]

    if not pre_existing_source:
        source_bin = attribute_binning(
            spark,
            idf_source,
            list_of_cols=num_cols,
            method_type=bin_method,
            bin_size=bin_size,
            pre_existing_model=False,
            model_path=source_path + "/" + model_directory,
        )
        source_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    target_bin = attribute_binning(
        spark,
        idf_target,
        list_of_cols=num_cols,
        method_type=bin_method,
        bin_size=bin_size,
        pre_existing_model=True,
        model_path=source_path + "/" + model_directory,
    )

    target_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    result = {"attribute": [], "flagged": []}

    for method in method_type:
        result[method] = []

    for i in list_of_cols:
        if pre_existing_source:
            x = spark.read.csv(
                source_path + "/" + model_directory + "/frequency_counts/" + i,
                header=True,
                inferSchema=True,
            )
        else:
            x = (
                source_bin.groupBy(i)
                .agg((F.count(i) / idf_source.count()).alias("p"))
                .fillna(-1)
            )
            x.coalesce(1).write.csv(
                source_path + "/" + model_directory + "/frequency_counts/" + i,
                header=True,
                mode="overwrite",
            )

        y = (
            target_bin.groupBy(i)
            .agg((F.count(i) / idf_target.count()).alias("q"))
            .fillna(-1)
        )

        xy = (
            x.join(y, i, "full_outer")
            .fillna(0.0001, subset=["p", "q"])
            .replace(0, 0.0001)
            .orderBy(i)
        )
        p = np.array(xy.select("p").rdd.flatMap(lambda x: x).collect())
        q = np.array(xy.select("q").rdd.flatMap(lambda x: x).collect())

        result["attribute"].append(i)
        counter = 0

        for idx, method in enumerate(method_type):
            drift_function = {
                "PSI": psi,
                "JSD": js_divergence,
                "HD": hellinger,
                "KS": ks,
            }
            metric = float(round(drift_function[method](p, q), 4))
            result[method].append(metric)
            if counter == 0:
                if metric > threshold:
                    result["flagged"].append(1)
                    counter = 1
            if (idx == (len(method_type) - 1)) & (counter == 0):
                result["flagged"].append(0)

    odf = (
        spark.createDataFrame(
            pd.DataFrame.from_dict(result, orient="index").transpose()
        )
        .select(["attribute"] + method_type + ["flagged"])
        .orderBy(F.desc("flagged"))
    )

    if print_impact:
        logger.info("All Attributes:")
        odf.show(len(list_of_cols))
        logger.info("Attributes meeting Data Drift threshold:")
        drift = odf.where(F.col("flagged") == 1)
        drift.show(drift.count())

    return odf


def stability_index_computation(
    spark,
    *idfs,
    list_of_cols="all",
    drop_cols=[],
    metric_weightages={"mean": 0.5, "stddev": 0.3, "kurtosis": 0.2},
    existing_metric_path="",
    appended_metric_path="",
    threshold=1,
    print_impact=False,
):
    """
    :param spark: Spark Session
    :param idfs: Variable number of input dataframes
    :param list_of_cols: List of numerical columns to check stability e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param metric_weightages: Takes input in dictionary format with keys being the metric name - "mean","stdev","kurtosis"
                              and value being the weightage of the metric (between 0 and 1). Sum of all weightages must be 1.
    :param existing_metric_path: This argument is path for referring pre-existing metrics of historical datasets and is
                                 of schema [idx, attribute, mean, stdev, kurtosis].
                                 idx is index number of historical datasets assigned in chronological order.
    :param appended_metric_path: This argument is path for saving input dataframes metrics after appending to the
                                 historical datasets' metrics.
    :param threshold: A column is flagged if the stability index is below the threshold, which varies between 0 to 4.
                      The following criteria can be used to classifiy stability_index (SI): very unstable: 0≤SI<1,
                      unstable: 1≤SI<2, marginally stable: 2≤SI<3, stable: 3≤SI<3.5 and very stable: 3.5≤SI≤4.
    :param print_impact: True, False
    :return: Dataframe [attribute, mean_si, stddev_si, kurtosis_si, mean_cv, stddev_cv, kurtosis_cv, stability_index].
             *_cv is coefficient of variation for each metric. *_si is stability index for each metric.
             stability_index is net weighted stability index based on the individual metrics' stability index.
    """

    num_cols = attributeType_segregation(idfs[0])[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")

    if (
        round(
            metric_weightages.get("mean", 0)
            + metric_weightages.get("stddev", 0)
            + metric_weightages.get("kurtosis", 0),
            3,
        )
        != 1
    ):
        raise ValueError(
            "Invalid input for metric weightages. Either metric name is incorrect or sum of metric weightages is not 1.0"
        )

    if existing_metric_path:
        existing_metric_df = spark.read.csv(
            existing_metric_path, header=True, inferSchema=True
        )
        dfs_count = existing_metric_df.select(F.max(F.col("idx"))).first()[0]
    else:
        schema = T.StructType(
            [
                T.StructField("idx", T.IntegerType(), True),
                T.StructField("attribute", T.StringType(), True),
                T.StructField("mean", T.DoubleType(), True),
                T.StructField("stddev", T.DoubleType(), True),
                T.StructField("kurtosis", T.DoubleType(), True),
            ]
        )
        existing_metric_df = spark.sparkContext.emptyRDD().toDF(schema)
        dfs_count = 0

    metric_ls = []
    for idf in idfs:
        for i in list_of_cols:
            mean, stddev, kurtosis = idf.select(
                F.mean(i), F.stddev(i), F.kurtosis(i)
            ).first()
            metric_ls.append(
                [dfs_count + 1, i, mean, stddev, kurtosis + 3.0 if kurtosis else None]
            )
        dfs_count += 1

    new_metric_df = spark.createDataFrame(
        metric_ls, schema=("idx", "attribute", "mean", "stddev", "kurtosis")
    )
    appended_metric_df = concatenate_dataset(existing_metric_df, new_metric_df)

    if appended_metric_path:
        appended_metric_df.coalesce(1).write.csv(
            appended_metric_path, header=True, mode="overwrite"
        )

    result = []
    for i in list_of_cols:
        i_output = [i]
        for metric in ["mean", "stddev", "kurtosis"]:
            metric_stats = (
                appended_metric_df.where(F.col("attribute") == i)
                .orderBy("idx")
                .select(metric)
                .fillna(np.nan)
                .rdd.flatMap(list)
                .collect()
            )
            metric_cv = round(float(variation([a for a in metric_stats])), 4) or None
            i_output.append(metric_cv)
        result.append(i_output)

    schema = T.StructType(
        [
            T.StructField("attribute", T.StringType(), True),
            T.StructField("mean_cv", T.FloatType(), True),
            T.StructField("stddev_cv", T.FloatType(), True),
            T.StructField("kurtosis_cv", T.FloatType(), True),
        ]
    )

    odf = spark.createDataFrame(result, schema=schema)

    def score_cv(cv, thresholds=[0.03, 0.1, 0.2, 0.5]):
        if cv is None:
            return None
        else:
            cv = abs(cv)
            stability_index = [4, 3, 2, 1, 0]
            for i, thresh in enumerate(thresholds):
                if cv < thresh:
                    return stability_index[i]
            return stability_index[-1]

    f_score_cv = F.udf(score_cv, T.IntegerType())

    odf = (
        odf.replace(np.nan, None)
        .withColumn("mean_si", f_score_cv(F.col("mean_cv")))
        .withColumn("stddev_si", f_score_cv(F.col("stddev_cv")))
        .withColumn("kurtosis_si", f_score_cv(F.col("kurtosis_cv")))
        .withColumn(
            "stability_index",
            F.round(
                (
                    F.col("mean_si") * metric_weightages.get("mean", 0)
                    + F.col("stddev_si") * metric_weightages.get("stddev", 0)
                    + F.col("kurtosis_si") * metric_weightages.get("kurtosis", 0)
                ),
                4,
            ),
        )
        .withColumn(
            "flagged",
            F.when(
                (F.col("stability_index") < threshold)
                | (F.col("stability_index").isNull()),
                1,
            ).otherwise(0),
        )
    )

    if print_impact:
        logger.info("All Attributes:")
        odf.show(len(list_of_cols))
        logger.info("Potential Unstable Attributes:")
        unstable = odf.where(F.col("flagged") == 1)
        unstable.show(unstable.count())

    return odf


def feature_stability_estimation(
    spark,
    attribute_stats,
    attribute_transformation,
    metric_weightages={"mean": 0.5, "stddev": 0.3, "kurtosis": 0.2},
    threshold=1,
    print_impact=False,
):
    """
    :param spark: Spark Session
    :param attribute_stats: Spark dataframe. The intermediate dataframe saved by running function
                            stabilityIndex_computation with schema [idx, attribute, mean, stddev, kurtosis].
                            It should contain all the attributes used in argument attribute_transformation.

    :param attribute_transformation: Takes input in dictionary format: each key-value combination represents one
                                     new feature. Each key is a string containing all the attributes involved in
                                     the new feature seperated by '|'. Each value is the transformation of the
                                     attributes in string. For example, {'X|Y|Z': 'X**2+Y/Z', 'A': 'log(A)'}
    :param metric_weightages: Takes input in dictionary format with keys being the metric name - "mean","stdev","kurtosis"
                              and value being the weightage of the metric (between 0 and 1). Sum of all weightages must be 1.
    :param threshold: A column is flagged if the stability index is below the threshold, which varies between 0 to 4.
                      The following criteria can be used to classifiy stability_index (SI): very unstable: 0≤SI<1,
                      unstable: 1≤SI<2, marginally stable: 2≤SI<3, stable: 3≤SI<3.5 and very stable: 3.5≤SI≤4.
    :param print_impact: True, False
    :return: Dataframe [feature_formula, mean_cv, stddev_cv, mean_si, stddev_si, stability_index_lower_bound,
             stability_index_upper_bound, flagged_lower, flagged_upper].
             *_cv is coefficient of variation for each metric. *_si is stability index for each metric.
             stability_index_lower_bound and stability_index_upper_bound form a range for estimated stability index.
             flagged_lower and flagged_upper indicate whether the feature is potentially unstable based on the lower
             and uppder bounds for stability index .
    """

    def stats_estimation(attributes, transformation, mean, stddev):
        attribute_means = list(zip(attributes, mean))
        first_dev = []
        second_dev = []
        est_mean = 0
        est_var = 0
        for attr, s in zip(attributes, stddev):
            first_dev = sp.diff(transformation, attr)
            second_dev = sp.diff(transformation, attr, 2)

            est_mean += s ** 2 * second_dev.subs(attribute_means) / 2
            est_var += s ** 2 * (first_dev.subs(attribute_means)) ** 2

        transformation = sp.parse_expr(transformation)
        est_mean += transformation.subs(attribute_means)

        return [float(est_mean), float(est_var)]

    f_stats_estimation = F.udf(stats_estimation, T.ArrayType(T.FloatType()))

    index = (
        attribute_stats.select("idx")
        .distinct()
        .orderBy("idx")
        .rdd.flatMap(list)
        .collect()
    )
    attribute_names = list(attribute_transformation.keys())
    transformations = list(attribute_transformation.values())

    feature_metric = []
    for attributes, transformation in zip(attribute_names, transformations):
        attributes = [x.strip() for x in attributes.split("|")]
        for idx in index:
            attr_mean_list, attr_stddev_list = [], []
            for attr in attributes:
                df_temp = attribute_stats.where(
                    (F.col("idx") == idx) & (F.col("attribute") == attr)
                )
                if df_temp.count() == 0:
                    raise TypeError(
                        "Invalid input for attribute_stats: all involved attributes must have available statistics across all time periods (idx)"
                    )
                attr_mean_list.append(
                    df_temp.select("mean").rdd.flatMap(lambda x: x).collect()[0]
                )
                attr_stddev_list.append(
                    df_temp.select("stddev").rdd.flatMap(lambda x: x).collect()[0]
                )
            feature_metric.append(
                [idx, transformation, attributes, attr_mean_list, attr_stddev_list]
            )

    schema = T.StructType(
        [
            T.StructField("idx", T.IntegerType(), True),
            T.StructField("transformation", T.StringType(), True),
            T.StructField("attributes", T.ArrayType(T.StringType()), True),
            T.StructField("attr_mean_list", T.ArrayType(T.FloatType()), True),
            T.StructField("attr_stddev_list", T.ArrayType(T.FloatType()), True),
        ]
    )

    df_feature_metric = (
        spark.createDataFrame(feature_metric, schema=schema)
        .withColumn(
            "est_feature_stats",
            f_stats_estimation(
                "attributes", "transformation", "attr_mean_list", "attr_stddev_list"
            ),
        )
        .withColumn("est_feature_mean", F.col("est_feature_stats")[0])
        .withColumn("est_feature_stddev", F.sqrt(F.col("est_feature_stats")[1]))
        .select(
            "idx",
            "attributes",
            "transformation",
            "est_feature_mean",
            "est_feature_stddev",
        )
    )

    output = []
    for idx, i in enumerate(transformations):
        i_output = [i]
        for metric in ["est_feature_mean", "est_feature_stddev"]:
            metric_stats = (
                df_feature_metric.where(F.col("transformation") == i)
                .orderBy("idx")
                .select(metric)
                .fillna(np.nan)
                .rdd.flatMap(list)
                .collect()
            )
            metric_cv = round(float(variation([a for a in metric_stats])), 4) or None
            i_output.append(metric_cv)
        output.append(i_output)

    schema = T.StructType(
        [
            T.StructField("feature_formula", T.StringType(), True),
            T.StructField("mean_cv", T.FloatType(), True),
            T.StructField("stddev_cv", T.FloatType(), True),
        ]
    )

    odf = spark.createDataFrame(output, schema=schema)

    def score_cv(cv, thresholds=[0.03, 0.1, 0.2, 0.5]):
        if cv is None:
            return None
        else:
            cv = abs(cv)
            stability_index = [4, 3, 2, 1, 0]
            for i, thresh in enumerate(thresholds):
                if cv < thresh:
                    return stability_index[i]
            return stability_index[-1]

    f_score_cv = F.udf(score_cv, T.IntegerType())

    odf = (
        odf.replace(np.nan, None)
        .withColumn("mean_si", f_score_cv(F.col("mean_cv")))
        .withColumn("stddev_si", f_score_cv(F.col("stddev_cv")))
        .withColumn(
            "stability_index_lower_bound",
            F.round(
                F.col("mean_si") * metric_weightages.get("mean", 0)
                + F.col("stddev_si") * metric_weightages.get("stddev", 0),
                4,
            ),
        )
        .withColumn(
            "stability_index_upper_bound",
            F.round(
                F.col("stability_index_lower_bound")
                + 4 * metric_weightages.get("kurtosis", 0),
                4,
            ),
        )
        .withColumn(
            "flagged_lower",
            F.when(
                (F.col("stability_index_lower_bound") < threshold)
                | (F.col("stability_index_lower_bound").isNull()),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "flagged_upper",
            F.when(
                (F.col("stability_index_upper_bound") < threshold)
                | (F.col("stability_index_upper_bound").isNull()),
                1,
            ).otherwise(0),
        )
    )

    if print_impact:
        logger.info("All Features:")
        odf.show(len(attribute_names), False)
        logger.info(
            "Potential Unstable Features Identified by Both Lower and Upper Bounds:"
        )
        unstable = odf.where(F.col("flagged_upper") == 1)
        unstable.show(unstable.count())

    return odf
