# coding=utf-8
import itertools
import math

import pyspark
from phik.phik import spark_phik_matrix_from_hist2d_dict
from popmon.analysis.hist_numpy import get_2dgrid
from pyspark.sql import Window
from pyspark.sql import functions as F
from varclushi import VarClusHi

from anovos.data_analyzer.stats_generator import uniqueCount_computation
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.transformers import (
    attribute_binning,
    monotonic_binning,
    cat_to_num_unsupervised,
    imputation_MMM,
)
from anovos.shared.utils import attributeType_segregation


def correlation_matrix(
    spark, idf, list_of_cols="all", drop_cols=[], stats_unique={}, print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: Dataframe [attribute,*col_names]
             Correlation between attribute X and Y can be found at an intersection of
             a) row with value X in ‘attribute’ column and column ‘Y’, or
             b) row with value Y in ‘attribute’ column and column ‘X’.
    """

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

    combis = [list(c) for c in itertools.combinations_with_replacement(list_of_cols, 2)]
    hists = idf.select(list_of_cols).pm_make_histograms(combis)
    grids = {k: get_2dgrid(h) for k, h in hists.items()}
    odf_pd = spark_phik_matrix_from_hist2d_dict(spark.sparkContext, grids)
    odf_pd["attribute"] = odf_pd.index
    list_of_cols.sort()
    odf = (
        spark.createDataFrame(odf_pd)
        .select(["attribute"] + list_of_cols)
        .orderBy("attribute")
    )

    if print_impact:
        odf.show(odf.count())

    return odf


def variable_clustering(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    sample_size=100000,
    stats_unique={},
    stats_mode={},
    print_impact=False,
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param sample_size: Maximum sample size (in terms of number of rows) taken for the computation.
                        Sample dataset is extracted using random sampling.
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param stats_mode: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                       to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
                       mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param print_impact: True, False
    :return: Dataframe [Cluster, Attribute, RS_Ratio]
             Attributes similar to each other are grouped together with the same cluster id.
             Attribute with the lowest (1 — RS_Ratio) can be chosen as a representative of the cluster.
    """

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")

    idf_sample = idf.sample(False, min(1.0, float(sample_size) / idf.count()), 0)
    idf_sample.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    if stats_unique == {}:
        remove_cols = (
            uniqueCount_computation(spark, idf_sample, list_of_cols)
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

    list_of_cols = [e for e in list_of_cols if e not in remove_cols]
    idf_sample = idf_sample.select(list_of_cols)
    num_cols, cat_cols, other_cols = attributeType_segregation(idf_sample)

    for i in idf_sample.dtypes:
        if i[1].startswith("decimal"):
            idf_sample = idf_sample.withColumn(i[0], F.col(i[0]).cast("double"))
    idf_encoded = cat_to_num_unsupervised(
        spark, idf_sample, list_of_cols=cat_cols, method_type=1
    )
    idf_imputed = imputation_MMM(spark, idf_encoded, stats_mode=stats_mode)
    idf_imputed.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    idf_sample.unpersist()
    idf_pd = idf_imputed.toPandas()
    vc = VarClusHi(idf_pd, maxeigval2=1, maxclus=None)
    vc.varclus()
    odf_pd = vc.rsquare
    odf = spark.createDataFrame(odf_pd).select(
        "Cluster",
        F.col("Variable").alias("Attribute"),
        F.round(F.col("RS_Ratio"), 4).alias("RS_Ratio"),
    )
    if print_impact:
        odf.show(odf.count())
    return odf


def IV_calculation(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    label_col="label",
    event_label=1,
    encoding_configs={
        "bin_method": "equal_frequency",
        "bin_size": 10,
        "monotonicity_check": 0,
    },
    print_impact=False,
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param label_col: Label/Target column
    :param event_label: Value of (positive) event (i.e label 1)
    :param encoding_configs: Takes input in dictionary format. Default {} i.e. empty dict means no encoding is required.
                             In case numerical columns are present and encoding is required, following keys shall be
                             provided - "bin_size" i.e. no. of bins for converting the numerical columns to categorical,
                             "bin_method" i.e. method of binning - "equal_frequency" or "equal_range" and
                             "monotonicity_check" 1 for monotonic binning else 0. monotonicity_check of 1 will
                             dynamically calculate the bin_size ensuring monotonic nature but can be expensive operation.
    :param print_impact: True, False
    :return: Dataframe [attribute, iv]
    """

    if label_col not in idf.columns:
        raise TypeError("Invalid input for Label Column")

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols

    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]

    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(
        set([e for e in list_of_cols if e not in (drop_cols + [label_col])])
    )

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")

    if idf.where(F.col(label_col) == event_label).count() == 0:
        raise TypeError("Invalid input for Event Label Value")

    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))

    if (len(num_cols) > 0) & bool(encoding_configs):
        bin_size = encoding_configs["bin_size"]
        bin_method = encoding_configs["bin_method"]
        monotonicity_check = encoding_configs["monotonicity_check"]
        if monotonicity_check == 1:
            idf_encoded = monotonic_binning(
                spark, idf, num_cols, [], label_col, event_label, bin_method, bin_size
            )
        else:
            idf_encoded = attribute_binning(
                spark, idf, num_cols, [], bin_method, bin_size
            )

        idf_encoded.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    else:
        idf_encoded = idf

    output = []
    for col in list_of_cols:
        df_iv = (
            idf_encoded.groupBy(col, label_col)
            .count()
            .withColumn(
                label_col, F.when(F.col(label_col) == event_label, 1).otherwise(0)
            )
            .groupBy(col)
            .pivot(label_col)
            .sum("count")
            .fillna(0.5)
            .withColumn("event_pct", F.col("1") / F.sum("1").over(Window.partitionBy()))
            .withColumn(
                "nonevent_pct", F.col("0") / F.sum("0").over(Window.partitionBy())
            )
            .withColumn(
                "iv",
                (F.col("nonevent_pct") - F.col("event_pct"))
                * F.log(F.col("nonevent_pct") / F.col("event_pct")),
            )
        )
        iv_value = df_iv.select(F.sum("iv")).collect()[0][0]
        output.append([col, iv_value])

    odf = (
        spark.createDataFrame(output, ["attribute", "iv"])
        .withColumn("iv", F.round(F.col("iv"), 4))
        .orderBy(F.desc("iv"))
    )
    if print_impact:
        odf.show(odf.count())

    return odf


def IG_calculation(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    label_col="label",
    event_label=1,
    encoding_configs={
        "bin_method": "equal_frequency",
        "bin_size": 10,
        "monotonicity_check": 0,
    },
    print_impact=False,
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param label_col: Label/Target column
    :param event_label: Value of (positive) event (i.e label 1)
    :param encoding_configs: Takes input in dictionary format. Default {} i.e. empty dict means no encoding is required.
                             In case numerical columns are present and encoding is required, following keys shall be
                             provided - "bin_size" i.e. no. of bins for converting the numerical columns to categorical,
                             "bin_method" i.e. method of binning - "equal_frequency" or "equal_range" and
                             "monotonicity_check" 1 for monotonic binning else 0. monotonicity_check of 1 will
                             dynamically calculate the bin_size ensuring monotonic nature but can be expensive operation.
    :param print_impact: True, False
    :return: Dataframe [attribute, ig]
    """

    if label_col not in idf.columns:
        raise TypeError("Invalid input for Label Column")

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(
        set([e for e in list_of_cols if e not in (drop_cols + [label_col])])
    )

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")
    if idf.where(F.col(label_col) == event_label).count() == 0:
        raise TypeError("Invalid input for Event Label Value")

    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))

    if (len(num_cols) > 0) & bool(encoding_configs):
        bin_size = encoding_configs["bin_size"]
        bin_method = encoding_configs["bin_method"]
        monotonicity_check = encoding_configs["monotonicity_check"]
        if monotonicity_check == 1:
            idf_encoded = monotonic_binning(
                spark, idf, num_cols, [], label_col, event_label, bin_method, bin_size
            )
        else:
            idf_encoded = attribute_binning(
                spark, idf, num_cols, [], bin_method, bin_size
            )
        idf_encoded.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    else:
        idf_encoded = idf

    output = []
    total_event = idf.where(F.col(label_col) == event_label).count() / idf.count()
    total_entropy = -(
        total_event * math.log2(total_event)
        + ((1 - total_event) * math.log2((1 - total_event)))
    )
    for col in list_of_cols:
        idf_entropy = (
            idf_encoded.withColumn(
                label_col, F.when(F.col(label_col) == event_label, 1).otherwise(0)
            )
            .groupBy(col)
            .agg(
                F.sum(F.col(label_col)).alias("event_count"),
                F.count(F.col(label_col)).alias("total_count"),
            )
            .dropna()
            .withColumn("event_pct", F.col("event_count") / F.col("total_count"))
            .withColumn(
                "segment_pct",
                F.col("total_count") / F.sum("total_count").over(Window.partitionBy()),
            )
            .withColumn(
                "entropy",
                -F.col("segment_pct")
                * (
                    (F.col("event_pct") * F.log2(F.col("event_pct")))
                    + ((1 - F.col("event_pct")) * F.log2((1 - F.col("event_pct"))))
                ),
            )
        )
        entropy = (
            idf_entropy.groupBy().sum("entropy").rdd.flatMap(lambda x: x).collect()[0]
        )
        ig_value = total_entropy - entropy if entropy else None
        output.append([col, ig_value])

    odf = (
        spark.createDataFrame(output, ["attribute", "ig"])
        .withColumn("ig", F.round(F.col("ig"), 4))
        .orderBy(F.desc("ig"))
    )
    if print_impact:
        odf.show(odf.count())

    return odf
