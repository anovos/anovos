# coding=utf-8
"""
This submodule focuses on understanding the interaction between different attributes and/or the relationship
between an attribute & the binary target variable.

Association between attributes is measured by:
- correlation_matrix
- variable_clustering

Association between an attribute and binary target is measured by:
- IV_calculation
- IG_calculation

"""
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
    cat_to_num_unsupervised,
    imputation_MMM,
    monotonic_binning,
)
from anovos.shared.utils import attributeType_segregation


def correlation_matrix(
    spark, idf, list_of_cols="all", drop_cols=[], stats_unique={}, print_impact=False
):
    """
    This function calculates correlation coefficient statistical, which measures the strength of the relationship
    between the relative movements of two attributes. Pearson’s correlation coefficient is a standard approach of
    measuring correlation between two variables. However, it has some drawbacks: a) It works only with continuous
    variables, b) It only accounts for a linear relationship between variables, and c) It is sensitive to outliers.
    To avoid these issues, we are computing Phik (𝜙k), which is a new and practical correlation coefficient that
    works consistently between categorical, ordinal and interval variables, captures non-linear dependency and
    reverts to the Pearson correlation coefficient in case of a bivariate normal input distribution. The correlation
    coefficient is calculated for every pair of attributes and its value lies between 0 and 1, where 0 means there is
    no correlation between the two attributes and 1 means strong correlation. However, this methodology have
    drawbacks of its own as it is found to be more computational expensive especially when number of columns in the
    input dataset is on higher side (number of pairs to analyse increases exponentially with number of columns).
    Further, there is no indication of the direction of the relationship. More detail can be referred from the [
    source paper] [1].

    [1]: https://arxiv.org/abs/1811.11440/     "source paper"

    This function returns a correlation matrix dataframe of schema – attribute, <attribute_names>. Correlation
    between attribute X and Y can be found at intersection of a) row with value X in ‘attribute’ column and b) column
    ‘Y’ (or row with value Y in ‘attribute’ column and column ‘X’).

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)

    Returns
    -------
    DataFrame
        [attribute,*attribute_names]

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
    Variable Clustering groups attributes that are as correlated as possible among themselves within a cluster and
    as uncorrelated as possible with attribute in other clusters. The function is leveraging [VarClusHi] [2] library
    to do variable clustering; however, this library is not implemented in a scalable manner due to which the
    analysis is done on a sample dataset. Further, it is found to be a bit computational expensive especially when
    number of columns in the input dataset is on higher side (number of pairs to analyse increases exponentially with
    number of columns).

    [2]: https://github.com/jingtt/varclushi   "VarCluShi"

    It returns a Spark Dataframe with schema – Cluster, Attribute, RS_Ratio. Attributes similar to each other are grouped
    together with the same cluster id. The attribute with the lowest (1 — RS_Ratio) can be chosen as a representative of the cluster
    while discarding the other attributes from that cluster. This can also help in achieving the dimension reduction, if required.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    sample_size
        Maximum sample size (in terms of number of rows) taken for the computation.
        Sample dataset is extracted using random sampling. (Default value = 100000)
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
        This is used to remove single value columns from the analysis purpose. (Default value = {})
    stats_mode
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
        mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
        This is used for MMM imputation as Variable Clustering doesn’t work with missing values. (Default value = {})
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)

    Returns
    -------
    DataFrame
        [Cluster, Attribute, RS_Ratio]

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
    Information Value (IV) is simple and powerful technique to conduct attribute relevance analysis. It measures
    how well an attribute is able to distinguish between a binary target variable i.e. label 0 from label 1,
    and hence helps in ranking attributes on the basis of their importance. In the heart of IV methodology are groups
    (bins) of observations. For categorical attributes, usually each category is a bin while numerical attributes
    need to be split into categories.

    IV = ∑ (% of non-events - % of events) * WOE
    <br>where:
    <br>WOE = In(% of non-events ➗ % of events)
    <br>% of event = % label 1 in a bin
    <br>% of non-event = % label 0 in a bin

    General rule of thumb while creating the bins are that a) each bin should have at least 5% of the observations,
    b) the WOE should be monotonic, i.e. either growing or decreasing with the bins, and c) missing values should be
    binned separately. An article  from listendata.com can be referred for good understanding of IV & WOE concepts.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    label_col
        Label/Target column (Default value = "label")
    event_label
        Value of (positive) event (i.e label 1) (Default value = 1)
    encoding_configs
        Takes input in dictionary format. {} i.e. empty dict means no encoding is required.
        In case numerical columns are present and encoding is required, following keys shall be
        provided - "bin_size" (Default value = 10) i.e. no. of bins for converting the numerical columns to categorical,
        "bin_method" i.e. method of binning - "equal_frequency" or "equal_range" (Default value = "equal_frequency") and
        "monotonicity_check" 1 for monotonic binning else 0. monotonicity_check of 1 will
        dynamically calculate the bin_size ensuring monotonic nature but can be expensive operation (Default value = 0).
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)

    Returns
    -------
    DataFrame
        [attribute, iv]

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
    Information Gain (IG) is another powerful technique for feature selection analysis. Information gain is
    calculated by comparing the entropy of the dataset before and after a transformation (introduction of attribute
    in this particular case). Similar to IV calculation, each category is a bin for categorical attributes,
    while numerical attributes need to be split into categories.

    IG = Total Entropy – Entropy

    Total Entropy= -%event*log⁡(%event)-(1-%event)*log⁡(1-%event)

    Entropy = ∑(-%〖event〗_i*log⁡(%〖event〗_i )-(1-%〖event〗_i )*log⁡(1-%〖event〗_i)


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    label_col
        Label/Target column (Default value = "label")
    event_label
        Value of (positive) event (i.e label 1) (Default value = 1)
    encoding_configs
        Takes input in dictionary format. {} i.e. empty dict means no encoding is required.
        In case numerical columns are present and encoding is required, following keys shall be
        provided - "bin_size" (Default value = 10) i.e. no. of bins for converting the numerical columns to categorical,
        "bin_method" i.e. method of binning - "equal_frequency" or "equal_range" (Default value = "equal_frequency") and
        "monotonicity_check" 1 for monotonic binning else 0. monotonicity_check of 1 will
        dynamically calculate the bin_size ensuring monotonic nature but can be expensive operation (Default value = 0).
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)


    Returns
    -------
    DataFrame
        [attribute, id]

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
