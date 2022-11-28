# coding=utf-8
"""
The data transformer module supports selected pre-processing & transformation functions, such as binning, encoding,
scaling, imputation, to name a few, which are required for statistics generation and quality checks. Functions
supported through this module are listed below:

- attribute_binning
- monotonic_binning
- cat_to_num_transformer
- cat_to_num_unsupervised
- cat_to_num_supervised
- z_standardization
- IQR_standardization
- normalization
- imputation_MMM
- imputation_sklearn
- imputation_matrixFactorization
- auto_imputation
- autoencoder_latentFeatures
- PCA_latentFeatures
- feature_transformation
- boxcox_transformation
- outlier_categories
- expression_parser

"""
import copy
import os
import pickle
import random
import platform
import subprocess
import tempfile
import warnings
from itertools import chain

import numpy as np
import pandas as pd
import pyspark
from packaging import version
from scipy import stats

if version.parse(pyspark.__version__) < version.parse("3.0.0"):
    from pyspark.ml.feature import OneHotEncoderEstimator as OneHotEncoder
else:
    from pyspark.ml.feature import OneHotEncoder

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    PCA,
    Imputer,
    ImputerModel,
    IndexToString,
    MinMaxScaler,
    MinMaxScalerModel,
    PCAModel,
    StringIndexer,
    StringIndexerModel,
    VectorAssembler,
)
from pyspark.ml.linalg import DenseVector
from pyspark.ml.recommendation import ALS
from pyspark.mllib.stat import Statistics
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from anovos.data_analyzer.stats_generator import (
    missingCount_computation,
    uniqueCount_computation,
)
from anovos.data_ingest.data_ingest import read_dataset, recast_column
from anovos.data_ingest.data_sampling import data_sample
from anovos.shared.utils import ends_with, attributeType_segregation, get_dtype

# enable_iterative_imputer is prequisite for importing IterativeImputer
# check the following issue for more details https://github.com/scikit-learn/scikit-learn/issues/16833
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer

if "arm64" not in platform.version().lower():
    import tensorflow
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU


def attribute_binning(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    method_type="equal_range",
    bin_size=10,
    bin_dtype="numerical",
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    print_impact=False,
):
    """
    Attribute binning (or discretization) is a method of numerical attribute into discrete (integer or categorical
    values) using pre-defined number of bins. This data pre-processing technique is used to reduce the effects of
    minor observation errors. Also, Binning introduces non-linearity and tends to improve the performance of the
    model. In this function, we are focussing on unsupervised way of binning i.e. without taking the target
    variable into account - Equal Range Binning, Equal Frequency Binning. In Equal Range method, each bin is of equal
    size/width and computed as:

    w = max- min / no. of bins

    *bins cutoff=[min, min+w,min+2w…..,max-w,max]*

    whereas in Equal Frequency binning method, bins are created in such a way that each bin has equal no. of rows,
    though the width of bins may vary from each other.

    w = 1 / no. of bins

    *bins cutoff=[min, wthpctile, 2wthpctile….,max ]*


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    method_type
        "equal_frequency", "equal_range".
        In "equal_range" method, each bin is of equal size/width and in "equal_frequency", each bin has
        equal no. of rows, though the width of bins may vary. (Default value = "equal_range")
    bin_size
        Number of bins. (Default value = 10)
    bin_dtype
        "numerical", "categorical".
        With "numerical" option, original value is replaced with an Integer (1,2,…) and
        with "categorical" option, original replaced with a string describing min and max value allowed
        in the bin ("minval-maxval"). (Default value = "numerical")
    pre_existing_model
        Boolean argument – True or False. True if binning model exists already, False Otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_binned" e.g. column X is appended as X_binned. (Default value = "replace")
    print_impact
        True, False (Default value = False)
        This argument is to print the number of categories generated for each attribute (may or may be not same as bin_size)

    Returns
    -------
    DataFrame
        Binned Dataframe
    """

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols

    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]

    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if len(list_of_cols) == 0:
        warnings.warn("No Binning Performed - No numerical column(s) to transform")
        return idf

    if method_type not in ("equal_frequency", "equal_range"):
        raise TypeError("Invalid input for method_type")

    if bin_size < 2:
        raise TypeError("Invalid input for bin_size")

    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    if pre_existing_model:
        df_model = spark.read.parquet(model_path + "/attribute_binning")
        bin_cutoffs = []
        for i in list_of_cols:
            mapped_value = (
                df_model.where(F.col("attribute") == i)
                .select("parameters")
                .rdd.flatMap(lambda x: x)
                .collect()[0]
            )
            bin_cutoffs.append(mapped_value)
    else:
        if method_type == "equal_frequency":
            pctile_width = 1 / bin_size
            pctile_cutoff = []
            for j in range(1, bin_size):
                pctile_cutoff.append(j * pctile_width)
            bin_cutoffs = idf.approxQuantile(list_of_cols, pctile_cutoff, 0.01)
        else:
            funs = [F.max, F.min]
            exprs = [f(F.col(c)) for f in funs for c in list_of_cols]
            list_result = idf.groupby().agg(*exprs).rdd.flatMap(lambda x: x).collect()
            bin_cutoffs = []
            drop_col_process = []
            for i in range(int(len(list_result) / 2)):
                bin_cutoff = []
                max_val = list_result[i]
                min_val = list_result[i + int(len(list_result) / 2)]
                if not max_val and max_val != 0:
                    drop_col_process.append(list_of_cols[i])
                    continue
                bin_width = (max_val - min_val) / bin_size
                for j in range(1, bin_size):
                    bin_cutoff.append(min_val + j * bin_width)
                bin_cutoffs.append(bin_cutoff)
            if drop_col_process:
                warnings.warn(
                    "Columns contains too much null values. Dropping "
                    + ", ".join(drop_col_process)
                )
                list_of_cols = list(
                    set([e for e in list_of_cols if e not in drop_col_process])
                )
        if model_path != "NA":
            df_model = spark.createDataFrame(
                zip(list_of_cols, bin_cutoffs), schema=["attribute", "parameters"]
            )

            df_model.write.parquet(model_path + "/attribute_binning", mode="overwrite")

    def bucket_label(value, index):
        if value is None:
            return None

        for i in range(0, len(bin_cutoffs[index])):
            if value <= bin_cutoffs[index][i]:
                if bin_dtype == "numerical":
                    return i + 1
                else:
                    if i == 0:
                        return "<= " + str(round(bin_cutoffs[index][i], 4))
                    else:
                        return (
                            str(round(bin_cutoffs[index][i - 1], 4))
                            + "-"
                            + str(round(bin_cutoffs[index][i], 4))
                        )
            else:
                next

        if bin_dtype == "numerical":
            return len(bin_cutoffs[0]) + 1
        else:
            return "> " + str(round(bin_cutoffs[index][len(bin_cutoffs[0]) - 1], 4))

    if bin_dtype == "numerical":
        f_bucket_label = F.udf(bucket_label, T.IntegerType())
    else:
        f_bucket_label = F.udf(bucket_label, T.StringType())

    odf = idf
    for idx, i in enumerate(list_of_cols):
        odf = odf.withColumn(i + "_binned", f_bucket_label(F.col(i), F.lit(idx)))

    if output_mode == "replace":
        for col in list_of_cols:
            odf = odf.drop(col).withColumnRenamed(col + "_binned", col)
    if print_impact:
        if output_mode == "replace":
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_binned") for i in list_of_cols]
        uniqueCount_computation(spark, odf, output_cols).show(len(output_cols), False)
    return odf


def monotonic_binning(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    label_col="label",
    event_label=1,
    bin_method="equal_range",
    bin_size=10,
    bin_dtype="numerical",
    output_mode="replace",
):
    """
    This function constitutes supervised way of binning the numerical attribute into discrete (integer or
    categorical values) attribute. Instead of pre-defined fixed number of bins, number of bins are dynamically
    computed to ensure the monotonic nature of bins i.e. % event should increase or decrease with the bin. Monotonic
    nature of bins is evaluated by looking at spearman rank correlation, which should be either +1 or -1, between the
    bin index and % event. In case, the monotonic nature is not attained, user defined fixed number of bins are used
    for the binning.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
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
    bin_method
        "equal_frequency", "equal_range".
        In "equal_range" method, each bin is of equal size/width and in "equal_frequency", each bin has
        equal no. of rows, though the width of bins may vary. (Default value = "equal_range")
    bin_size
        Default number of bins in case monotonicity is not achieved.
    bin_dtype
        "numerical", "categorical".
        With "numerical" option, original value is replaced with an Integer (1,2,…) and
        with "categorical" option, original replaced with a string describing min and max value allowed
        in the bin ("minval-maxval"). (Default value = "numerical")
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_binned" e.g. column X is appended as X_binned. (Default value = "replace")

    Returns
    -------
    DataFrame
        Binned Dataframe
    """
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(
        set([e for e in list_of_cols if e not in (drop_cols + [label_col])])
    )

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    odf = idf
    for col in list_of_cols:
        n = 20
        r = 0
        while n > 2:
            tmp = (
                attribute_binning(
                    spark,
                    idf,
                    [col],
                    drop_cols=[],
                    method_type=bin_method,
                    bin_size=n,
                    output_mode="append",
                )
                .select(label_col, col, col + "_binned")
                .withColumn(
                    label_col, F.when(F.col(label_col) == event_label, 1).otherwise(0)
                )
                .groupBy(col + "_binned")
                .agg(F.avg(col).alias("mean_val"), F.avg(label_col).alias("mean_label"))
                .dropna()
            )
            r, p = stats.spearmanr(
                tmp.toPandas()[["mean_val"]], tmp.toPandas()[["mean_label"]]
            )
            if r == 1.0:
                odf = attribute_binning(
                    spark,
                    odf,
                    [col],
                    drop_cols=[],
                    method_type=bin_method,
                    bin_size=n,
                    bin_dtype=bin_dtype,
                    output_mode=output_mode,
                )
                break
            n = n - 1
            r = 0
        if r < 1.0:
            odf = attribute_binning(
                spark,
                odf,
                [col],
                drop_cols=[],
                method_type=bin_method,
                bin_size=bin_size,
                bin_dtype=bin_dtype,
                output_mode=output_mode,
            )

    return odf


def cat_to_num_transformer(
    spark, idf, list_of_cols, drop_cols, method_type, encoding, label_col, event_label
):

    """
    This is method which helps converting a categorical attribute into numerical attribute(s) based on the analysis dataset. If there's a presence of label column then the relevant processing would happen through cat_to_num_supervised. However, for unsupervised scenario, the processing would happen through cat_to_num_unsupervised. Computation details can be referred from the respective functions.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of categorical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all categorical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them.
    method_type
        Depending upon the use case the method type can be either Supervised or Unsupervised. For Supervised use case, label_col is mandatory.
    encoding
        "label_encoding" or "onehot_encoding"
        In label encoding, each categorical value is assigned a unique integer based on alphabetical
        or frequency ordering (both ascending & descending options are available that can be selected by index_order argument).
        In one-hot encoding, every unique value in the column will be added in a form of dummy/binary column.
    label_col
        Label/Target column
    event_label
        Value of (positive) event
    """

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    if len(cat_cols) > 0:
        if list_of_cols == "all":
            list_of_cols = cat_cols
        if isinstance(list_of_cols, str):
            list_of_cols = [x.strip() for x in list_of_cols.split("|")]
        if isinstance(drop_cols, str):
            drop_cols = [x.strip() for x in drop_cols.split("|")]

        if any(x not in cat_cols for x in list_of_cols):
            raise TypeError("Invalid input for Column(s)")

        if (method_type == "supervised") & (label_col is not None):

            odf = cat_to_num_supervised(
                spark, idf, label_col=label_col, event_label=event_label
            )
            odf = odf.withColumn(
                label_col,
                F.when(F.col(label_col) == event_label, F.lit(1)).otherwise(F.lit(0)),
            )

            return odf

        elif (method_type == "unsupervised") & (label_col is None):
            odf = cat_to_num_unsupervised(
                spark,
                idf,
                method_type=encoding,
                index_order="frequencyDesc",
            )

            return odf
    else:

        return idf


def cat_to_num_unsupervised(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    method_type="label_encoding",
    index_order="frequencyDesc",
    cardinality_threshold=50,
    pre_existing_model=False,
    model_path="NA",
    stats_unique={},
    output_mode="replace",
    print_impact=False,
):
    """
    This is unsupervised method of converting a categorical attribute into numerical attribute(s). This is among
    the most important transformations required for any modelling exercise, as most of the machine learning
    algorithms cannot process categorical values. It covers two popular encoding techniques – label encoding &
    one-hot encoding.

    In label encoding, each categorical value is assigned a unique integer based on alphabetical or frequency
    ordering (both ascending & descending options are available – can be selected by index_order argument). One of
    the pitfalls of using this technique is that the model may learn some spurious relationship, which doesn't exist
    or might not make any logical sense in the real world settings. In one-hot encoding, every unique value in the
    attribute will be added as a feature in a form of dummy/binary attribute. However, using this method on high
    cardinality attributes can further aggravate the dimensionality issue.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of categorical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all categorical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    method_type
        "label_encoding" or "onehot_encoding"
        In label encoding, each categorical value is assigned a unique integer based on alphabetical
        or frequency ordering (both ascending & descending options are available that can be selected by index_order argument).
        In one-hot encoding, every unique value in the column will be added in a form of dummy/binary column. (Default value = 1)
    index_order
        "frequencyDesc", "frequencyAsc", "alphabetDesc", "alphabetAsc".
        Valid only for Label Encoding method_type. (Default value = "frequencyDesc")
    cardinality_threshold
        Defines threshold to skip columns with higher cardinality values from encoding - a warning is issued. (Default value = 50)
    pre_existing_model
        Boolean argument - True or False. True if encoding model exists already, False Otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre existing model nor there is a need to save one.
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_index" for label encoding e.g. column X is appended as X_index, or
        a postfix "_{n}" for one hot encoding, n varies from 0 to unique value count e.g. column X is appended as X_0,
        X_1, X_2 (n = 3 i.e. no. of unique values for X). (Default value = "replace")
    print_impact
        True, False (Default value = False)
        This argument is to print out the change in schema (one hot encoding) or descriptive statistics (label encoding)

    Returns
    -------
    DataFrame
        Encoded Dataframe

    """

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == "all":
        list_of_cols = cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    if any(x not in cat_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if method_type not in ("onehot_encoding", "label_encoding"):
        raise TypeError("Invalid input for method_type")
    if index_order not in (
        "frequencyDesc",
        "frequencyAsc",
        "alphabetDesc",
        "alphabetAsc",
    ):
        raise TypeError("Invalid input for Encoding Index Order")
    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    if stats_unique == {}:
        skip_cols = (
            uniqueCount_computation(spark, idf, list_of_cols)
            .where(F.col("unique_values") > cardinality_threshold)
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
    else:
        skip_cols = (
            read_dataset(spark, **stats_unique)
            .where(F.col("unique_values") > cardinality_threshold)
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
    skip_cols = list(
        set([e for e in skip_cols if e in list_of_cols and e not in drop_cols])
    )

    if skip_cols:
        warnings.warn(
            "Columns dropped from encoding due to high cardinality: "
            + ",".join(skip_cols)
        )

    list_of_cols = list(
        set([e for e in list_of_cols if e not in drop_cols + skip_cols])
    )

    if len(list_of_cols) == 0:
        warnings.warn("No Encoding Computation - No categorical column(s) to transform")
        return idf

    list_of_cols_vec = []
    list_of_cols_idx = []
    for i in list_of_cols:
        list_of_cols_vec.append(i + "_vec")
        list_of_cols_idx.append(i + "_index")

    odf_indexed = idf
    if version.parse(pyspark.__version__) < version.parse("3.0.0"):
        for idx, i in enumerate(list_of_cols):
            if pre_existing_model:
                indexerModel = StringIndexerModel.load(
                    model_path + "/cat_to_num_unsupervised/indexer-model/" + i
                )
            else:
                stringIndexer = StringIndexer(
                    inputCol=i,
                    outputCol=i + "_index",
                    stringOrderType=index_order,
                    handleInvalid="keep",
                )
                indexerModel = stringIndexer.fit(idf.select(i))

                if model_path != "NA":
                    indexerModel.write().overwrite().save(
                        model_path + "/cat_to_num_unsupervised/indexer-model/" + i
                    )

            odf_indexed = indexerModel.transform(odf_indexed)

    else:
        if pre_existing_model:
            indexerModel = StringIndexerModel.load(
                model_path + "/cat_to_num_unsupervised/indexer"
            )
        else:
            stringIndexer = StringIndexer(
                inputCols=list_of_cols,
                outputCols=list_of_cols_idx,
                stringOrderType=index_order,
                handleInvalid="keep",
            )
            indexerModel = stringIndexer.fit(odf_indexed)
            if model_path != "NA":
                indexerModel.write().overwrite().save(
                    model_path + "/cat_to_num_unsupervised/indexer"
                )
        odf_indexed = indexerModel.transform(odf_indexed)

    if method_type == "onehot_encoding":
        if pre_existing_model:
            encoder = OneHotEncoder.load(
                model_path + "/cat_to_num_unsupervised/encoder"
            )
        else:
            encoder = OneHotEncoder(
                inputCols=list_of_cols_idx,
                outputCols=list_of_cols_vec,
                handleInvalid="keep",
            )
            if model_path != "NA":
                encoder.write().overwrite().save(
                    model_path + "/cat_to_num_unsupervised/encoder"
                )

        odf = encoder.fit(odf_indexed).transform(odf_indexed)

        new_cols = []
        odf_sample = odf.take(1)
        for i in list_of_cols:
            odf_schema = odf.schema
            uniq_cats = odf_sample[0].asDict()[i + "_vec"].size
            for j in range(0, uniq_cats):
                odf_schema = odf_schema.add(
                    T.StructField(i + "_" + str(j), T.IntegerType())
                )
                new_cols.append(i + "_" + str(j))

            odf = odf.rdd.map(
                lambda x: (
                    *x,
                    *(DenseVector(x[i + "_vec"]).toArray().astype(int).tolist()),
                )
            ).toDF(schema=odf_schema)

            if output_mode == "replace":
                odf = odf.drop(i, i + "_vec", i + "_index")
            else:
                odf = odf.drop(i + "_vec", i + "_index")

    else:
        odf = odf_indexed
        for i in list_of_cols:
            odf = odf.withColumn(
                i + "_index",
                F.when(F.col(i).isNull(), None).otherwise(
                    F.col(i + "_index").cast(T.IntegerType())
                ),
            )
        if output_mode == "replace":
            for i in list_of_cols:
                odf = odf.drop(i).withColumnRenamed(i + "_index", i)
            odf = odf.select(idf.columns)

    if print_impact:
        if method_type == "label_encoding":
            if output_mode == "append":
                new_cols = [i + "_index" for i in list_of_cols]
            else:
                new_cols = list_of_cols
            print("Before")
            idf.select(list_of_cols).summary("count", "min", "max").show(3, False)
            print("After")
            odf.select(new_cols).summary("count", "min", "max").show(3, False)

        if method_type == "onehot_encoding":
            print("Before")
            idf.select(list_of_cols).printSchema()
            print("After")
            if output_mode == "append":
                odf.select(list_of_cols + new_cols).printSchema()
            else:
                odf.select(new_cols).printSchema()
        if skip_cols:
            print(
                "Columns dropped from encoding due to high cardinality: "
                + ",".join(skip_cols)
            )
    return odf


def cat_to_num_supervised(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    label_col="label",
    event_label=1,
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    persist=False,
    persist_option=pyspark.StorageLevel.MEMORY_AND_DISK,
    print_impact=False,
):
    """
    This is a supervised method to convert a categorical attribute into a numerical attribute. It takes a
    label/target column to indicate whether the event is positive or negative. For each column, the positive event
    rate for each categorical value is used as the encoded numerical value.

    For example, there are 3 distinct values in a categorical attribute X: X1, X2 and X3. Within the input dataframe, there are
    - 15 positive events and 5 negative events with X==X1;
    - 10 positive events and 40 negative events with X==X2;
    - 20 positive events and 20 negative events with X==X3.

    Thus, value X1 is mapped to 15/(15+5) = 0.75, value X2 is mapped to 10/(10+40) = 0.2 and value X3 is mapped to
    20/(20+20) = 0.5. This mapping will be applied to all values in attribute X. This encoding method can be helpful
    in avoiding creating too many dummy variables which may cause dimensionality issue and it also works with
    categorical attributes without an order or rank.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of categorical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all categorical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
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
    pre_existing_model
        Boolean argument – True or False. True if model (original and mapped numerical value
        for each column) exists already, False Otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" is used to save the model in "intermediate_data/" folder for optimization purpose.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_encoded" e.g. column X is appended as X_encoded. (Default value = "replace")
    persist
        Boolean argument - True or False. This parameter is for optimization purpose. If True, repeatedly used dataframe
        will be persisted (StorageLevel can be specified in persist_option). We recommend setting this parameter as True
        if at least one of the following criteria is True:
        (1) The underlying data source is in csv format
        (2) The transformation will be applicable to most columns. (Default value = False)
    persist_option
        A pyspark.StorageLevel instance. This parameter is useful only when persist is True.
        (Default value = pyspark.StorageLevel.MEMORY_AND_DISK)
    print_impact
        True, False (Default value = False)
        This argument is to print out the descriptive statistics of encoded columns.

    Returns
    -------
    DataFrame
        Encoded Dataframe
    """

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == "all":
        list_of_cols = cat_cols
    elif isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]
    list_of_cols = list(
        set([e for e in list_of_cols if (e not in drop_cols) & (e != label_col)])
    )

    if any(x not in cat_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn("No Categorical Encoding - No categorical column(s) to transform")
        return idf
    if label_col not in idf.columns:
        raise TypeError("Invalid input for Label Column")

    odf = idf
    label_col_bool = label_col + "_cat_to_num_sup_temp"
    idf = idf.withColumn(
        label_col_bool,
        F.when(F.col(label_col) == event_label, "1").otherwise("0"),
    )

    if model_path == "NA":
        skip_if_error = True
        model_path = "intermediate_data"
    else:
        skip_if_error = False
    save_model = True

    if persist:
        idf = idf.persist(persist_option)
    for index, i in enumerate(list_of_cols):
        if pre_existing_model:
            df_tmp = spark.read.csv(
                model_path + "/cat_to_num_supervised/" + i,
                header=True,
                inferSchema=True,
            )
        else:
            df_tmp = (
                idf.select(i, label_col_bool)
                .groupBy(i)
                .pivot(label_col_bool)
                .count()
                .fillna(0)
                .withColumn(
                    i + "_encoded", F.round(F.col("1") / (F.col("1") + F.col("0")), 4)
                )
                .drop(*["1", "0"])
            )

            if save_model:
                try:
                    df_tmp.coalesce(1).write.csv(
                        model_path + "/cat_to_num_supervised/" + i,
                        header=True,
                        mode="overwrite",
                        ignoreLeadingWhiteSpace=False,
                        ignoreTrailingWhiteSpace=False,
                    )
                    df_tmp = spark.read.csv(
                        model_path + "/cat_to_num_supervised/" + i,
                        header=True,
                        inferSchema=True,
                    )
                except Exception as error:
                    if skip_if_error:
                        warnings.warn(
                            "For optimization purpose, we recommend specifying a valid model_path value to save the intermediate data. Saving to the default path - '"
                            + model_path
                            + "/cat_to_num_supervised/"
                            + i
                            + "' faced an error."
                        )
                        save_model = False
                    else:
                        raise error

        if df_tmp.count() > 1:
            odf = odf.join(df_tmp, i, "left_outer")
        else:
            odf = odf.crossJoin(df_tmp)

    if output_mode == "replace":
        for i in list_of_cols:
            odf = odf.drop(i).withColumnRenamed(i + "_encoded", i)
        odf = odf.select([i for i in idf.columns if i != label_col_bool])

    if print_impact:
        if output_mode == "replace":
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_encoded") for i in list_of_cols]
        print("Before: ")
        idf.select(list_of_cols).summary("count", "min", "max").show(3, False)
        print("After: ")
        odf.select(output_cols).summary("count", "min", "max").show(3, False)

    if persist:
        idf.unpersist()
    return odf


def z_standardization(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    print_impact=False,
):
    """
    Standardization is commonly used in data pre-processing process. z_standardization standardizes the selected
    attributes of an input dataframe by normalizing each attribute to have standard deviation of 1 and mean of 0. For
    each attribute, the standard deviation (s) and mean (u) are calculated and a sample x will be standardized into (
    x-u)/s. If the standard deviation of an attribute is 0, it will be excluded in standardization and a warning will
    be shown. None values will be kept as None in the output dataframe.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    pre_existing_model
        Boolean argument – True or False. True if model files (Mean/stddev for each feature) exists already, False Otherwise (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_scaled" e.g. column X is appended as X_scaled. (Default value = "replace")
    print_impact
        True, False (Default value = False)
        This argument is to print out the before and after descriptive statistics of rescaled columns.

    Returns
    -------
    DataFrame
        Rescaled Dataframe

    """
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn(
            "No Standardization Performed - No numerical column(s) to transform"
        )
        return idf

    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    parameters = []
    excluded_cols = []
    if pre_existing_model:
        df_model = spark.read.parquet(model_path + "/z_standardization")
        for i in list_of_cols:
            mapped_value = (
                df_model.where(F.col("feature") == i)
                .select("parameters")
                .rdd.flatMap(lambda x: x)
                .collect()[0]
            )
            parameters.append(mapped_value)
    else:
        for i in list_of_cols:
            mean, stddev = idf.select(F.mean(i), F.stddev(i)).first()
            parameters.append(
                [float(mean) if mean else None, float(stddev) if stddev else None]
            )
            if stddev:
                if round(stddev, 5) == 0.0:
                    excluded_cols.append(i)
            else:
                excluded_cols.append(i)
    if len(excluded_cols) > 0:
        warnings.warn(
            "The following column(s) are excluded from standardization because the standard deviation is zero:"
            + str(excluded_cols)
        )

    odf = idf
    for index, i in enumerate(list_of_cols):
        if i not in excluded_cols:
            modify_col = (i + "_scaled") if (output_mode == "append") else i
            odf = odf.withColumn(
                modify_col, (F.col(i) - parameters[index][0]) / parameters[index][1]
            )

    if (not pre_existing_model) & (model_path != "NA"):
        df_model = spark.createDataFrame(
            zip(list_of_cols, parameters), schema=["feature", "parameters"]
        )
        df_model.coalesce(1).write.parquet(
            model_path + "/z_standardization", mode="overwrite"
        )

    if print_impact:
        if output_mode == "replace":
            output_cols = list_of_cols
        else:
            output_cols = [
                (i + "_scaled") for i in list_of_cols if i not in excluded_cols
            ]
        print("Before: ")
        idf.select(list_of_cols).describe().show(5, False)
        print("After: ")
        odf.select(output_cols).describe().show(5, False)

    return odf


def IQR_standardization(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    print_impact=False,
):
    """

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    pre_existing_model
        Boolean argument – True or False. True if model files (25/50/75 percentile for each feature) exists already, False Otherwise (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_scaled" e.g. column X is appended as X_scaled. (Default value = "replace")
    print_impact
        True, False (Default value = False)
        This argument is to print out the before and after descriptive statistics of rescaled columns.

    Returns
    -------
    DataFrame
        Rescaled Dataframe
    """
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn(
            "No Standardization Performed - No numerical column(s) to transform"
        )
        return idf

    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    if pre_existing_model:
        df_model = spark.read.parquet(model_path + "/IQR_standardization")
        parameters = []
        for i in list_of_cols:
            mapped_value = (
                df_model.where(F.col("feature") == i)
                .select("parameters")
                .rdd.flatMap(lambda x: x)
                .collect()[0]
            )
            parameters.append(mapped_value)
    else:
        parameters = idf.approxQuantile(list_of_cols, [0.25, 0.5, 0.75], 0.01)

    excluded_cols = []
    for i, param in zip(list_of_cols, parameters):
        if len(param) > 0:
            if round(param[0], 5) == round(param[2], 5):
                excluded_cols.append(i)
        else:
            excluded_cols.append(i)
    if len(excluded_cols) > 0:
        warnings.warn(
            "The following column(s) are excluded from standardization because the 75th and 25th percentiles are the same:"
            + str(excluded_cols)
        )

    odf = idf
    for index, i in enumerate(list_of_cols):
        if i not in excluded_cols:
            modify_col = (i + "_scaled") if (output_mode == "append") else i
            odf = odf.withColumn(
                modify_col,
                (F.col(i) - parameters[index][1])
                / (parameters[index][2] - parameters[index][0]),
            )

    if (not pre_existing_model) & (model_path != "NA"):
        df_model = spark.createDataFrame(
            zip(list_of_cols, parameters), schema=["feature", "parameters"]
        )
        df_model.coalesce(1).write.parquet(
            model_path + "/IQR_standardization", mode="overwrite"
        )

    if print_impact:
        if output_mode == "replace":
            output_cols = list_of_cols
        else:
            output_cols = [
                (i + "_scaled") for i in list_of_cols if i not in excluded_cols
            ]
        print("Before: ")
        idf.select(list_of_cols).describe().show(5, False)
        print("After: ")
        odf.select(output_cols).describe().show(5, False)

    return odf


def normalization(
    idf,
    list_of_cols="all",
    drop_cols=[],
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    print_impact=False,
):
    """

    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    pre_existing_model
        Boolean argument – True or False. True if normalization/scalar model exists already, False Otherwise (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_scaled" e.g. column X is appended as X_scaled. (Default value = "replace")
    print_impact
        True, False (Default value = False)
        This argument is to print out before and after descriptive statistics of rescaled columns.

    Returns
    -------
    DataFrame
        Rescaled Dataframe

    """
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn(
            "No Normalization Performed - No numerical column(s) to transform"
        )
        return idf

    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    idf_id = idf.withColumn("tempID", F.monotonically_increasing_id())
    idf_partial = idf_id.select(["tempID"] + list_of_cols)

    assembler = VectorAssembler(
        inputCols=list_of_cols, outputCol="list_of_cols_vector", handleInvalid="keep"
    )
    assembled_data = assembler.transform(idf_partial)
    if pre_existing_model:
        scalerModel = MinMaxScalerModel.load(model_path + "/normalization")
    else:
        scaler = MinMaxScaler(
            inputCol="list_of_cols_vector", outputCol="list_of_cols_scaled"
        )
        scalerModel = scaler.fit(assembled_data)

        if model_path != "NA":
            scalerModel.write().overwrite().save(model_path + "/normalization")

    scaledData = scalerModel.transform(assembled_data)

    def vector_to_array(v):
        return v.toArray().tolist()

    f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.FloatType()))

    odf_partial = scaledData.withColumn(
        "list_of_cols_array", f_vector_to_array("list_of_cols_scaled")
    ).drop(*["list_of_cols_scaled", "list_of_cols_vector"])

    odf_schema = odf_partial.schema
    for i in list_of_cols:
        odf_schema = odf_schema.add(T.StructField(i + "_scaled", T.FloatType()))
    odf_partial = (
        odf_partial.rdd.map(lambda x: (*x, *x["list_of_cols_array"]))
        .toDF(schema=odf_schema)
        .drop("list_of_cols_array")
    )

    odf = idf_id.join(odf_partial.drop(*list_of_cols), "tempID", "left_outer").select(
        idf.columns
        + [
            (
                F.when(F.isnan(F.col(i + "_scaled")), None).otherwise(
                    F.col(i + "_scaled")
                )
            ).alias(i + "_scaled")
            for i in list_of_cols
        ]
    )
    if output_mode == "replace":
        for i in list_of_cols:
            odf = odf.drop(i).withColumnRenamed(i + "_scaled", i)
        odf = odf.select(idf.columns)

    if print_impact:
        if output_mode == "replace":
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_scaled") for i in list_of_cols]
        print("Before: ")
        idf.select(list_of_cols).describe().show(5, False)
        print("After: ")
        odf.select(output_cols).describe().show(5, False)

    return odf


def imputation_MMM(
    spark,
    idf,
    list_of_cols="missing",
    drop_cols=[],
    method_type="median",
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    stats_missing={},
    stats_mode={},
    print_impact=False,
):
    """
    This function handles missing value related issues by substituting null values by the measure of central
    tendency (mode for categorical features and mean/median for numerical features). For numerical attributes,
    it leverages [Imputer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer
    .html) functionality of Spark MLlib. Though, Imputer can be used for categorical attributes but this feature is
    available only in Spark3.x, therefore for categorical features, we compute mode or leverage mode computation from
    Measures of Central Tendency.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to impute e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all (non-array) columns for analysis.
        This is super useful instead of specifying all column names manually.
        "missing" (default) can be passed to include only those columns with missing values.
        One of the usecases where "all" may be preferable over "missing" is when the user wants to save
        the imputation model for the future use e.g. a column may not have missing value in the training
        dataset but missing values may possibly appear in the prediction dataset.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    method_type
        "median", "mean" (valid only for for numerical columns attributes).
        Mode is only option for categorical columns. (Default value = "median")
    pre_existing_model
        Boolean argument – True or False. True if imputation model exists already, False otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_imputed" e.g. column X is appended as X_imputed. (Default value = "replace")
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    stats_mode
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
        mode_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    print_impact
        True, False (Default value = False)
        This argument is to print out before and after missing counts of imputed columns.

    Returns
    -------
    DataFrame
        Imputed Dataframe

    """
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
    else:
        missing_df = read_dataset(spark, **stats_missing).select(
            "attribute", "missing_count", "missing_pct"
        )

    missing_cols = (
        missing_df.where(F.col("missing_count") > 0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    if str(pre_existing_model).lower() == "true":
        pre_existing_model = True
    elif str(pre_existing_model).lower() == "false":
        pre_existing_model = False
    else:
        raise TypeError("Non-Boolean input for pre_existing_model")

    if (len(missing_cols) == 0) & (not pre_existing_model) & (model_path == "NA"):
        return idf

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)
    if list_of_cols == "all":
        list_of_cols = num_cols + cat_cols
    if list_of_cols == "missing":
        list_of_cols = [x for x in missing_cols if x in num_cols + cat_cols]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if len(list_of_cols) == 0:
        warnings.warn("No Imputation performed- No column(s) to impute")
        return idf
    if any(x not in num_cols + cat_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if method_type not in ("mode", "mean", "median"):
        raise TypeError("Invalid input for method_type")
    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))

    odf = idf
    if len(num_cols) > 0:
        recast_cols = []
        recast_type = []
        for i in num_cols:
            if get_dtype(idf, i) not in ("float", "double"):
                odf = odf.withColumn(i, F.col(i).cast(T.DoubleType()))
                recast_cols.append(i + "_imputed")
                recast_type.append(get_dtype(idf, i))

        # For mode imputation
        if method_type == "mode":
            if stats_mode == {}:
                parameters = [
                    str(
                        (
                            idf.select(i)
                            .dropna()
                            .groupby(i)
                            .count()
                            .orderBy("count", ascending=False)
                            .first()
                            or [None]
                        )[0]
                    )
                    for i in num_cols
                ]
            else:
                mode_df = read_dataset(spark, **stats_mode).replace("None", None)
                mode_df_cols = list(mode_df.select("attribute").toPandas()["attribute"])
                parameters = []
                for i in num_cols:
                    if i not in mode_df_cols:
                        parameters.append(
                            str(
                                (
                                    idf.select(i)
                                    .dropna()
                                    .groupby(i)
                                    .count()
                                    .orderBy("count", ascending=False)
                                    .first()
                                    or [None]
                                )[0]
                            )
                        )
                    else:
                        parameters.append(
                            mode_df.where(F.col("attribute") == i)
                            .select("mode")
                            .rdd.flatMap(list)
                            .collect()[0]
                        )

            for index, i in enumerate(num_cols):
                odf = odf.withColumn(
                    i + "_imputed",
                    F.when(F.col(i).isNull(), parameters[index]).otherwise(F.col(i)),
                )

        else:  # For mean, median imputation
            # Building new imputer model or uploading the existing model
            if pre_existing_model:
                imputerModel = ImputerModel.load(
                    model_path + "/imputation_MMM/num_imputer-model"
                )
            else:
                imputer = Imputer(
                    strategy=method_type,
                    inputCols=num_cols,
                    outputCols=[(e + "_imputed") for e in num_cols],
                )
                imputerModel = imputer.fit(odf)

            # Applying model
            # odf = recast_column(imputerModel.transform(odf), recast_cols, recast_type)
            odf = imputerModel.transform(odf)
            for i, j in zip(recast_cols, recast_type):
                odf = odf.withColumn(i, F.col(i).cast(j))

            # Saving model if required
            if (not pre_existing_model) & (model_path != "NA"):
                imputerModel.write().overwrite().save(
                    model_path + "/imputation_MMM/num_imputer-model"
                )

    if len(cat_cols) > 0:
        if pre_existing_model:
            df_model = spark.read.csv(
                model_path + "/imputation_MMM/cat_imputer",
                header=True,
                inferSchema=True,
            )
            parameters = []
            for i in cat_cols:
                mapped_value = (
                    df_model.where(F.col("attribute") == i)
                    .select("parameters")
                    .rdd.flatMap(lambda x: x)
                    .collect()[0]
                )
                parameters.append(mapped_value)
        else:
            if stats_mode == {}:
                parameters = [
                    str(
                        (
                            idf.select(i)
                            .dropna()
                            .groupby(i)
                            .count()
                            .orderBy("count", ascending=False)
                            .first()
                            or [None]
                        )[0]
                    )
                    for i in cat_cols
                ]
            else:
                mode_df = read_dataset(spark, **stats_mode).replace("None", None)
                parameters = [
                    mode_df.where(F.col("attribute") == i)
                    .select("mode")
                    .rdd.flatMap(list)
                    .collect()[0]
                    for i in cat_cols
                ]

        for index, i in enumerate(cat_cols):
            odf = odf.withColumn(
                i + "_imputed",
                F.when(F.col(i).isNull(), parameters[index]).otherwise(F.col(i)),
            )

        if (not pre_existing_model) & (model_path != "NA"):
            df_model = spark.createDataFrame(
                zip(cat_cols, parameters), schema=["attribute", "parameters"]
            )
            df_model.repartition(1).write.csv(
                model_path + "/imputation_MMM/cat_imputer",
                header=True,
                mode="overwrite",
            )

    for i in num_cols + cat_cols:
        if i not in missing_cols:
            odf = odf.drop(i + "_imputed")
        elif output_mode == "replace":
            odf = odf.drop(i).withColumnRenamed(i + "_imputed", i)

    if print_impact:
        if output_mode == "replace":
            odf_print = missing_df.select(
                "attribute", F.col("missing_count").alias("missingCount_before")
            ).join(
                missingCount_computation(spark, odf, list_of_cols).select(
                    "attribute", F.col("missing_count").alias("missingCount_after")
                ),
                "attribute",
                "inner",
            )
        else:
            output_cols = [
                (i + "_imputed")
                for i in [e for e in (num_cols + cat_cols) if e in missing_cols]
            ]
            odf_print = missing_df.select(
                "attribute", F.col("missing_count").alias("missingCount_before")
            ).join(
                missingCount_computation(spark, odf, output_cols)
                .withColumnRenamed("attribute", "attribute_after")
                .withColumn(
                    "attribute",
                    F.expr("substring(attribute_after, 1, length(attribute_after)-8)"),
                )
                .drop("missing_pct"),
                "attribute",
                "inner",
            )
        odf_print.show(len(list_of_cols), False)
    return odf


def imputation_sklearn(
    spark,
    idf,
    list_of_cols="missing",
    drop_cols=[],
    missing_threshold=1.0,
    method_type="regression",
    use_sampling=True,
    sample_method="random",
    strata_cols="all",
    stratified_type="population",
    sample_size=10000,
    sample_seed=42,
    persist=True,
    persist_option=pyspark.StorageLevel.MEMORY_AND_DISK,
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    stats_missing={},
    run_type="local",
    auth_key="NA",
    print_impact=False,
):
    """
    The function "imputation_sklearn" leverages sklearn imputer algorithms. Two methods are supported via this function:
    “KNN” and “regression”.

    “KNN” option trains a sklearn.impute.KNNImputer which is based on k-Nearest Neighbors algorithm. The missing
    values of a sample are imputed using the mean of its 5 nearest neighbors in the training set. “regression” option
    trains a sklearn.impute.IterativeImputer which models attribute to impute as a function of rest of the attributes
    and imputes using the estimation. Imputation is performed in an iterative way from attributes with fewest number
    of missing values to most. All the hyperparameters used in the above mentioned imputers are their default values.

    However, sklearn imputers are not scalable, which might be slow if the size of the input dataframe is large.
    In fact, if the input dataframe size exceeds 10 GigaBytes, the model fitting step powered by sklearn might fail.
    Thus, an input sample_size (the default value is 10,000) can be set to control the number of samples to be used
    to train the imputer. If the total number of input dataset exceeds sample_size, the rest of the samples will be imputed
    using the trained imputer in a scalable manner. This is one of the way to demonstrate how Anovos has been
    designed as a scalable feature engineering library.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to impute e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all (non-array) columns for analysis.
        This is super useful instead of specifying all column names manually.
        "missing" (default) can be passed to include only those columns with missing values.
        One of the usecases where "all" may be preferable over "missing" is when the user wants to save
        the imputation model for the future use e.g. a column may not have missing value in the training
        dataset but missing values may possibly appear in the prediction dataset.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    missing_threshold
        Float argument - If list_of_cols is "missing", this argument is used to determined the missing threshold
        for every column. The column that has more (count of missing value/ count of total value) >= missing_threshold
        will be excluded from the list of columns to be imputed. (Default value = 1.0)
    method_type
        "KNN", "regression".
        "KNN" option trains a sklearn.impute.KNNImputer. "regression" option trains a sklearn.impute.IterativeImputer
        (Default value = "regression")
    use_sampling
        Boolean argument - True or False. This argument is used to determine whether to use sampling on
        source and target dataset, True will enable the use of sample method, otherwise False.
        It is recommended to set this as True for large datasets. (Default value = True)
    sample_method
        If use_sampling is True, this argument is used to determine the sampling method.
        "stratified" for Stratified sampling, "random" for Random Sampling.
        For more details, please refer to https://docs.anovos.ai/api/data_ingest/data_sampling.html.
        (Default value = "random")
    strata_cols
        If use_sampling is True and sample_method is "stratified", this argument is used to determine the list
        of columns used to be treated as strata. For more details, please refer to
        https://docs.anovos.ai/api/data_ingest/data_sampling.html. (Default value = "all")
    stratified_type
        If use_sampling is True and sample_method is "stratified", this argument is used to determine the stratified
        sampling method. "population" stands for Proportionate Stratified Sampling,
        "balanced" stands for Optimum Stratified Sampling. For more details, please refer to
        https://docs.anovos.ai/api/data_ingest/data_sampling.html. (Default value = "population")
    sample_size
        If use_sampling is True, this argument is used to determine maximum rows for training the sklearn imputer
        (Default value = 10000)
    sample_seed
        If use_sampling is True, this argument is used to determine the seed of sampling method.
        (Default value = 42)
    persist
        Boolean argument - True or False. This argument is used to determine whether to persist on
        binning result of source and target dataset, True will enable the use of persist, otherwise False.
        It is recommended to set this as True for large datasets. (Default value = True)
    persist_option
        If persist is True, this argument is used to determine the type of persist.
        For all the pyspark.StorageLevel option available in persist, please refer to
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.StorageLevel.html
        (Default value = pyspark.StorageLevel.MEMORY_AND_DISK)
    pre_existing_model
        Boolean argument – True or False. True if imputation model exists already, False otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_imputed" e.g. column X is appended as X_imputed. (Default value = "replace")
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    run_type
        "local", "emr", "databricks", "ak8s" (Default value = "local")
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"
    print_impact
        True, False (Default value = False)
        This argument is to print out before and after missing counts of imputed columns.

    Returns
    -------
    DataFrame
        Imputed Dataframe

    """

    if persist:
        idf = idf.persist(persist_option)
    num_cols = attributeType_segregation(idf)[0]
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf, num_cols)
    else:
        missing_df = (
            read_dataset(spark, **stats_missing)
            .select("attribute", "missing_count", "missing_pct")
            .where(F.col("attribute").isin(num_cols))
        )
    empty_cols = (
        missing_df.where(F.col("missing_pct") == 1.0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    if len(empty_cols) > 0:
        warnings.warn(
            "Following columns dropped from the imputation as all values are null: "
            + ",".join(empty_cols)
        )

    missing_cols = (
        missing_df.where(F.col("missing_count") > 0)
        .where(F.col("missing_pct") < missing_threshold)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    if list_of_cols == "all":
        list_of_cols = num_cols
    if list_of_cols == "missing":
        list_of_cols = missing_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(
        set([e for e in list_of_cols if (e not in drop_cols) & (e not in empty_cols)])
    )

    if len(list_of_cols) <= 1:
        warnings.warn(
            "No Imputation Performed - No Column(s) or Insufficient Column(s) to Impute"
        )
        return idf

    if str(pre_existing_model).lower() == "true":
        pre_existing_model = True
    elif str(pre_existing_model).lower() == "false":
        pre_existing_model = False
    else:
        raise TypeError("Non-Boolean input for pre_existing_model")
    if (
        (len([e for e in list_of_cols if e in missing_cols]) == 0)
        & (not pre_existing_model)
        & (model_path == "NA")
    ):
        warnings.warn(
            "No Imputation Performed - No Column(s) to Impute and No Imputation Model to be saved"
        )
        return idf

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if method_type not in ("KNN", "regression"):
        raise TypeError("Invalid input for method_type")
    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    if pre_existing_model:
        if run_type == "emr":
            bash_cmd = "aws s3 cp " + model_path + "/imputation_sklearn.sav ."
            output = subprocess.check_output(["bash", "-c", bash_cmd])
            imputer = pickle.load(open("imputation_sklearn.sav", "rb"))
        elif run_type == "ak8s":
            bash_cmd = (
                'azcopy cp "'
                + model_path
                + "/imputation_sklearn.sav"
                + str(auth_key)
                + '" .'
            )
            output = subprocess.check_output(["bash", "-c", bash_cmd])
            imputer = pickle.load(open("imputation_sklearn.sav", "rb"))
        else:
            imputer = pickle.load(open(model_path + "/imputation_sklearn.sav", "rb"))
    else:
        if use_sampling:
            count_idf = idf.count()
            if count_idf > sample_size:
                idf_model = data_sample(
                    idf,
                    strata_cols=strata_cols,
                    fraction=sample_size / count_idf,
                    method_type=sample_method,
                    stratified_type=stratified_type,
                    seed_value=sample_seed,
                )
            else:
                idf_model = idf
        else:
            idf_model = idf
        if persist:
            idf_model = idf_model.persist(persist_option)
        idf_pd = idf_model.select(list_of_cols).toPandas()

        if method_type == "KNN":
            imputer = KNNImputer(
                n_neighbors=5, weights="uniform", metric="nan_euclidean"
            )
        if method_type == "regression":
            imputer = IterativeImputer()
        imputer.fit(idf_pd)

        if (not pre_existing_model) & (model_path != "NA"):
            if run_type == "emr":
                pickle.dump(imputer, open("imputation_sklearn.sav", "wb"))
                bash_cmd = (
                    "aws s3 cp imputation_sklearn.sav "
                    + model_path
                    + "/imputation_sklearn.sav"
                )
                output = subprocess.check_output(["bash", "-c", bash_cmd])
                imputer = pickle.load(open("imputation_sklearn.sav", "rb"))
            elif run_type == "ak8s":
                pickle.dump(imputer, open("imputation_sklearn.sav", "wb"))
                bash_cmd = (
                    'azcopy cp "imputation_sklearn.sav" "'
                    + ends_with(model_path)
                    + "imputation_sklearn.sav"
                    + str(auth_key)
                    + '"'
                )
                output = subprocess.check_output(["bash", "-c", bash_cmd])
                imputer = pickle.load(open("imputation_sklearn.sav", "rb"))
            else:
                local_path = model_path + "/imputation_sklearn.sav"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                pickle.dump(imputer, open(local_path, "wb"))
                imputer = pickle.load(
                    open(model_path + "/imputation_sklearn.sav", "rb")
                )

    @F.pandas_udf(returnType=T.ArrayType(T.DoubleType()))
    def prediction(*cols):
        input_pdf = pd.concat(cols, axis=1)
        return pd.Series(row.tolist() for row in imputer.transform(input_pdf))

    result_df = idf.withColumn("features", prediction(*list_of_cols))
    if persist:
        result_df = result_df.persist(persist_option)

    odf_schema = result_df.schema
    for i in list_of_cols:
        odf_schema = odf_schema.add(T.StructField(i + "_imputed", T.FloatType()))
    odf = (
        result_df.rdd.map(lambda x: (*x, *x["features"]))
        .toDF(schema=odf_schema)
        .drop("features")
    )

    output_cols = []
    for i in list_of_cols:
        if output_mode == "append":
            if i not in missing_cols:
                odf = odf.drop(i + "_imputed")
            else:
                output_cols.append(i + "_imputed")
        else:
            odf = odf.drop(i).withColumnRenamed(i + "_imputed", i)
    odf = odf.select(idf.columns + output_cols)

    if print_impact:
        if output_mode == "replace":
            odf_print = missing_df.select(
                "attribute", F.col("missing_count").alias("missingCount_before")
            ).join(
                missingCount_computation(spark, odf, list_of_cols).select(
                    "attribute", F.col("missing_count").alias("missingCount_after")
                ),
                "attribute",
                "inner",
            )
        else:
            odf_print = missing_df.select(
                "attribute", F.col("missing_count").alias("missingCount_before")
            ).join(
                missingCount_computation(spark, odf, output_cols)
                .withColumnRenamed("attribute", "attribute_after")
                .withColumn(
                    "attribute",
                    F.expr("substring(attribute_after, 1, length(attribute_after)-8)"),
                )
                .drop("missing_pct"),
                "attribute",
                "inner",
            )
        odf_print.show(len(list_of_cols), False)
    if persist:
        idf.unpersist()
        if not pre_existing_model:
            idf_model.unpersist()
        result_df.unpersist()
    return odf


def imputation_matrixFactorization(
    spark,
    idf,
    list_of_cols="missing",
    drop_cols=[],
    id_col="",
    output_mode="replace",
    stats_missing={},
    print_impact=False,
):
    """
    imputation_matrixFactorization uses collaborative filtering technique to impute missing values. Collaborative
    filtering is commonly used in recommender systems to fill the missing user-item entries and PySpark provides an
    implementation using alternating least squares (ALS) algorithm, which is used in this function. To fit our
    problem into the ALS model, each attribute is treated as an item and an id column needs to be specified by the
    user to generate the user-item pairs. In the case, ID column doesn't exist in the dataset and proxu ID column is
    implicitly generated by the function. Subsequently, all user-item pairs with known values will be used to train
    the ALS model and the trained model can be used to predict the user-item pairs with missing values.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to impute e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all (non-array) columns for analysis.
        This is super useful instead of specifying all column names manually.
        "missing" (default) can be passed to include only those columns with missing values.
        One of the usecases where "all" may be preferable over "missing" is when the user wants to save
        the imputation model for the future use e.g. a column may not have missing value in the training
        dataset but missing values may possibly appear in the prediction dataset.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    id_col
        ID column (Default value = "")
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_imputed" e.g. column X is appended as X_imputed. (Default value = "replace")
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    print_impact
        True, False (Default value = False)
        This argument is to print out before and after missing counts of imputed columns.

    Returns
    -------
    DataFrame
        Imputed Dataframe

    """

    num_cols = attributeType_segregation(idf)[0]
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf, num_cols)
    else:
        missing_df = (
            read_dataset(spark, **stats_missing)
            .select("attribute", "missing_count", "missing_pct")
            .where(F.col("attribute").isin(num_cols))
        )

    empty_cols = (
        missing_df.where(F.col("missing_pct") == 1.0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    if len(empty_cols) > 0:
        warnings.warn(
            "Following columns dropped from the imputation as all values are null: "
            + ",".join(empty_cols)
        )

    missing_cols = (
        missing_df.where(F.col("missing_count") > 0)
        .where(F.col("missing_pct") < 1.0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    if list_of_cols == "all":
        list_of_cols = num_cols
    if list_of_cols == "missing":
        list_of_cols = missing_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(
        set(
            [
                e
                for e in list_of_cols
                if (e not in drop_cols) & (e != id_col) & (e not in empty_cols)
            ]
        )
    )

    if (len(list_of_cols) == 0) | (
        len([e for e in list_of_cols if e in missing_cols]) == 0
    ):
        warnings.warn("No Imputation Performed - No Column(s) to Impute")
        return idf
    if len(list_of_cols) == 1:
        warnings.warn(
            "No Imputation Performed - Needs more than 1 column for matrix factorization"
        )
        return idf
    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    remove_id = False
    if id_col == "":
        idf = idf.withColumn("id", F.monotonically_increasing_id()).withColumn(
            "id", F.row_number().over(Window.orderBy("id"))
        )
        id_col = "id"
        remove_id = True

    key_and_val = F.create_map(
        list(chain.from_iterable([[F.lit(c), F.col(c)] for c in list_of_cols]))
    )
    df_flatten = idf.select(id_col, F.explode(key_and_val)).withColumn(
        "key", F.concat(F.col("key"), F.lit("_imputed"))
    )

    id_type = get_dtype(idf, id_col)
    if id_type == "string":
        id_indexer = StringIndexer().setInputCol(id_col).setOutputCol("IDLabel")
        id_indexer_model = id_indexer.fit(df_flatten)
        df_flatten = id_indexer_model.transform(df_flatten).drop(id_col)
    else:
        df_flatten = df_flatten.withColumnRenamed(id_col, "IDLabel")

    indexer = StringIndexer().setInputCol("key").setOutputCol("keyLabel")
    indexer_model = indexer.fit(df_flatten)
    df_encoded = indexer_model.transform(df_flatten).drop("key")
    df_model = df_encoded.where(F.col("value").isNotNull())
    df_test = df_encoded.where(F.col("value").isNull())
    if (
        df_model.select("IDLabel").distinct().count()
        < df_encoded.select("IDLabel").distinct().count()
    ):
        warnings.warn(
            "The returned odf may not be fully imputed because values for all list_of_cols are null for some IDs"
        )
    als = ALS(
        maxIter=20,
        regParam=0.01,
        userCol="IDLabel",
        itemCol="keyLabel",
        ratingCol="value",
        coldStartStrategy="drop",
    )
    model = als.fit(df_model)
    df_pred = (
        model.transform(df_test).drop("value").withColumnRenamed("prediction", "value")
    )
    df_encoded_pred = df_model.union(df_pred.select(df_model.columns))

    if id_type == "string":
        IDlabelReverse = IndexToString().setInputCol("IDLabel").setOutputCol(id_col)
        df_encoded_pred = IDlabelReverse.transform(df_encoded_pred)
    else:
        df_encoded_pred = df_encoded_pred.withColumnRenamed("IDLabel", id_col)

    keylabelReverse = IndexToString().setInputCol("keyLabel").setOutputCol("key")
    odf_imputed = (
        keylabelReverse.transform(df_encoded_pred)
        .groupBy(id_col)
        .pivot("key")
        .agg(F.first("value"))
        .select(
            [id_col] + [(i + "_imputed") for i in list_of_cols if i in missing_cols]
        )
    )

    odf = idf.join(odf_imputed, id_col, "left_outer")

    for i in list_of_cols:
        if i not in missing_cols:
            odf = odf.drop(i + "_imputed")
        elif output_mode == "replace":
            odf = odf.drop(i).withColumnRenamed(i + "_imputed", i)

    if remove_id:
        odf = odf.drop("id")

    if print_impact:
        if output_mode == "replace":
            odf_print = missing_df.select(
                "attribute", F.col("missing_count").alias("missingCount_before")
            ).join(
                missingCount_computation(spark, odf, list_of_cols).select(
                    "attribute", F.col("missing_count").alias("missingCount_after")
                ),
                "attribute",
                "inner",
            )
        else:
            output_cols = [
                (i + "_imputed") for i in [e for e in list_of_cols if e in missing_cols]
            ]
            odf_print = missing_df.select(
                "attribute", F.col("missing_count").alias("missingCount_before")
            ).join(
                missingCount_computation(spark, odf, output_cols)
                .withColumnRenamed("attribute", "attribute_after")
                .withColumn(
                    "attribute",
                    F.expr("substring(attribute_after, 1, length(attribute_after)-8)"),
                )
                .drop("missing_pct"),
                "attribute",
                "inner",
            )
        odf_print.show(len(list_of_cols), False)
    return odf


def auto_imputation(
    spark,
    idf,
    list_of_cols="missing",
    drop_cols=[],
    id_col="",
    null_pct=0.1,
    stats_missing={},
    output_mode="replace",
    run_type="local",
    root_path="",
    auth_key="NA",
    print_impact=True,
):
    """
    auto_imputation tests for 5 imputation methods using the other imputation functions provided in this module
    and returns the one with the best performance. The 5 methods are: (1) imputation_MMM with method_type="mean" (2)
    imputation_MMM with method_type="median" (3) imputation_sklearn with method_type="KNN" (4) imputation_sklearn
    with method_type="regression" (5) imputation_matrixFactorization

    Samples without missing values in attributes to impute are used for testing by removing some % of values and impute
    them again using the above 5 methods. RMSE/attribute_mean is used as the evaluation metric for each attribute to
    reduce the effect of unit difference among attributes. The final error of a method is calculated by the sum of (
    RMSE/attribute_mean) for all numerical attributes to impute and the method with the least error will be selected.

    The above testing is only applicable for numerical attributes. If categorical attributes are included,
    they will be automatically imputed using imputation_MMM. In addition, if there is only one numerical attribute to
    impute, only method (1) and (2) will be tested because the rest of the methods require more than one column.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to impute e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all (non-array) columns for analysis.
        This is super useful instead of specifying all column names manually.
        "missing" (default) can be passed to include only those columns with missing values.
        One of the usecases where "all" may be preferable over "missing" is when the user wants to save
        the imputation model for the future use e.g. a column may not have missing value in the training
        dataset but missing values may possibly appear in the prediction dataset.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    id_col
        ID column (Default value = "")
    null_pct
        proportion of the valid input data to be replaced by None to form the test data (Default value = 0.1)
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_imputed" e.g. column X is appended as X_imputed. (Default value = "replace")
    run_type
        "local", "emr", "databricks", "ak8s" (Default value = "local")
    root_path
        This argument takes in a base folder path for writing out intermediate_data/ folder to. Default value is ""
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"
    print_impact
        True, False (Default value = False)
        This argument is to print out before and after missing counts of imputed columns. It also print the name of best
        performing imputation method along with RMSE details.

    Returns
    -------
    DataFrame
        Imputed Dataframe

    """

    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
        missing_df.write.parquet(
            root_path
            + "intermediate_data/imputation_comparison/missingCount_computation",
            mode="overwrite",
        )
        stats_missing = {
            "file_path": root_path
            + "intermediate_data/imputation_comparison/missingCount_computation",
            "file_type": "parquet",
        }
    else:
        missing_df = read_dataset(spark, **stats_missing).select(
            "attribute", "missing_count", "missing_pct"
        )

    empty_cols = (
        missing_df.where(F.col("missing_pct") == 1.0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    if len(empty_cols) > 0:
        warnings.warn("Following columns have all null values: " + ",".join(empty_cols))

    missing_cols = (
        missing_df.where(F.col("missing_count") > 0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    if list_of_cols == "all":
        list_of_cols = idf.columns
    if list_of_cols == "missing":
        list_of_cols = missing_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(
        set([e for e in list_of_cols if (e not in drop_cols) & (e != id_col)])
    )
    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    del_cols = [e for e in list_of_cols if e in empty_cols]
    odf_del = idf.drop(*del_cols)

    list_of_cols = [e for e in list_of_cols if e not in empty_cols]
    num_cols, cat_cols, other_cols = attributeType_segregation(
        odf_del.select(list_of_cols)
    )
    missing_catcols = [e for e in cat_cols if e in missing_cols]
    missing_numcols = [e for e in num_cols if e in missing_cols]

    if missing_catcols:
        odf_imputed_cat = imputation_MMM(
            spark, odf_del, list_of_cols=missing_catcols, stats_missing=stats_missing
        )
    else:
        odf_imputed_cat = odf_del

    if len(missing_numcols) == 0:
        warnings.warn(
            "No Imputation Performed for numerical columns - No Column(s) to Impute"
        )
        return odf_imputed_cat

    idf_test = (
        odf_imputed_cat.dropna(subset=missing_numcols)
        .withColumn("index", F.monotonically_increasing_id())
        .withColumn("index", F.row_number().over(Window.orderBy("index")))
    )
    null_count = int(null_pct * idf_test.count())
    idf_null = idf_test
    for i in missing_numcols:
        null_index = random.sample(range(idf_test.count()), null_count)
        idf_null = idf_null.withColumn(
            i, F.when(F.col("index").isin(null_index), None).otherwise(F.col(i))
        )

    idf_null.write.parquet(
        root_path + "intermediate_data/imputation_comparison/test_dataset",
        mode="overwrite",
    )
    idf_null = spark.read.parquet(
        root_path + "intermediate_data/imputation_comparison/test_dataset"
    )

    method1 = imputation_MMM(
        spark,
        idf_null,
        list_of_cols=missing_numcols,
        method_type="mean",
        stats_missing=stats_missing,
        output_mode=output_mode,
    )
    method2 = imputation_MMM(
        spark,
        idf_null,
        list_of_cols=missing_numcols,
        method_type="median",
        stats_missing=stats_missing,
        output_mode=output_mode,
    )
    valid_methods = [method1, method2]
    if len(num_cols) > 1:
        method3 = imputation_sklearn(
            spark,
            idf_null,
            list_of_cols=num_cols,
            method_type="KNN",
            stats_missing=stats_missing,
            output_mode=output_mode,
            auth_key=auth_key,
        )
        method4 = imputation_sklearn(
            spark,
            idf_null,
            list_of_cols=num_cols,
            method_type="regression",
            stats_missing=stats_missing,
            output_mode=output_mode,
            auth_key=auth_key,
        )
        method5 = imputation_matrixFactorization(
            spark,
            idf_null,
            list_of_cols=num_cols,
            id_col=id_col,
            stats_missing=stats_missing,
            output_mode=output_mode,
        )
        valid_methods = [method1, method2, method3, method4, method5]

    nrmse_all = []
    method_all = ["MMM-mean", "MMM-median", "KNN", "regression", "matrix_factorization"]

    for index, method in enumerate(valid_methods):
        nrmse = 0
        for i in missing_numcols:
            pred_col = (i + "_imputed") if output_mode == "append" else i
            idf_joined = (
                idf_test.select("index", F.col(i).alias("val"))
                .join(
                    method.select("index", F.col(pred_col).alias("pred")),
                    "index",
                    "left_outer",
                )
                .dropna()
            )
            idf_joined = recast_column(
                idf=idf_joined,
                list_of_cols=["val", "pred"],
                list_of_dtypes=["double", "double"],
            )
            pred_mean = float(
                method.select(F.mean(pred_col)).rdd.flatMap(lambda x: x).collect()[0]
            )
            i_nrmse = (
                RegressionEvaluator(
                    metricName="rmse", labelCol="val", predictionCol="pred"
                ).evaluate(idf_joined)
            ) / abs(pred_mean)
            nrmse += i_nrmse
        nrmse_all.append(nrmse)

    min_index = nrmse_all.index(np.min(nrmse_all))
    best_method = method_all[min_index]
    odf = valid_methods[min_index]

    if print_impact:
        print(list(zip(method_all, nrmse_all)))
        print("Best Imputation Method: ", best_method)
    return odf


def autoencoder_latentFeatures(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    reduction_params=0.5,
    sample_size=500000,
    epochs=100,
    batch_size=256,
    pre_existing_model=False,
    model_path="NA",
    standardization=True,
    standardization_configs={"pre_existing_model": False, "model_path": "NA"},
    imputation=False,
    imputation_configs={"imputation_function": "imputation_MMM"},
    stats_missing={},
    output_mode="replace",
    run_type="local",
    root_path="",
    auth_key="NA",
    print_impact=False,
):
    """
    Many machine learning models suffer from "the curse of dimensionality" when the number of features is too
    large. autoencoder_latentFeatures is able to reduce the dimensionality by compressing input attributes to a
    smaller number of latent features.

    To be more specific, it trains a neural network model using TensorFlow library. The neural network contains an
    encoder and a decoder, where the encoder learns to represent the input using smaller number of latent features
    controlled by the input *reduction_params* and the decoder learns to reproduce the input using the latent
    features. In the end, only the encoder will be kept and the latent features generated by the encoder will be
    added to the output dataframe.

    However, the neural network model is not trained in a scalable manner, which might not be able to handle large
    input dataframe. Thus, an input *sample_size* (the default value is 500,000) can be set to control the number of
    samples to be used to train the model. If the total number of samples exceeds *sample_size*, the rest of the
    samples will be predicted using the fitted encoder.

    Standardization is highly recommended if the input attributes are not of the same scale. Otherwise, the model
    might not converge smoothly. Inputs *standardization* and *standardization_configs* can be set accordingly to
    perform standardization within the function. In addition, if a sample contains missing values in the model input,
    the output values for its latent features will all be None. Thus data imputation is also recommended if missing
    values exist, which can be done within the function by setting inputs *imputation* and *imputation_configs*.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to encode e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    reduction_params
        Determines the number of encoded features in the result.
        If reduction_params < 1, int(reduction_params * <number of columns>)
        columns will be generated. Else, reduction_params columns will be generated. (Default value = 0.5)
    sample_size
        Maximum rows for training the autoencoder model using tensorflow. (Default value = 500000)
    epochs
        Number of epochs to train the tensorflow model. (Default value = 100)
    batch_size
        Number of samples per gradient update when fitting the tensorflow model. (Default value = 256)
    pre_existing_model
        Boolean argument – True or False. True if model exists already, False Otherwise (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    standardization
        Boolean argument – True or False. True, if the standardization required. (Default value = True)
    standardization_configs
        z_standardization function arguments in dictionary format. (Default value = {"pre_existing_model": False)
    imputation
        Boolean argument – True or False. True, if the imputation required. (Default value = False)
    imputation_configs
        Takes input in dictionary format.
        Imputation function name is provided with key "imputation_name".
        optional arguments pertaining to that imputation function can be provided with argument name as key. (Default value = {"imputation_function": "imputation_MMM"})
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed columns: latent_<col_index>.
        “append” option append transformed columns with format latent_<col_index> to the input dataset,
        e.g. latent_0, latent_1 will be appended if reduction_params=2. (Default value = "replace")
    run_type
        "local", "emr", "databricks", "ak8s" (Default value = "local")
    root_path
        This argument takes in a base folder path for writing out intermediate_data/ folder to. Default value is ""
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"
    print_impact
        True, False
        This argument is to print descriptive statistics of the latest features (Default value = False)

    Returns
    -------
    DataFrame
        Dataframe with Latent Features

    """
    if "arm64" in platform.version().lower():
        warnings.warn(
            "This function is currently not supported for ARM64 - Mac M1 Machine"
        )
        return idf

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn("No Latent Features Generated - No Column(s) to Transform")
        return idf

    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf, list_of_cols)
        missing_df.write.parquet(
            root_path
            + "intermediate_data/autoencoder_latentFeatures/missingCount_computation",
            mode="overwrite",
        )
        stats_missing = {
            "file_path": root_path
            + "intermediate_data/autoencoder_latentFeatures/missingCount_computation",
            "file_type": "parquet",
        }
    else:
        missing_df = (
            read_dataset(spark, **stats_missing)
            .select("attribute", "missing_count", "missing_pct")
            .where(F.col("attribute").isin(list_of_cols))
        )

    empty_cols = (
        missing_df.where(F.col("missing_pct") == 1.0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    if len(empty_cols) > 0:
        warnings.warn(
            "The following column(s) are excluded from dimensionality reduction as all values are null: "
            + ",".join(empty_cols)
        )
        list_of_cols = [e for e in list_of_cols if e not in empty_cols]

    if standardization:
        idf_standardized = z_standardization(
            spark,
            idf,
            list_of_cols=list_of_cols,
            output_mode="append",
            **standardization_configs,
        )
        list_of_cols_scaled = [
            i + "_scaled"
            for i in list_of_cols
            if (i + "_scaled") in idf_standardized.columns
        ]
    else:
        idf_standardized = idf
        for i in list_of_cols:
            idf_standardized = idf_standardized.withColumn(i + "_scaled", F.col(i))
            list_of_cols_scaled = [i + "_scaled" for i in list_of_cols]

    if imputation:
        all_functions = globals().copy()
        all_functions.update(locals())
        f = all_functions.get(imputation_configs["imputation_function"])
        args = copy.deepcopy(imputation_configs)
        args.pop("imputation_function", None)
        missing_df_scaled = (
            read_dataset(spark, **stats_missing)
            .select("attribute", "missing_count", "missing_pct")
            .withColumn("attribute", F.concat(F.col("attribute"), F.lit("_scaled")))
        )
        missing_df_scaled.write.parquet(
            root_path
            + "intermediate_data/autoencoder_latentFeatures/missingCount_computation_scaled",
            mode="overwrite",
        )
        stats_missing_scaled = {
            "file_path": root_path
            + "intermediate_data/autoencoder_latentFeatures/missingCount_computation_scaled",
            "file_type": "parquet",
        }
        idf_imputed = f(
            spark,
            idf_standardized,
            list_of_cols_scaled,
            stats_missing=stats_missing_scaled,
            **args,
        )
    else:
        idf_imputed = idf_standardized.dropna(subset=list_of_cols_scaled)

    n_inputs = len(list_of_cols_scaled)
    if reduction_params < 1:
        n_bottleneck = int(reduction_params * n_inputs)
    else:
        n_bottleneck = int(reduction_params)

    if pre_existing_model:
        if run_type == "emr":
            bash_cmd = (
                "aws s3 cp " + model_path + "/autoencoders_latentFeatures/encoder.h5 ."
            )
            output = subprocess.check_output(["bash", "-c", bash_cmd])
            bash_cmd = (
                "aws s3 cp " + model_path + "/autoencoders_latentFeatures/model.h5 ."
            )
            output = subprocess.check_output(["bash", "-c", bash_cmd])
            encoder = load_model("encoder.h5")
            model = load_model("model.h5")
        elif run_type == "ak8s":
            bash_cmd = (
                'azcopy cp "'
                + model_path
                + "/autoencoders_latentFeatures/encoder.h5"
                + str(auth_key)
                + '" .'
            )
            output = subprocess.check_output(["bash", "-c", bash_cmd])
            bash_cmd = (
                'azcopy cp "'
                + model_path
                + "/autoencoders_latentFeatures/model.h5"
                + str(auth_key)
                + '" .'
            )
            output = subprocess.check_output(["bash", "-c", bash_cmd])
            encoder = load_model("encoder.h5")
            model = load_model("model.h5")
        else:
            encoder = load_model(model_path + "/autoencoders_latentFeatures/encoder.h5")
            model = load_model(model_path + "/autoencoders_latentFeatures/model.h5")
    else:
        idf_valid = idf_imputed.select(list_of_cols_scaled)
        idf_model = idf_valid.sample(
            False, min(1.0, float(sample_size) / idf_valid.count()), 0
        )

        idf_train = idf_model.sample(False, 0.8, 0)
        idf_test = idf_model.subtract(idf_train)
        X_train = idf_train.toPandas()
        X_test = idf_test.toPandas()

        visible = Input(shape=(n_inputs,))
        e = Dense(n_inputs * 2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        bottleneck = Dense(n_bottleneck)(e)
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        d = Dense(n_inputs * 2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        output = Dense(n_inputs, activation="linear")(d)

        model = Model(inputs=visible, outputs=output)
        encoder = Model(inputs=visible, outputs=bottleneck)
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(
            X_train,
            X_train,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=2,
            validation_data=(X_test, X_test),
        )

        if (not pre_existing_model) & (model_path != "NA"):
            if run_type == "emr":
                encoder.save("encoder.h5")
                model.save("model.h5")
                bash_cmd = (
                    "aws s3 cp encoder.h5 "
                    + model_path
                    + "/autoencoders_latentFeatures/encoder.h5"
                )
                output = subprocess.check_output(["bash", "-c", bash_cmd])
                bash_cmd = (
                    "aws s3 cp model.h5 "
                    + model_path
                    + "/autoencoders_latentFeatures/model.h5"
                )
                output = subprocess.check_output(["bash", "-c", bash_cmd])
            elif run_type == "ak8s":
                encoder.save("encoder.h5")
                model.save("model.h5")
                bash_cmd = (
                    'azcopy cp "encoder.h5" "'
                    + model_path
                    + "/autoencoders_latentFeatures/encoder.h5"
                    + str(auth_key)
                    + '" '
                )
                output = subprocess.check_output(["bash", "-c", bash_cmd])
                bash_cmd = (
                    'azcopy cp "model.h5" "'
                    + model_path
                    + "/autoencoders_latentFeatures/model.h5"
                    + str(auth_key)
                    + '" '
                )
                output = subprocess.check_output(["bash", "-c", bash_cmd])
            else:
                if not os.path.exists(model_path + "/autoencoders_latentFeatures/"):
                    os.makedirs(model_path + "/autoencoders_latentFeatures/")
                encoder.save(model_path + "/autoencoders_latentFeatures/encoder.h5")
                model.save(model_path + "/autoencoders_latentFeatures/model.h5")

    class ModelWrapperPickable:
        def __init__(self, model):
            self.model = model

        def __getstate__(self):
            model_str = ""
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
                tensorflow.keras.models.save_model(self.model, fd.name, overwrite=True)
                model_str = fd.read()
            d = {"model_str": model_str}
            return d

        def __setstate__(self, state):
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
                fd.write(state["model_str"])
                fd.flush()
                self.model = tensorflow.keras.models.load_model(fd.name)

    model_wrapper = ModelWrapperPickable(encoder)

    def compute_output_pandas_udf(model_wrapper):
        @F.pandas_udf(returnType=T.ArrayType(T.DoubleType()))
        def predict_pandas_udf(*cols):
            X = pd.concat(cols, axis=1)
            return pd.Series(row.tolist() for row in model_wrapper.model.predict(X))

        return predict_pandas_udf

    odf = idf_imputed.withColumn(
        "predicted_output",
        compute_output_pandas_udf(model_wrapper)(*list_of_cols_scaled),
    )
    odf_schema = odf.schema
    for i in range(0, n_bottleneck):
        odf_schema = odf_schema.add(T.StructField("latent_" + str(i), T.FloatType()))
    odf = (
        odf.rdd.map(lambda x: (*x, *x["predicted_output"]))
        .toDF(schema=odf_schema)
        .drop("predicted_output")
    )

    odf = odf.drop(*list_of_cols_scaled)

    if output_mode == "replace":
        odf = odf.drop(*list_of_cols)

    if print_impact:
        output_cols = ["latent_" + str(j) for j in range(0, n_bottleneck)]
        odf.select(output_cols).describe().show(5, False)

    return odf


def PCA_latentFeatures(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    explained_variance_cutoff=0.95,
    pre_existing_model=False,
    model_path="NA",
    standardization=True,
    standardization_configs={"pre_existing_model": False, "model_path": "NA"},
    imputation=False,
    imputation_configs={"imputation_function": "imputation_MMM"},
    stats_missing={},
    output_mode="replace",
    run_type="local",
    root_path="",
    auth_key="NA",
    print_impact=False,
):
    """
    Similar to autoencoder_latentFeatures, PCA_latentFeatures also generates latent features which reduces the
    dimensionality of the input dataframe but through a different technique: Principal Component Analysis (PCA). PCA
    algorithm produces principal components such that it can describe most of the remaining variance and all the
    principal components are orthogonal to each other. The final number of generated principal components is
    controlled by the input *explained_variance_cutoff*. In other words, the number of selected principal components,
    k, is the minimum value such that top k components (latent features) can explain at least
    *explained_variance_cutoff* of the total variance.

    Standardization is highly recommended if the input attributes are not of the same scale. Otherwise the generated
    latent features might be dominated by the attributes with larger variance. Inputs *standardization* and
    *standardization_configs* can be set accordingly to perform standardization within the function. In addition,
    data imputation is also recommended if missing values exist, which can be done within the function by setting
    inputs *imputation* and *imputation_configs*.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to encode e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    explained_variance_cutoff
        Determines the number of latent columns in the output. If N is the smallest
        integer such that top N latent columns explain more than explained_variance_cutoff
        variance, these N columns will be selected. (Default value = 0.95)
    pre_existing_model
        Boolean argument – True or False. True if model exists already, False Otherwise (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for referring the pre-saved model.
        If pre_existing_model is False, this argument can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    standardization
        Boolean argument – True or False. True, if the standardization required. (Default value = True)
    standardization_configs
        z_standardization function arguments in dictionary format. (Default value = {"pre_existing_model": False)
    imputation
        Boolean argument – True or False. True, if the imputation required. (Default value = False)
    imputation_configs
        Takes input in dictionary format.
        Imputation function name is provided with key "imputation_name".
        optional arguments pertaining to that imputation function can be provided with argument name as key. (Default value = {"imputation_function": "imputation_MMM"})
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed columns: latent_<col_index>.
        “append” option append transformed columns with format latent_<col_index> to the input dataset,
        e.g. latent_0, latent_1. (Default value = "replace")
    run_type
        "local", "emr", "databricks", "ak8s" (Default value = "local")
    root_path
        This argument takes in a base folder path for writing out intermediate_data/ folder to. Default value is ""
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for ak8s run_type. Default value is kept as "NA"
    print_impact
        True, False
        This argument is to print descriptive statistics of the latest features (Default value = False)

    Returns
    -------
    DataFrame
        Dataframe with Latent Features

    """

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn("No Latent Features Generated - No Column(s) to Transform")
        return idf

    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf, list_of_cols)
        missing_df.write.parquet(
            root_path + "intermediate_data/PCA_latentFeatures/missingCount_computation",
            mode="overwrite",
        )
        stats_missing = {
            "file_path": root_path
            + "intermediate_data/PCA_latentFeatures/missingCount_computation",
            "file_type": "parquet",
        }
    else:
        missing_df = (
            read_dataset(spark, **stats_missing)
            .select("attribute", "missing_count", "missing_pct")
            .where(F.col("attribute").isin(list_of_cols))
        )

    empty_cols = (
        missing_df.where(F.col("missing_pct") == 1.0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    if len(empty_cols) > 0:
        warnings.warn(
            "The following column(s) are excluded from dimensionality reduction as all values are null: "
            + ",".join(empty_cols)
        )
        list_of_cols = [e for e in list_of_cols if e not in empty_cols]

    if standardization:
        idf_standardized = z_standardization(
            spark,
            idf,
            list_of_cols=list_of_cols,
            output_mode="append",
            **standardization_configs,
        )
        list_of_cols_scaled = [
            i + "_scaled"
            for i in list_of_cols
            if (i + "_scaled") in idf_standardized.columns
        ]
    else:
        idf_standardized = idf
        for i in list_of_cols:
            idf_standardized = idf_standardized.withColumn(i + "_scaled", F.col(i))
            list_of_cols_scaled = [i + "_scaled" for i in list_of_cols]

    if imputation:
        all_functions = globals().copy()
        all_functions.update(locals())
        f = all_functions.get(imputation_configs["imputation_function"])
        args = copy.deepcopy(imputation_configs)
        args.pop("imputation_function", None)
        missing_df_scaled = (
            read_dataset(spark, **stats_missing)
            .select("attribute", "missing_count", "missing_pct")
            .withColumn("attribute", F.concat(F.col("attribute"), F.lit("_scaled")))
        )
        missing_df_scaled.write.parquet(
            root_path
            + "intermediate_data/PCA_latentFeatures/missingCount_computation_scaled",
            mode="overwrite",
        )
        stats_missing_scaled = {
            "file_path": root_path
            + "intermediate_data/PCA_latentFeatures/missingCount_computation_scaled",
            "file_type": "parquet",
        }
        idf_imputed = f(
            spark,
            idf_standardized,
            list_of_cols_scaled,
            stats_missing=stats_missing_scaled,
            **args,
        )
    else:
        idf_imputed = idf_standardized.dropna(subset=list_of_cols_scaled)

    idf_imputed.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    assembler = VectorAssembler(inputCols=list_of_cols_scaled, outputCol="features")
    assembled_data = assembler.transform(idf_imputed).drop(*list_of_cols_scaled)

    if pre_existing_model:
        pca = PCA.load(model_path + "/PCA_latentFeatures/pca_path")
        pcaModel = PCAModel.load(model_path + "/PCA_latentFeatures/pcaModel_path")
        n = pca.getK()
    else:
        pca = PCA(
            k=len(list_of_cols_scaled), inputCol="features", outputCol="features_pca"
        )
        pcaModel = pca.fit(assembled_data)
        explained_variance = 0
        for n in range(1, len(list_of_cols) + 1):
            explained_variance += pcaModel.explainedVariance[n - 1]
            if explained_variance > explained_variance_cutoff:
                break

        pca = PCA(k=n, inputCol="features", outputCol="features_pca")
        pcaModel = pca.fit(assembled_data)
        if (not pre_existing_model) & (model_path != "NA"):
            pcaModel.write().overwrite().save(
                model_path + "/PCA_latentFeatures/pcaModel_path"
            )
            pca.write().overwrite().save(model_path + "/PCA_latentFeatures/pca_path")

    def vector_to_array(v):
        return v.toArray().tolist()

    f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.FloatType()))

    odf = (
        pcaModel.transform(assembled_data)
        .withColumn("features_pca_array", f_vector_to_array("features_pca"))
        .drop(*["features", "features_pca"])
    )

    odf_schema = odf.schema
    for i in range(0, n):
        odf_schema = odf_schema.add(T.StructField("latent_" + str(i), T.FloatType()))
    odf = (
        odf.rdd.map(lambda x: (*x, *x["features_pca_array"]))
        .toDF(schema=odf_schema)
        .drop("features_pca_array")
        .replace(float("nan"), None, subset=["latent_" + str(j) for j in range(0, n)])
    )

    if output_mode == "replace":
        odf = odf.drop(*list_of_cols)

    if print_impact:
        print("Explained Variance: ", round(np.sum(pcaModel.explainedVariance[0:n]), 4))
        output_cols = ["latent_" + str(j) for j in range(0, n)]
        odf.select(output_cols).describe().show(5, False)

    return odf


def feature_transformation(
    idf,
    list_of_cols="all",
    drop_cols=[],
    method_type="sqrt",
    N=None,
    output_mode="replace",
    print_impact=False,
):
    """
    As the name indicates, feature_transformation performs mathematical transformation over selected attributes.
    The following methods are supported for an input attribute x: ln(x), log10(x), log2(x), e^x, 2^x, 10^x, N^x,
    square and cube root of x, x^2, x^3, x^N, trigonometric transformations of x (sin, cos, tan, asin, acos, atan),
    radians, x%N, x!, 1/x, floor and ceiling of x and x rounded to N decimal places. Some transformations only work
    with positive or non-negative input values such as log and square root and an error will be returned if violated.


    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    method_type
        "ln", "log10", "log2", "exp", "powOf2" (2^x), "powOf10" (10^x), "powOfN" (N^x),
        "sqrt" (square root), "cbrt" (cube root), "sq" (square), "cb" (cube), "toPowerN" (x^N),
        "sin", "cos", "tan", "asin", "acos", "atan", "radians",
        "remainderDivByN" (x%N), "factorial" (x!), "mul_inv" (1/x),
        "floor", "ceil", "roundN" (round to N decimal places) (Default value = "sqrt")
    N
        None by default. If method_type is "powOfN", "toPowerN", "remainderDivByN" or "roundN", N will
        be used as the required constant.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed columns.
        “append” option append transformed columns with a postfix (E.g. "_ln", "_powOf<N>")
        to the input dataset. (Default value = "replace")
    print_impact
        True, False
        This argument is to print before and after descriptive statistics of the transformed features (Default value = False)

    Returns
    -------
    DataFrame
        Transformed Dataframe

    """

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if (len(list_of_cols) == 0) | (any(x not in num_cols for x in list_of_cols)):
        raise TypeError("Invalid input for Column(s)")

    if method_type not in [
        "ln",
        "log10",
        "log2",
        "exp",
        "powOf2",
        "powOf10",
        "powOfN",
        "sqrt",
        "cbrt",
        "sq",
        "cb",
        "toPowerN",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "radians",
        "remainderDivByN",
        "factorial",
        "mul_inv",
        "floor",
        "ceil",
        "roundN",
    ]:
        raise TypeError("Invalid input method_type")

    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols
    odf = idf

    transformation_function = {
        "ln": F.log,
        "log10": F.log10,
        "log2": F.log2,
        "exp": F.exp,
        "powOf2": (lambda x: F.pow(2.0, x)),
        "powOf10": (lambda x: F.pow(10.0, x)),
        "powOfN": (lambda x: F.pow(N, x)),
        "sqrt": F.sqrt,
        "cbrt": F.cbrt,
        "sq": (lambda x: x**2),
        "cb": (lambda x: x**3),
        "toPowerN": (lambda x: x**N),
        "sin": F.sin,
        "cos": F.cos,
        "tan": F.tan,
        "asin": F.asin,
        "acos": F.acos,
        "atan": F.atan,
        "radians": F.radians,
        "remainderDivByN": (lambda x: x % F.lit(N)),
        "factorial": F.factorial,
        "mul_inv": (lambda x: F.lit(1) / x),
        "floor": F.floor,
        "ceil": F.ceil,
        "roundN": (lambda x: F.round(x, N)),
    }

    def get_col_name(i):
        if output_mode == "replace":
            return i
        else:
            if method_type in ["powOfN", "toPowerN", "remainderDivByN", "roundN"]:
                return i + "_" + method_type[:-1] + str(N)
            else:
                return i + "_" + method_type

    output_cols = []
    for i in list_of_cols:
        modify_col = get_col_name(i)
        odf = odf.withColumn(modify_col, transformation_function[method_type](F.col(i)))
        output_cols.append(modify_col)

    if print_impact:
        print("Before:")
        idf.select(list_of_cols).describe().show(5, False)
        print("After:")
        odf.select(output_cols).describe().show(5, False)

    return odf


def boxcox_transformation(
    idf,
    list_of_cols="all",
    drop_cols=[],
    boxcox_lambda=None,
    output_mode="replace",
    print_impact=False,
):
    """
    Some machine learning algorithms require the input data to follow normal distributions. Thus, when the input
    data is too skewed, boxcox_transformation can be used to transform it into a more normal-like distribution. The
    transformed value of a sample x depends on a coefficient lambda:
    (1) if lambda = 0, x is transformed into log(x);
    (2) if lambda !=0, x is transformed into (x^lambda-1)/lambda.

    The value of lambda can be either specified by the user or automatically selected within the function.
    If lambda needs to be selected within the function, a range of values (0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25, 3, -3, 4, -4, 5, -5)
    will be tested and the lambda, which optimizes the KolmogorovSmirnovTest by PySpark with the theoretical distribution being normal,
    will be used to perform the transformation. Different lambda values can be assigned to different attributes but one attribute
    can only be assigned one lambda value.


    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    boxcox_lambda
        Lambda value for box_cox transormation.
        If boxcox_lambda is not None, it will be directly used for the transformation. It can be a
        (1) list: each element represents a lambda value for an attribute and the length of the list
        must be the same as the number of columns to transform.
        (2) int/float: all attributes will be assigned the same lambda value.
        Else, search for the best lambda among [1,-1,0.5,-0.5,2,-2,0.25,-0.25,3,-3,4,-4,5,-5]
        for each column and apply the transformation (Default value = None)
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed columns.
        “append” option append transformed columns with a postfix "_bxcx_<lambda>"
        to the input dataset. (Default value = "replace")
    print_impact
        True, False
        This argument is to print before and after descriptive statistics of the transformed features (Default value = False)

    Returns
    -------
    DataFrame
        Transformed Dataframe

    """

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if (len(list_of_cols) == 0) | (any(x not in num_cols for x in list_of_cols)):
        raise TypeError("Invalid input for Column(s)")

    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols
    odf = idf
    col_mins = idf.select([F.min(i) for i in list_of_cols])
    if any([i <= 0 for i in col_mins.rdd.flatMap(lambda x: x).collect()]):
        col_mins.show(1, False)
        raise ValueError("Data must be positive")

    if boxcox_lambda is not None:
        if isinstance(boxcox_lambda, (list, tuple)):
            if len(boxcox_lambda) != len(list_of_cols):
                raise TypeError("Invalid input for boxcox_lambda")
            elif not all([isinstance(l, (float, int)) for l in boxcox_lambda]):
                raise TypeError("Invalid input for boxcox_lambda")
            else:
                boxcox_lambda_list = list(boxcox_lambda)

        elif isinstance(boxcox_lambda, (float, int)):
            boxcox_lambda_list = [boxcox_lambda] * len(list_of_cols)
        else:
            raise TypeError("Invalid input for boxcox_lambda")

    else:
        boxcox_lambda_list = []
        for i in list_of_cols:
            lambdaVal = [1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25, 3, -3, 4, -4, 5, -5]
            best_pVal = 0
            for j in lambdaVal:
                pVal = Statistics.kolmogorovSmirnovTest(
                    odf.select(F.pow(F.col(i), j)).rdd.flatMap(lambda x: x), "norm"
                ).pValue
                if pVal > best_pVal:
                    best_pVal = pVal
                    best_lambdaVal = j

            pVal = Statistics.kolmogorovSmirnovTest(
                odf.select(F.log(F.col(i))).rdd.flatMap(lambda x: x), "norm"
            ).pValue
            if pVal > best_pVal:
                best_pVal = pVal
                best_lambdaVal = 0
            boxcox_lambda_list.append(best_lambdaVal)

    output_cols = []
    for i, curr_lambdaVal in zip(list_of_cols, boxcox_lambda_list):
        if curr_lambdaVal != 1:
            modify_col = (
                (i + "_bxcx_" + str(curr_lambdaVal)) if output_mode == "append" else i
            )
            output_cols.append(modify_col)
            if curr_lambdaVal == 0:
                odf = odf.withColumn(modify_col, F.log(F.col(i)))
            else:
                odf = odf.withColumn(modify_col, F.pow(F.col(i), curr_lambdaVal))
    if len(output_cols) == 0:
        warnings.warn(
            "lambdaVal for all columns are 1 so no transformation is performed and idf is returned"
        )
        return idf

    if print_impact:
        print("Transformed Columns: ", list_of_cols)
        print("Best BoxCox Parameter(s): ", boxcox_lambda_list)
        print("Before:")
        idf.select(list_of_cols).describe().unionByName(
            idf.select([F.skewness(i).alias(i) for i in list_of_cols]).withColumn(
                "summary", F.lit("skewness")
            )
        ).show(6, False)
        print("After:")
        if output_mode == "replace":
            odf.select(list_of_cols).describe().unionByName(
                odf.select([F.skewness(i).alias(i) for i in list_of_cols]).withColumn(
                    "summary", F.lit("skewness")
                )
            ).show(6, False)
        else:
            output_cols = [("`" + i + "`") for i in output_cols]
            odf.select(output_cols).describe().unionByName(
                odf.select(
                    [F.skewness(i).alias(i[1:-1]) for i in output_cols]
                ).withColumn("summary", F.lit("skewness"))
            ).show(6, False)

    return odf


def outlier_categories(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    coverage=1.0,
    max_category=50,
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    print_impact=False,
):
    """
    This function replaces less frequently seen values (called as outlier values in the current context) in a
    categorical column by 'outlier_categories'. Outlier values can be defined in two ways –
    a) Max N categories, where N is user defined value. In this method, top N-1 frequently seen categories are considered
    and rest are clubbed under single category 'outlier_categories'. or Alternatively,
    b) Coverage – top frequently seen categories are considered till it covers minimum N% of rows and rest lesser seen values
    are mapped to 'outlier_categories'. Even if the Coverage is less, maximum category constraint is given priority. Further,
    there is a caveat that when multiple categories have same rank. Then, number of categorical values can be more than
    max_category defined by the user.

    This function performs better when distinct values, in any column, are not more than 100. It is recommended that to drop those
    columns from the computation which has more than 100 distinct values, to get better performance out of this function.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
        list_of_cols
        List of categorical columns to transform e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all categorical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    coverage
        Defines the minimum % of rows that will be mapped to actual category name and the rest to be mapped
        to 'outlier_categories' and takes value between 0 to 1. Coverage of 0.8 can be interpreted as top frequently seen
        categories are considered till it covers minimum 80% of rows and rest lesser seen values are mapped to 'outlier_categories'. (Default value = 1.0)
    max_category
        Even if coverage is less, only (max_category - 1) categories will be mapped to actual name and rest to 'outlier_categories'.
        Caveat is when multiple categories have same rank, then #categories can be more than max_category. (Default value = 50)
    pre_existing_model
        Boolean argument – True or False. True if the model with the outlier/other values
        for each attribute exists already to be used, False Otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for the pre-saved model.
        If pre_existing_model is False, this field can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    output_mode
        "replace", "append".
        “replace” option replaces original columns with transformed column. “append” option append transformed
        column to the input dataset with a postfix "_outliered" e.g. column X is appended as X_outliered. (Default value = "replace")
    print_impact
        True, False
        This argument is to print before and after unique count of the transformed features (Default value = False)

    Returns
    -------
    DataFrame
        Transformed Dataframe

    """
    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == "all":
        list_of_cols = cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in cat_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if len(list_of_cols) == 0:
        warnings.warn(
            "No Outlier Categories Computation - No categorical column(s) to transform"
        )
        return idf
    if (coverage <= 0) | (coverage > 1):
        raise TypeError("Invalid input for Coverage Value")
    if max_category < 2:
        raise TypeError("Invalid input for Maximum No. of Categories Allowed")
    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")

    idf = idf.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    if pre_existing_model:
        df_model = spark.read.csv(
            model_path + "/outlier_categories", header=True, inferSchema=True
        )
    else:
        for index, i in enumerate(list_of_cols):
            window = Window.partitionBy().orderBy(F.desc("count_pct"))
            df_cats = (
                idf.select(i)
                .groupBy(i)
                .count()
                .dropna()
                .withColumn(
                    "count_pct",
                    F.col("count") / F.sum("count").over(Window.partitionBy()),
                )
                .withColumn("rank", F.rank().over(window))
                .withColumn(
                    "cumu",
                    F.sum("count_pct").over(
                        window.rowsBetween(Window.unboundedPreceding, 0)
                    ),
                )
                .withColumn("lag_cumu", F.lag("cumu").over(window))
                .fillna(0)
                .where(~((F.col("cumu") >= coverage) & (F.col("lag_cumu") >= coverage)))
                .where(F.col("rank") <= (max_category - 1))
                .select(F.lit(i).alias("attribute"), F.col(i).alias("parameters"))
            )
            if index == 0:
                df_model = df_cats
            else:
                df_model = df_model.union(df_cats)

    df_params = df_model.rdd.groupByKey().mapValues(list).collect()
    dict_params = dict(df_params)
    broadcast_params = spark.sparkContext.broadcast(dict_params)

    def get_params(key):
        parameters = list()
        dict_params = broadcast_params.value
        params = dict_params.get(key)
        if params:
            parameters = params
        return parameters

    odf = idf
    for i in list_of_cols:
        parameters = get_params(i)
        if output_mode == "replace":
            odf = odf.withColumn(
                i,
                F.when(
                    (F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)
                ).otherwise("outlier_categories"),
            )
        else:
            odf = odf.withColumn(
                i + "_outliered",
                F.when(
                    (F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)
                ).otherwise("outlier_categories"),
            )

    if (not pre_existing_model) & (model_path != "NA"):
        df_model.repartition(1).write.csv(
            model_path + "/outlier_categories", header=True, mode="overwrite"
        )

    if print_impact:
        if output_mode == "replace":
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_outliered") for i in list_of_cols]
        uniqueCount_computation(spark, idf, list_of_cols).select(
            "attribute", F.col("unique_values").alias("uniqueValues_before")
        ).show(len(list_of_cols), False)
        uniqueCount_computation(spark, odf, output_cols).select(
            "attribute", F.col("unique_values").alias("uniqueValues_after")
        ).show(len(list_of_cols), False)

    idf.unpersist()

    return odf


def expression_parser(idf, list_of_expr, postfix="", print_impact=False):
    """
    expression_parser can be used to evaluate a list of SQL expressions and output the result as new features. It
    is able to handle column names containing special characters such as “.”, “-”, “@”, “^”, etc, by converting them
    to “_” first before the evaluation and convert them back to the original names before returning the output
    dataframe.


    Parameters
    ----------
    idf
        Input Dataframe
    list_of_expr
        List of expressions to evaluate as new features e.g., ["expr1","expr2"].
        Alternatively, expressions can be specified in a string format,
        where different expressions are separated by pipe delimiter “|” e.g., "expr1|expr2".
    postfix
        postfix for new feature name.Naming convention "f" + expression_index + postfix
        e.g. with postfix of "new", new added features are named as f0new, f1new etc. (Default value = "")
    print_impact
        True, False
        This argument is to print the descriptive statistics of the parsed features (Default value = False)

    Returns
    -------
    DataFrame
        Parsed Dataframe

    """
    if isinstance(list_of_expr, str):
        list_of_expr = [x.strip() for x in list_of_expr.split("|")]

    special_chars = [
        "&",
        "$",
        ";",
        ":",
        ",",
        "*",
        "#",
        "@",
        "?",
        "%",
        "!",
        "^",
        "(",
        ")",
        "-",
        "/",
        "'",
        ".",
        '"',
    ]
    rename_cols = []
    replace_chars = {}
    for char in special_chars:
        for col in idf.columns:
            if char in col:
                rename_cols.append(col)
                if col in replace_chars.keys():
                    (replace_chars[col]).append(char)
                else:
                    replace_chars[col] = [char]

    rename_mapping_to_new, rename_mapping_to_old = {}, {}
    idf_renamed = idf
    for col in rename_cols:
        new_col = col
        for char in replace_chars[col]:
            new_col = new_col.replace(char, "_")
        rename_mapping_to_old[new_col] = col
        rename_mapping_to_new[col] = new_col

        idf_renamed = idf_renamed.withColumnRenamed(col, new_col)

    list_of_expr_ = []
    for expr in list_of_expr:
        new_expr = expr
        for col in rename_cols:
            if col in expr:
                new_expr = new_expr.replace(col, rename_mapping_to_new[col])
        list_of_expr_.append(new_expr)

    list_of_expr = list_of_expr_

    odf = idf_renamed
    new_cols = []
    for index, exp in enumerate(list_of_expr):
        odf = odf.withColumn("f" + str(index) + postfix, F.expr(exp))
        new_cols.append("f" + str(index) + postfix)

    for new_col, col in rename_mapping_to_old.items():
        odf = odf.withColumnRenamed(new_col, col)

    if print_impact:
        print("Columns Added: ", new_cols)
        odf.select(new_cols).describe().show(5, False)

    return odf
