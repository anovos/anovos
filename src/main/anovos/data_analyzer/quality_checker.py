# coding=utf-8
"""
This submodule focuses on assessing the data quality at both row-level and column-level and also provides an
appropriate treatment option to fix quality issues.

At the row level, the following checks are done:

- duplicate_detection
- nullRows_detection

At the column level, the following checks are done:

- nullColumns_detection
- outlier_detection
- IDness_detection
- biasedness_detection
- invalidEntries_detection

"""
import copy
import functools
import pandas as pd
import re
import warnings

from pyspark.sql import functions as F
from pyspark.sql import types as T

from anovos.data_analyzer.stats_generator import (
    measures_of_cardinality,
    missingCount_computation,
    mode_computation,
    uniqueCount_computation,
)
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.transformers import (
    auto_imputation,
    imputation_matrixFactorization,
    imputation_MMM,
    imputation_sklearn,
)
from anovos.shared.utils import (
    attributeType_segregation,
    get_dtype,
    transpose_dataframe,
)


def duplicate_detection(
    spark, idf, list_of_cols="all", drop_cols=[], treatment=True, print_impact=False
):
    """
    As the name implies, this function detects duplication in the input dataset. This means, for a pair of
    duplicate rows, the values in each column coincide. Duplication check is confined to the list of columns passed
    in the arguments. As part of treatment, duplicated rows are removed. This function returns two dataframes in
    tuple format; the 1st dataframe is the input dataset after deduplication (if treated else the original dataset).
    The 2nd dataframe is of schema – metric, value and contains the total number of rows, number of unique rows,
    number of duplicate rows and percentage of duplicate rows in total.


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
    treatment
        Boolean argument – True or False. If True, duplicate rows are removed from the input dataframe. (Default value = True)
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)

    Returns
    -------
    if print_impact is True:
        odf : DataFrame
            de-duplicated dataframe if treated, else original input dataframe.
        odf_print : DataFrame
            schema [metric, value] and contains metrics - number of rows, number of unique rows,
            number of duplicate rows and percentage of duplicate rows in total.
    if print_impact is False:
        odf : DataFrame
            de-duplicated dataframe if treated, else original input dataframe.
    """
    if not treatment and not print_impact:
        warnings.warn(
            "The original idf will be the only output. Set print_impact=True to perform detection without treatment"
        )
        return idf
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
    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")

    odf_tmp = idf.groupby(list_of_cols).count().drop("count")
    odf = odf_tmp if treatment else idf

    if print_impact:
        idf_count = idf.count()
        odf_tmp_count = odf_tmp.count()
        odf_print = spark.createDataFrame(
            [
                ["rows_count", float(idf_count)],
                ["unique_rows_count", float(odf_tmp_count)],
                ["duplicate_rows", float(idf_count - odf_tmp_count)],
                ["duplicate_pct", round((idf_count - odf_tmp_count) / idf_count, 4)],
            ],
            schema=["metric", "value"],
        )

        print("No. of Rows: " + str(idf_count))
        print("No. of UNIQUE Rows: " + str(odf_tmp_count))
        print("No. of Duplicate Rows: " + str(idf_count - odf_tmp_count))
        print(
            "Percentage of Duplicate Rows: "
            + str(round((idf_count - odf_tmp_count) / idf_count, 4))
        )

    if print_impact:
        return odf, odf_print
    else:
        return odf


def nullRows_detection(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    treatment=False,
    treatment_threshold=0.8,
    print_impact=False,
):
    """
    This function inspects the row quality and computes the number of columns that are missing for a row. This
    metric is further aggregated to check how many columns are missing for how many rows (or % rows). Intuition
    is if too many columns are missing for a row, removing it from the modeling may give better results than relying
    on its imputed values. Therefore as part of the treatment, rows with missing columns above the specified
    threshold are removed. This function returns two dataframes in tuple format; the 1st dataframe is the input
    dataset after filtering rows with a high number of missing columns (if treated else the original dataframe).
    The 2nd dataframe is of schema – null_cols_count, row_count, row_pct, flagged/treated.

    | null_cols_count | row_count | row_pct | flagged |
    |-----------------|-----------|---------|---------|
    | 5               | 11        | 3.0E-4  | 0       |
    | 7               | 1306      | 0.0401  | 1       |


    Interpretation: 1306 rows (4.01% of total rows) have 7 missing columns and flagged for are removal because
    null_cols_count is above the threshold. If treatment is True, then flagged column is renamed as treated to
    show rows which has been removed.

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
    treatment
        Boolean argument – True or False. If True, rows with high no. of null columns (defined by
        treatment_threshold argument) are removed from the input dataframe. (Default value = False)
    treatment_threshold
        Defines % of columns allowed to be Null per row and takes value between 0 to 1.
        If % of null columns is above the threshold for a row, it is removed from the dataframe.
        There is no row removal if the threshold is 1.0. And if the threshold is 0, all rows with
        null value are removed. (Default value = 0.8)
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)

    Returns
    -------
    odf : DataFrame
        Dataframe after row removal if treated, else original input dataframe.
    odf_print : DataFrame
        schema [null_cols_count, row_count, row_pct, flagged/treated].
        null_cols_count is defined as no. of missing columns in a row.
        row_count is no. of rows with null_cols_count missing columns.
        row_pct is row_count divided by number of rows.
        flagged/treated is 1 if null_cols_count is more than (threshold  X Number of Columns), else 0.

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

    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")

    treatment_threshold = float(treatment_threshold)
    if (treatment_threshold < 0) | (treatment_threshold > 1):
        raise TypeError("Invalid input for Treatment Threshold Value")

    def null_count(*cols):
        return cols.count(None)

    f_null_count = F.udf(null_count, T.LongType())

    odf_tmp = idf.withColumn("null_cols_count", f_null_count(*list_of_cols)).withColumn(
        "flagged",
        F.when(
            F.col("null_cols_count") > (len(list_of_cols) * treatment_threshold), 1
        ).otherwise(0),
    )

    if treatment_threshold == 1:
        odf_tmp = odf_tmp.withColumn(
            "flagged",
            F.when(F.col("null_cols_count") == len(list_of_cols), 1).otherwise(0),
        )

    odf_print = (
        odf_tmp.groupBy("null_cols_count", "flagged")
        .agg(F.count(F.lit(1)).alias("row_count"))
        .withColumn("row_pct", F.round(F.col("row_count") / float(idf.count()), 4))
        .select("null_cols_count", "row_count", "row_pct", "flagged")
        .orderBy("null_cols_count")
    )

    if treatment:
        odf = odf_tmp.where(F.col("flagged") == 0).drop(*["null_cols_count", "flagged"])
        odf_print = odf_print.withColumnRenamed("flagged", "treated")
    else:
        odf = idf

    if print_impact:
        odf_print.show(odf.count())

    return odf, odf_print


def nullColumns_detection(
    spark,
    idf,
    list_of_cols="missing",
    drop_cols=[],
    treatment=False,
    treatment_method="row_removal",
    treatment_configs={},
    stats_missing={},
    stats_unique={},
    stats_mode={},
    print_impact=False,
):
    """
    This function inspects the column quality and computes the number of rows that are missing for a column. This
    function also leverages statistics computed as part of the State Generator module. Statistics are not computed
    twice if already available.

    As part of treatments, it currently supports the following methods – Mean Median Mode (MMM), row_removal,
    column_removal, KNN, regression, Matrix Factorization (MF), auto imputation (auto).
    - MMM replaces null value with the measure of central tendency (mode for categorical features and mean/median for numerical features).
    - row_removal removes all rows with any missing value (output of this treatment is same as nullRows_detection with treatment_threshold of 0).
    - column_removal remove a column if %rows with a missing value is above treatment_threshold.
    - KNN/regression create an imputation model for every to-be-imputed column based on the rest of columns in the list_of_cols columns.
      KNN leverages sklearn.impute.KNNImputer and regression sklearn.impute.IterativeImputer. Since sklearn algorithms are not
      scalable, we create imputation model on sample dataset and apply that model on the whole dataset in distributed manner using
      pyspark pandas udf.
    - Matrix Factorization leverages pyspark.ml.recommendation.ALS algorithm.
    - auto imputation compares all imputation methods and select the best imputation method based on the least RMSE.

    This function returns two dataframes in tuple format – 1st dataframe is input dataset after imputation (if
    treated else the original dataset) and  2nd dataframe is of schema – attribute, missing_count, missing_pct.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to inspect e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all (non-array) columns for analysis. This is super useful instead of specifying all column names manually.
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
    treatment
        Boolean argument – True or False. If True, missing values are treated as per treatment_method argument. (Default value = False)
    treatment_method
        "MMM", "row_removal", "column_removal", "KNN", "regression", "MF", "auto".
        (Default value = "row_removal")
    treatment_configs
        Takes input in dictionary format.
        For column_removal treatment, key ‘treatment_threshold’ is provided with a value between 0 to 1 (remove column
        if % of rows with missing value is above this threshold)
        For row_removal, this argument can be skipped.
        For MMM, arguments corresponding to imputation_MMM function (transformer module) are provided,
        where each key is an argument from imputation_MMM function.
        For KNN, arguments corresponding to imputation_sklearn function (transformer module) are provided,
        where each key is an argument from imputation_sklearn function. method_type should be "KNN"
        For regression, arguments corresponding to imputation_sklearn function (transformer module) are provided,
        where each key is an argument from imputation_sklearn function. method_type should be "regression"
        For MF, arguments corresponding to imputation_matrixFactorization function (transformer module) are provided,
        where each key is an argument from imputation_matrixFactorization function.
        For auto, arguments corresponding to auto_imputation function (transformer module) are provided,
        where each key is an argument from auto_imputation function. (Default value = {})
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    stats_mode
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
        mode_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    print_impact
        True, False
        This argument is to print out the statistics or the impact of imputation (if applicable).(Default value = False)

    Returns
    -------
    odf : DataFrame
        Imputed dataframe if treated, else original input dataframe.
    odf_print : DataFrame
        schema [attribute, missing_count, missing_pct].
        missing_count is number of rows with null values for an attribute, and
        missing_pct is missing_count divided by number of rows.
    """
    if stats_missing == {}:
        odf_print = missingCount_computation(spark, idf)
    else:
        odf_print = read_dataset(spark, **stats_missing).select(
            "attribute", "missing_count", "missing_pct"
        )

    missing_cols = (
        odf_print.where(F.col("missing_count") > 0)
        .select("attribute")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols

    if list_of_cols == "missing":
        list_of_cols = missing_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if len(list_of_cols) == 0:
        warnings.warn("No Null Detection - No column(s) to analyze")
        odf = idf
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("missing_count", T.StringType(), True),
                T.StructField("missing_pct", T.StringType(), True),
            ]
        )
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print

    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")
    if treatment_method not in (
        "MMM",
        "row_removal",
        "column_removal",
        "KNN",
        "regression",
        "MF",
        "auto",
    ):
        raise TypeError("Invalid input for method_type")

    treatment_threshold = treatment_configs.pop("treatment_threshold", None)
    if treatment_threshold:
        treatment_threshold = float(treatment_threshold)
    else:
        if treatment_method == "column_removal":
            raise TypeError("Invalid input for column removal threshold")

    odf_print = odf_print.where(F.col("attribute").isin(list_of_cols))

    if treatment:
        if treatment_threshold:
            threshold_cols = (
                odf_print.where(F.col("attribute").isin(list_of_cols))
                .where(F.col("missing_pct") > treatment_threshold)
                .select("attribute")
                .rdd.flatMap(lambda x: x)
                .collect()
            )

        if treatment_method == "column_removal":
            odf = idf.drop(*threshold_cols)
            if print_impact:
                odf_print.show(len(list_of_cols))
                print("Removed Columns: ", threshold_cols)

        if treatment_method == "row_removal":
            remove_cols = (
                odf_print.where(F.col("attribute").isin(list_of_cols))
                .where(F.col("missing_pct") == 1.0)
                .select("attribute")
                .rdd.flatMap(lambda x: x)
                .collect()
            )
            list_of_cols = [e for e in list_of_cols if e not in remove_cols]
            if treatment_threshold:
                list_of_cols = [e for e in threshold_cols if e not in remove_cols]
            odf = idf.dropna(subset=list_of_cols)

            if print_impact:
                odf_print.show(len(list_of_cols))
                print("Before Count: " + str(idf.count()))
                print("After Count: " + str(odf.count()))

        if treatment_method == "MMM":
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
            list_of_cols = [e for e in list_of_cols if e not in remove_cols]
            if treatment_threshold:
                list_of_cols = [e for e in threshold_cols if e not in remove_cols]
            odf = imputation_MMM(
                spark,
                idf,
                list_of_cols,
                **treatment_configs,
                stats_missing=stats_missing,
                stats_mode=stats_mode,
                print_impact=print_impact
            )

        if treatment_method in ("KNN", "regression", "MF", "auto"):

            if treatment_threshold:
                list_of_cols = threshold_cols
            list_of_cols = [e for e in list_of_cols if e in num_cols]
            func_mapping = {
                "KNN": imputation_sklearn,
                "regression": imputation_sklearn,
                "MF": imputation_matrixFactorization,
                "auto": auto_imputation,
            }
            func = func_mapping[treatment_method]
            odf = func(
                spark,
                idf,
                list_of_cols,
                **treatment_configs,
                stats_missing=stats_missing,
                print_impact=print_impact
            )

    else:
        odf = idf
        if print_impact:
            odf_print.show(len(list_of_cols))

    return odf, odf_print


def outlier_detection(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    detection_side="upper",
    detection_configs={
        "pctile_lower": 0.05,
        "pctile_upper": 0.95,
        "stdev_lower": 3.0,
        "stdev_upper": 3.0,
        "IQR_lower": 1.5,
        "IQR_upper": 1.5,
        "min_validation": 2,
    },
    treatment=True,
    treatment_method="value_replacement",
    pre_existing_model=False,
    model_path="NA",
    sample_size=1000000,
    output_mode="replace",
    print_impact=False,
):
    """
    In Machine Learning, outlier detection identifies values that deviate drastically from the rest of the
    attribute values. An outlier may be caused simply by chance, measurement error, or inherent heavy-tailed
    distribution. This function identifies extreme values in both directions (or any direction provided by the user
    via detection_side argument). By default, outlier is identified by 3 different methodologies and tagged an outlier
    only if it is validated by at least 2 methods. Users can customize the methodologies they would like to apply and
    the minimum number of methodologies to be validated under detection_configs argument.

    - Percentile Method: In this methodology, a value higher than a certain (default 95th) percentile value is considered
      as an outlier. Similarly, a value lower than a certain (default 5th) percentile value is considered as an outlier.

    - Standard Deviation Method: In this methodology, if a value is a certain number of standard deviations (default 3)
      away from the mean, it is identified as an outlier.

    - Interquartile Range (IQR) Method: if a value is a certain number of IQRs (default 1.5) below Q1 or above Q3,
    it is identified as an outlier. Q1 is in first quantile/25th percentile, Q3 is in third quantile/75th percentile,
    and IQR is the difference between third quantile & first quantile.

    As part of treatments available, outlier values can be replaced by null so that it can be imputed by a reliable
    imputation methodology (null_replacement). It can also be replaced by maximum or minimum permissible by above
    methodologies (value_replacement). Lastly, rows can be removed if it is identified with any outlier (row_removal).

    This function returns two dataframes in tuple format – 1st dataframe is input dataset after treating outlier (the
    original dataset if no treatment) and  2nd dataframe is of schema – attribute, lower_outliers, upper_outliers. If
    outliers are checked only for upper end, then lower_outliers column will be shown all zero. Similarly if checked
    only for lower end, then upper_outliers will be zero for all attributes.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of numerical columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all numerical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    detection_side
        "upper", "lower", "both".
        "lower" detects outliers in the lower spectrum of the column range, whereas "upper" detects in the upper spectrum.
        "both" detects in both upper and lower end of the spectrum. (Default value = "upper")
    detection_configs
        Takes input in dictionary format with keys representing upper & lower parameter for
        three outlier detection methodologies.
        a) Percentile Method: lower and upper percentile threshold can be set via "pctile_lower" & "pctile_upper" (default 0.05 & 0.95)
        Any value above "pctile_upper" is considered as an outlier. Similarly, a value lower than "pctile_lower" is considered as an outlier.
        b) Standard Deviation Method: In this methodology, if a value which is below (mean - "stdev_lower" * standard deviation) or above
        (mean + "stdev_upper" * standard deviation), then it is identified as an outlier (default 3.0 & 3.0).
        c) Interquartile Range (IQR) Method: A value which is below (Q1 – "IQR_lower" * IQR) or above (Q3 + "IQR_lower" * IQR)
        is identified as outliers, where Q1 is first quartile/25th percentile, Q3 is third quartile/75th percentile and IQR is difference between
        third quartile & first quartile (default 1.5 & 1.5).
        If an attribute value is less (more) than its derived lower (upper) bound value, it is considered as outlier by a methodology.
        A attribute value is considered as outlier if it is declared as outlier by at least 'min_validation' methodologies (default 2).
        If 'min_validation' is not specified, the total number of methodologies will be used. In addition, it cannot be larger than the
        total number of methodologies applied.
        If detection_side is "upper", then "pctile_lower", "stdev_lower" and "IQR_lower" will be ignored and vice versa.
        Examples (detection_side = "lower")
        - If detection_configs={"pctile_lower": 0.05, "stdev_lower": 3.0, "min_validation": 1}, Percentile and Standard Deviation
        methods will be applied and a value is considered as outlier if at least 1 methodology categorizes it as an outlier.
        - If detection_configs={"pctile_lower": 0.05, "stdev_lower": 3.0}, since "min_validation" is not specified, 2 will be used
        because there are 2 methodologies specified. A value is considered as outlier if at both 2 methodologies categorize it as an outlier.
    treatment
        Boolean argument - True or False. If True, outliers are treated as per treatment_method argument.
        If treatment is False, print_impact should be True to perform detection without treatment. (Default value = True)
    treatment_method
        "null_replacement", "row_removal", "value_replacement".
        In "null_replacement", outlier values are replaced by null so that it can be imputed by a
        reliable imputation methodology. In "value_replacement", outlier values are replaced by
        maximum or minimum permissible value by above methodologies. Lastly in "row_removal", rows
        are removed if it is found with any outlier. (Default value = "value_replacement")
    pre_existing_model
        Boolean argument – True or False. True if the model with upper/lower permissible values
        for each attribute exists already to be used, False otherwise. (Default value = False)
    model_path
        If pre_existing_model is True, this argument is path for the pre-saved model.
        If pre_existing_model is False, this field can be used for saving the model.
        Default "NA" means there is neither pre-existing model nor there is a need to save one.
    sample_size
        The maximum number of rows used to calculate the thresholds of outlier detection. Relevant computation includes
        percentiles, means, standard deviations and quantiles calculation. The computed thresholds will be applied over
        all rows in the original idf to detect outliers. If the number of rows of idf is smaller than sample_size,
        the original idf will be used. (Default value = 1000000)
    output_mode
        "replace", "append".
        “replace” option replaces original columns with treated column. “append” option append treated
        column to the input dataset with a postfix "_outliered" e.g. column X is appended as X_outliered. (Default value = "replace")
    print_impact
        True, False
        This argument is to calculate and print out the impact of treatment (if applicable).
        If treatment is False, print_impact should be True to perform detection without treatment. (Default value = False).

    Returns
    -------
    if print_impact is True:
        odf : DataFrame
            Dataframe with outliers treated if treatment is True, else original input dataframe.
        odf_print : DataFrame
            schema [attribute, lower_outliers, upper_outliers, excluded_due_to_skewness].
            lower_outliers is no. of outliers found in the lower spectrum of the attribute range,
            upper_outliers is outlier count in the upper spectrum, and
            excluded_due_to_skewness is 0 or 1 indicating whether an attribute is excluded from detection due to skewness.
    if print_impact is False:
        odf : DataFrame
            Dataframe with outliers treated if treatment is True, else original input dataframe.
    """
    column_order = idf.columns
    num_cols = attributeType_segregation(idf)[0]

    if not treatment and not print_impact:
        if (not pre_existing_model and model_path == "NA") | pre_existing_model:
            warnings.warn(
                "The original idf will be the only output. Set print_impact=True to perform detection without treatment"
            )
            return idf
    if list_of_cols == "all":
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    schema = T.StructType(
        [
            T.StructField("attribute", T.StringType(), True),
            T.StructField("lower_outliers", T.StringType(), True),
            T.StructField("upper_outliers", T.StringType(), True),
        ]
    )
    empty_odf_print = spark.sparkContext.emptyRDD().toDF(schema)

    if not list_of_cols:
        warnings.warn("No Outlier Check - No numerical column to analyze")
        if print_impact:
            empty_odf_print.show()
            return idf, empty_odf_print
        else:
            return idf

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if detection_side not in ("upper", "lower", "both"):
        raise TypeError("Invalid input for detection_side")
    if treatment_method not in ("null_replacement", "row_removal", "value_replacement"):
        raise TypeError("Invalid input for treatment_method")
    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")
    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")
    if str(pre_existing_model).lower() == "true":
        pre_existing_model = True
    elif str(pre_existing_model).lower() == "false":
        pre_existing_model = False
    else:
        raise TypeError("Non-Boolean input for pre_existing_model")
    for arg in ["pctile_lower", "pctile_upper"]:
        if arg in detection_configs:
            if (detection_configs[arg] < 0) | (detection_configs[arg] > 1):
                raise TypeError("Invalid input for " + arg)

    if pre_existing_model:
        df_model = spark.read.parquet(model_path + "/outlier_numcols")
        model_dict_list = (
            df_model.where(F.col("attribute").isin(list_of_cols))
            .rdd.map(lambda row: {row[0]: row[1]})
            .collect()
        )
        model_dict = {}
        for d in model_dict_list:
            model_dict.update(d)

        params = []
        present_cols, skewed_cols = [], []
        for i in list_of_cols:
            param = model_dict.get(i)
            if param:
                if "skewed_attribute" in param:
                    skewed_cols.append(i)
                else:
                    param = [float(p) if p else p for p in param]
                    params.append(param)
                    present_cols.append(i)

        diff_cols = list(set(list_of_cols) - set(present_cols) - set(skewed_cols))
        if diff_cols:
            warnings.warn("Columns not found in model_path: " + ",".join(diff_cols))
        if skewed_cols:
            warnings.warn(
                "Columns excluded from outlier detection due to highly skewed distribution: "
                + ",".join(skewed_cols)
            )
        list_of_cols = present_cols

        if not list_of_cols:
            warnings.warn("No Outlier Check - No numerical column to analyze")
            if print_impact:
                empty_odf_print.show()
                return idf, empty_odf_print
            else:
                return idf

    else:
        check_dict = {
            "pctile": {"lower": 0, "upper": 0},
            "stdev": {"lower": 0, "upper": 0},
            "IQR": {"lower": 0, "upper": 0},
        }
        side_mapping = {
            "lower": ["lower"],
            "upper": ["upper"],
            "both": ["lower", "upper"],
        }

        for methodology in ["pctile", "stdev", "IQR"]:
            for side in side_mapping[detection_side]:
                if methodology + "_" + side in detection_configs:
                    check_dict[methodology][side] = 1

        methodologies = []
        for key, val in list(check_dict.items()):
            val_list = list(val.values())
            if detection_side == "both":
                if val_list in ([1, 0], [0, 1]):
                    raise TypeError(
                        "Invalid input for detection_configs. If detection_side is 'both', the methodologies used on both sides should be the same"
                    )
                if val_list[0]:
                    methodologies.append(key)
            else:
                if val[detection_side]:
                    methodologies.append(key)
        num_methodologies = len(methodologies)

        if "min_validation" in detection_configs:
            if detection_configs["min_validation"] > num_methodologies:
                raise TypeError(
                    "Invalid input for min_validation of detection_configs. It cannot be larger than the total number of methodologies on any side that detection will be applied over."
                )
        else:
            # if min_validation is not present, num of specified methodologies will be used
            detection_configs["min_validation"] = num_methodologies

        empty_params = [[None, None]] * len(list_of_cols)

        idf_count = idf.count()
        if idf_count > sample_size:
            idf_sample = idf.sample(sample_size / idf_count, False, 11).select(
                list_of_cols
            )
        else:
            idf_sample = idf.select(list_of_cols)

        for i in list_of_cols:
            if get_dtype(idf_sample, i).startswith("decimal"):
                idf_sample = idf_sample.withColumn(i, F.col(i).cast(T.DoubleType()))

        pctiles = [
            detection_configs.get("pctile_lower", 0.05),
            detection_configs.get("pctile_upper", 0.95),
        ]
        pctile_params = idf_sample.approxQuantile(list_of_cols, pctiles, 0.01)
        skewed_cols = []
        for i, p in zip(list_of_cols, pctile_params):
            if p[0] == p[1]:
                skewed_cols.append(i)

        if skewed_cols:
            warnings.warn(
                "Columns excluded from outlier detection due to highly skewed distribution: "
                + ",".join(skewed_cols)
            )
            for i in skewed_cols:
                idx = list_of_cols.index(i)
                list_of_cols.pop(idx)
                pctile_params.pop(idx)

        if "pctile" not in methodologies:
            pctile_params = copy.deepcopy(empty_params)

        if "stdev" in methodologies:
            exprs = [f(F.col(c)) for f in [F.mean, F.stddev] for c in list_of_cols]
            stats = idf_sample.select(exprs).rdd.flatMap(lambda x: x).collect()

            mean, stdev = stats[: len(list_of_cols)], stats[len(list_of_cols) :]
            stdev_lower = pd.Series(mean) - detection_configs.get(
                "stdev_lower", 0.0
            ) * pd.Series(stdev)
            stdev_upper = pd.Series(mean) + detection_configs.get(
                "stdev_upper", 0.0
            ) * pd.Series(stdev)
            stdev_params = list(zip(stdev_lower, stdev_upper))
        else:
            stdev_params = copy.deepcopy(empty_params)

        if "IQR" in methodologies:
            quantiles = idf_sample.approxQuantile(list_of_cols, [0.25, 0.75], 0.01)
            IQR_params = [
                [
                    e[0] - detection_configs.get("IQR_lower", 0.0) * (e[1] - e[0]),
                    e[1] + detection_configs.get("IQR_upper", 0.0) * (e[1] - e[0]),
                ]
                for e in quantiles
            ]
        else:
            IQR_params = copy.deepcopy(empty_params)

        n = detection_configs["min_validation"]
        params = []
        for x, y, z in list(zip(pctile_params, stdev_params, IQR_params)):
            lower = sorted(
                [i for i in [x[0], y[0], z[0]] if i is not None], reverse=True
            )[n - 1]
            upper = sorted([i for i in [x[1], y[1], z[1]] if i is not None])[n - 1]
            if detection_side == "lower":
                param = [lower, None]
            elif detection_side == "upper":
                param = [None, upper]
            else:
                param = [lower, upper]
            params.append(param)

        # Saving model File if required
        if model_path != "NA":
            if detection_side == "lower":
                skewed_param = ["skewed_attribute", None]
            elif detection_side == "upper":
                skewed_param = [None, "skewed_attribute"]
            else:
                skewed_param = ["skewed_attribute", "skewed_attribute"]

            schema = T.StructType(
                [
                    T.StructField("attribute", T.StringType(), True),
                    T.StructField("parameters", T.ArrayType(T.StringType()), True),
                ]
            )
            df_model = spark.createDataFrame(
                zip(
                    list_of_cols + skewed_cols,
                    params + [skewed_param] * len(skewed_cols),
                ),
                schema=schema,
            )
            df_model.coalesce(1).write.parquet(
                model_path + "/outlier_numcols", mode="overwrite"
            )

            if not treatment and not print_impact:
                return idf

    def composite_outlier_pandas(col_param):
        def inner(v):
            v = v.astype(float, errors="raise")
            if detection_side in ("lower", "both"):
                lower_v = ((v - col_param[0]) < 0).replace(True, -1).replace(False, 0)
            if detection_side in ("upper", "both"):
                upper_v = ((v - col_param[1]) > 0).replace(True, 1).replace(False, 0)

            if detection_side == "upper":
                return upper_v
            elif detection_side == "lower":
                return lower_v
            else:
                return lower_v + upper_v

        return inner

    odf = idf

    list_odf = []
    for index, i in enumerate(list_of_cols):
        f_composite_outlier = F.pandas_udf(
            composite_outlier_pandas(params[index]), returnType=T.IntegerType()
        )
        odf = odf.withColumn(i + "_outliered", f_composite_outlier(i))

        if print_impact:
            odf_agg_col = (
                odf.select(i + "_outliered").groupby().pivot(i + "_outliered").count()
            )
            odf_print_col = (
                odf_agg_col.withColumn(
                    "lower_outliers",
                    F.col("-1") if "-1" in odf_agg_col.columns else F.lit(0),
                )
                .withColumn(
                    "upper_outliers",
                    F.col("1") if "1" in odf_agg_col.columns else F.lit(0),
                )
                .withColumn("excluded_due_to_skewness", F.lit(0))
                .withColumn("attribute", F.lit(str(i)))
                .select(
                    "attribute",
                    "lower_outliers",
                    "upper_outliers",
                    "excluded_due_to_skewness",
                )
                .fillna(0)
            )
            list_odf.append(odf_print_col)

        if treatment & (treatment_method in ("value_replacement", "null_replacement")):
            replace_vals = {
                "value_replacement": [params[index][0], params[index][1]],
                "null_replacement": [None, None],
            }
            odf = odf.withColumn(
                i + "_outliered",
                F.when(
                    F.col(i + "_outliered") == 1, replace_vals[treatment_method][1]
                ).otherwise(
                    F.when(
                        F.col(i + "_outliered") == -1,
                        replace_vals[treatment_method][0],
                    ).otherwise(F.col(i))
                ),
            )
            if output_mode == "replace":
                odf = odf.drop(i).withColumnRenamed(i + "_outliered", i)

    if print_impact:

        def unionAll(dfs):
            first, *_ = dfs
            return first.sql_ctx.createDataFrame(
                first.sql_ctx._sc.union([df.rdd for df in dfs]), first.schema
            )

        odf_print = unionAll(list_odf)

        if skewed_cols:
            skewed_cols_print = [(i, 0, 0, 1) for i in skewed_cols]
            skewed_cols_odf_print = spark.createDataFrame(
                skewed_cols_print, schema=odf_print.columns
            )
            odf_print = unionAll([odf_print, skewed_cols_odf_print])

        odf_print.show(len(list_of_cols) + len(skewed_cols), False)

    if treatment & (treatment_method == "row_removal"):
        conditions = [
            (F.col(i + "_outliered") == 0) | (F.col(i + "_outliered").isNull())
            for i in list_of_cols
        ]
        conditions_combined = functools.reduce(lambda a, b: a & b, conditions)
        odf = odf.where(conditions_combined).drop(
            *[i + "_outliered" for i in list_of_cols]
        )

    if treatment:
        if output_mode == "replace":
            odf = odf.select(column_order)
    else:
        odf = idf

    if print_impact:
        return odf, odf_print
    else:
        return odf


def IDness_detection(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    treatment=False,
    treatment_threshold=0.8,
    stats_unique={},
    print_impact=False,
):
    """
    IDness of an attribute is defined as the ratio of number of unique values seen in an attribute by number of
    non-null rows. It varies between 0 to 100% where IDness of 100% means there are as many unique values as number
    of rows (primary key in the input dataset). IDness is computed only for discrete features. This function
    leverages the statistics from Measures of Cardinality function and flag the columns if IDness is above a certain
    threshold. Such columns can be deleted from the modelling analysis if directed for a treatment.

    This function returns two dataframes in tuple format – 1st dataframe is input dataset after removing high IDness
    columns (if treated else the original dataset) and  2nd dataframe is of schema – attribute, unique_values, IDness.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of Discrete (Categorical + Integer) columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all discrete columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    treatment
        Boolean argument – True or False. If True, columns with high IDness (defined by
        treatment_threshold argument) are removed from the input dataframe. (Default value = False)
    treatment_threshold
        Defines acceptable level of IDness (calculated as no. of unique values divided by no. of
        non-null values) for a column and takes value between 0 to 1. Default threshold
        of 0.8 can be interpreted as remove column if its unique values count is more than
        80% of total rows (after excluding null values).
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    print_impact
        True, False
        This argument is to print out the statistics and the impact of treatment (if applicable).(Default value = False)

    Returns
    -------
    odf : DataFrame
        Dataframe after column removal if treated, else original input dataframe.
    odf_print : DataFrame
        schema [attribute, unique_values, IDness, flagged/treated].
        unique_values is no. of distinct values in a column,
        IDness is unique_values divided by no. of non-null values.
        A column is flagged 1 if IDness is above the threshold, else 0.
    """

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    for i in idf.select(list_of_cols).dtypes:
        if i[1] not in ("string", "int", "bigint", "long"):
            list_of_cols.remove(i[0])

    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if len(list_of_cols) == 0:
        warnings.warn("No IDness Check - No discrete column(s) to analyze")
        odf = idf
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("unique_values", T.StringType(), True),
                T.StructField("IDness", T.StringType(), True),
                T.StructField("flagged", T.StringType(), True),
            ]
        )
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print
    treatment_threshold = float(treatment_threshold)
    if (treatment_threshold < 0) | (treatment_threshold > 1):
        raise TypeError("Invalid input for Treatment Threshold Value")
    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")

    if stats_unique == {}:
        odf_print = measures_of_cardinality(spark, idf, list_of_cols)
    else:
        odf_print = read_dataset(spark, **stats_unique).where(
            F.col("attribute").isin(list_of_cols)
        )

    odf_print = odf_print.withColumn(
        "flagged", F.when(F.col("IDness") >= treatment_threshold, 1).otherwise(0)
    )

    if treatment:
        remove_cols = (
            odf_print.where(F.col("flagged") == 1)
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        odf = idf.drop(*remove_cols)
        odf_print = odf_print.withColumnRenamed("flagged", "treated")
    else:
        odf = idf

    if print_impact:
        odf_print.show(len(list_of_cols))
        if treatment:
            print("Removed Columns: ", remove_cols)

    return odf, odf_print


def biasedness_detection(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    treatment=False,
    treatment_threshold=0.8,
    stats_mode={},
    print_impact=False,
):
    """
    This function flags column if they are biased or skewed towards one specific value and leverages
    mode_pct computation from Measures of Central Tendency i.e. number of rows with mode value (most frequently seen
    value) divided by number of non-null values. It varies between 0 to 100% where biasedness of 100% means there is
    only a single value (other than null). The function flags a column if its biasedness is above a certain
    threshold. Such columns can be deleted from the modelling analysis, if required.

    This function returns two dataframes in tuple format – 1st dataframe is input dataset after removing high biased
    columns (the original dataset if no treatment) and  2nd dataframe is of schema – attribute, mode, mode_pct.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of Discrete (Categorical + Integer) columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all discrete columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    treatment
        Boolean argument – True or False. If True, columns with high biasedness (defined by
        treatment_threshold argument) are removed from the input dataframe. (Default value = False)
    treatment_threshold
        Defines acceptable level of biasedness (frequency of most-frequently seen value)for
        a column and takes value between 0 to 1. Default threshold of 0.8 can be interpreted as
        remove column if the number of rows with most-frequently seen value is more than 80%
        of total rows (after excluding null values).
    stats_mode
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
        mode_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    print_impact
        True, False
        This argument is to print out the statistics and the impact of treatment (if applicable).(Default value = False)

    Returns
    -------
    odf : DataFrame
        Dataframe after column removal if treated, else original input dataframe.
    odf_print : DataFrame
        schema [attribute, mode, mode_rows, mode_pct, flagged/treated].
        mode is the most frequently seen value,
        mode_rows is number of rows with mode value, and
        mode_pct is number of rows with mode value divided by non-null values.
        A column is flagged 1 if mode_pct is above the threshold else 0.
    """

    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    for i in idf.select(list_of_cols).dtypes:
        if i[1] not in ("string", "int", "bigint", "long"):
            list_of_cols.remove(i[0])

    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if len(list_of_cols) == 0:
        warnings.warn("No biasedness Check - No discrete column(s) to analyze")
        odf = idf
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("mode", T.StringType(), True),
                T.StructField("mode_rows", T.StringType(), True),
                T.StructField("mode_pct", T.StringType(), True),
                T.StructField("flagged", T.StringType(), True),
            ]
        )
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print

    if (treatment_threshold < 0) | (treatment_threshold > 1):
        raise TypeError("Invalid input for Treatment Threshold Value")
    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")

    if stats_mode == {}:
        odf_print = (
            transpose_dataframe(idf.select(list_of_cols).summary("count"), "summary")
            .withColumnRenamed("key", "attribute")
            .join(mode_computation(spark, idf, list_of_cols), "attribute", "full_outer")
            .withColumn(
                "mode_pct",
                F.round(F.col("mode_rows") / F.col("count").cast(T.DoubleType()), 4),
            )
            .select("attribute", "mode", "mode_rows", "mode_pct")
        )
    else:
        odf_print = (
            read_dataset(spark, **stats_mode)
            .select("attribute", "mode", "mode_rows", "mode_pct")
            .where(F.col("attribute").isin(list_of_cols))
        )

    odf_print = odf_print.withColumn(
        "flagged",
        F.when(
            (F.col("mode_pct") >= treatment_threshold) | (F.col("mode_pct").isNull()), 1
        ).otherwise(0),
    )

    if treatment:
        remove_cols = (
            odf_print.where(
                (F.col("mode_pct") >= treatment_threshold)
                | (F.col("mode_pct").isNull())
            )
            .select("attribute")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        odf = idf.drop(*remove_cols)
        odf_print = odf_print.withColumnRenamed("flagged", "treated")

    else:
        odf = idf

    if print_impact:
        odf_print.show(len(list_of_cols))
        if treatment:
            print("Removed Columns: ", remove_cols)

    return odf, odf_print


def invalidEntries_detection(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    detection_type="auto",
    invalid_entries=[],
    valid_entries=[],
    partial_match=False,
    treatment=False,
    treatment_method="null_replacement",
    treatment_configs={},
    stats_missing={},
    stats_unique={},
    stats_mode={},
    output_mode="replace",
    print_impact=False,
):
    """
    This function checks for certain suspicious patterns in attributes’ values. Patterns that are considered for this quality check:

    - Missing Values: The function checks for all column values which directly or indirectly indicate the missing
    value in an attribute such as 'nan', 'null', 'na', 'inf', 'n/a', 'not defined' etc. The function also check for special characters.

    - Repetitive Characters: Certain attributes’ values with repetitive characters may be default value or system
    error, rather than being a legit value etc xx, zzzzz, 99999 etc. Such values are flagged for the user to take an
    appropriate action. There may be certain false positive which are legit values.

    - Consecutive Characters: Similar to repetitive characters, consecutive characters (at least 3 characters long) such as abc, 1234 etc
      may not be legit values, and hence flagged. There may be certain false positive which are legit values.

    This function returns two dataframes in tuple format – 1st dataframe is input dataset after treating the invalid values
    (or the original dataset if no treatment) and  2nd dataframe is of schema – attribute,
    invalid_entries, invalid_count, invalid_pct. All potential invalid values (separated by delimiter pipe “|”) are
    shown under invalid_entries column. Total number of rows impacted by these entries for each attribute is shown
    under invalid_count. invalid_pct is invalid_count divided by number of rows.


    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of Discrete (Categorical + Integer) columns to analyse e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all discrete columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in drop_cols argument
        is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of list_of_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    detection_type
        "auto","manual","both" (Default value = "auto")
    invalid_entries
        List of values or regex patterns to be classified as invalid.
        Valid only for "auto" or "both" detection type. (Default value = [])
    valid_entries
        List of values or regex patterns such that a value will be classified as invalid if it
        does not match any value or regex pattern in it. Valid only for "auto" or "both" detection type. (Default value = [])
    partial_match
        Boolean argument – True or False. If True, values with substring same as invalid_entries is declared invalid. (Default value = False)
    treatment
        Boolean argument – True or False. If True, outliers are treated as per treatment_method argument. (Default value = False)
    treatment_method
        "MMM", "null_replacement", "column_removal" (more methods to be added soon).
        MMM (Mean Median Mode) replaces invalid value by the measure of central tendency (mode for
        categorical features and mean or median for numerical features).
        null_replacement removes all values with any invalid values as null.
        column_removal remove a column if % of rows with invalid value is above a threshold (defined
        by key "treatment_threshold" under treatment_configs argument). (Default value = "null_replacement")
    treatment_configs
        Takes input in dictionary format.
        For column_removal treatment, key ‘treatment_threshold’ is provided with a value between 0 to 1.
        For value replacement, by MMM, arguments corresponding to imputation_MMM function (transformer module) are provided,
        where each key is an argument from imputation_MMM function.
        For null_replacement, this argument can be skipped. (Default value = {})
    stats_missing
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
        missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    stats_unique
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
        uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    stats_mode
        Takes arguments for read_dataset (data_ingest module) function in a dictionary format
        to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
        mode_computation (data_analyzer.stats_generator module) has been computed & saved before. (Default value = {})
    output_mode
        "replace", "append".
        “replace” option replaces original columns with treated column. “append” option append treated
        column to the input dataset with a postfix "_invalid" e.g. column X is appended as X_invalid. (Default value = "replace")
    print_impact
        True, False
        This argument is to print out the statistics.(Default value = False)

    Returns
    -------
    odf : DataFrame
        Dataframe after treatment if applicable, else original input dataframe.
    odf_print : DataFrame
        schema [attribute, invalid_entries, invalid_count, invalid_pct].
        invalid_entries are all potential invalid values (separated by delimiter pipe “|”),
        invalid_count is no. of rows which are impacted by invalid entries, and
        invalid_pct is invalid_count divided by no of rows.

    """

    if list_of_cols == "all":
        list_of_cols = []
        for i in idf.dtypes:
            if i[1] in ("string", "int", "bigint", "long"):
                list_of_cols.append(i[0])
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if len(list_of_cols) == 0:
        warnings.warn("No Invalid Entries Check - No discrete column(s) to analyze")
        odf = idf
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("invalid_entries", T.StringType(), True),
                T.StructField("invalid_count", T.StringType(), True),
                T.StructField("invalid_pct", T.StringType(), True),
            ]
        )
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print

    if output_mode not in ("replace", "append"):
        raise TypeError("Invalid input for output_mode")
    if str(treatment).lower() == "true":
        treatment = True
    elif str(treatment).lower() == "false":
        treatment = False
    else:
        raise TypeError("Non-Boolean input for treatment")

    if treatment_method not in ("MMM", "null_replacement", "column_removal"):
        raise TypeError("Invalid input for method_type")

    treatment_threshold = treatment_configs.pop("treatment_threshold", None)
    if treatment_threshold:
        treatment_threshold = float(treatment_threshold)
    else:
        if treatment_method == "column_removal":
            raise TypeError("Invalid input for column removal threshold")

    null_vocab = [
        "",
        " ",
        "nan",
        "null",
        "na",
        "inf",
        "n/a",
        "not defined",
        "none",
        "undefined",
        "blank",
        "unknown",
    ]
    special_chars_vocab = [
        "&",
        "$",
        ";",
        ":",
        ".",
        ",",
        "*",
        "#",
        "@",
        "_",
        "?",
        "%",
        "!",
        "^",
        "(",
        ")",
        "-",
        "/",
        "'",
    ]

    def detect(*v):
        output = []
        for idx, e in enumerate(v):
            if e is None:
                output.append(None)
                continue
            if detection_type in ("auto", "both"):
                e = str(e).lower().strip()
                # Null & Special Chars Search
                if e in (null_vocab + special_chars_vocab):
                    output.append(1)
                    continue
                # Consecutive Identical Chars Search
                regex = "\\b([a-zA-Z0-9])\\1\\1+\\b"
                p = re.compile(regex)
                if re.search(p, e):
                    output.append(1)
                    continue
                # Ordered Chars Search
                l = len(e)
                check = 0
                if l >= 3:
                    for i in range(1, l):
                        if ord(e[i]) - ord(e[i - 1]) != 1:
                            check = 1
                            break
                    if check == 0:
                        output.append(1)
                        continue

            check = 0
            if detection_type in ("manual", "both"):
                e = str(e).lower().strip()
                for regex in invalid_entries:
                    p = re.compile(regex)
                    if partial_match:
                        if re.search(p, e):
                            check = 1
                            output.append(1)
                            break
                    else:
                        if p.fullmatch(e):
                            check = 1
                            output.append(1)
                            break

                match_valid_entries = []
                for regex in valid_entries:
                    p = re.compile(regex)
                    if partial_match:
                        if re.search(p, e):
                            match_valid_entries.append(1)
                        else:
                            match_valid_entries.append(0)
                    else:
                        if p.fullmatch(e):
                            match_valid_entries.append(1)
                        else:
                            match_valid_entries.append(0)

                if (len(match_valid_entries) > 0) & (sum(match_valid_entries) == 0):
                    check = 1
                    output.append(1)

            if check == 0:
                output.append(0)

        return output

    f_detect = F.udf(detect, T.ArrayType(T.LongType()))

    odf = idf.withColumn("invalid", f_detect(*list_of_cols))

    odf.persist()
    output_print = []

    for index, i in enumerate(list_of_cols):
        tmp = odf.withColumn(i + "_invalid", F.col("invalid")[index])
        invalid = (
            tmp.where(F.col(i + "_invalid") == 1)
            .select(i)
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        invalid = [str(x) for x in invalid]
        invalid_count = tmp.where(F.col(i + "_invalid") == 1).count()
        output_print.append(
            [i, "|".join(invalid), invalid_count, round(invalid_count / idf.count(), 4)]
        )

    odf_print = spark.createDataFrame(
        output_print,
        schema=["attribute", "invalid_entries", "invalid_count", "invalid_pct"],
    )

    if treatment:
        if treatment_threshold:
            threshold_cols = (
                odf_print.where(F.col("attribute").isin(list_of_cols))
                .where(F.col("invalid_pct") > treatment_threshold)
                .select("attribute")
                .rdd.flatMap(lambda x: x)
                .collect()
            )
        if treatment_method in ("null_replacement", "MMM"):
            for index, i in enumerate(list_of_cols):
                if treatment_threshold:
                    if i not in threshold_cols:
                        odf = odf.drop(i + "_invalid")
                        continue
                odf = odf.withColumn(
                    i + "_invalid",
                    F.when(F.col("invalid")[index] == 1, None).otherwise(F.col(i)),
                )
                if output_mode == "replace":
                    odf = odf.drop(i).withColumnRenamed(i + "_invalid", i)
                else:
                    if (
                        odf_print.where(F.col("attribute") == i)
                        .select("invalid_pct")
                        .collect()[0][0]
                        == 0.0
                    ):
                        odf = odf.drop(i + "_invalid")
            odf = odf.drop("invalid")

        if treatment_method == "column_removal":
            odf = idf.drop(*threshold_cols)
            if print_impact:
                print("Removed Columns: ", threshold_cols)

        if treatment_method == "MMM":
            if stats_unique == {} or output_mode == "append":
                remove_cols = (
                    uniqueCount_computation(spark, odf, list_of_cols)
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
            if treatment_threshold:
                list_of_cols = [e for e in threshold_cols if e not in remove_cols]
            if output_mode == "append":
                if len(list_of_cols) > 0:
                    list_of_cols = [e + "_invalid" for e in list_of_cols]
            odf = imputation_MMM(
                spark,
                odf,
                list_of_cols,
                **treatment_configs,
                stats_missing=stats_missing,
                stats_mode=stats_mode,
                print_impact=print_impact
            )
    else:
        odf = idf

    if print_impact:
        odf_print.show(len(list_of_cols))

    return odf, odf_print
