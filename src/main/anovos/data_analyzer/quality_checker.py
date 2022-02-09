# coding=utf-8
import re
import warnings

from pyspark.sql import functions as F
from pyspark.sql import types as T

from anovos.data_analyzer.stats_generator import (
    uniqueCount_computation,
    missingCount_computation,
    mode_computation,
    measures_of_cardinality,
)
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.transformers import imputation_MMM
from anovos.shared.utils import (
    attributeType_segregation,
    transpose_dataframe,
    get_dtype,
)


def duplicate_detection(
    spark, idf, list_of_cols="all", drop_cols=[], treatment=False, print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, duplicate rows are removed from the input dataframe.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is de-duplicated dataframe if treated, else original input dataframe.
              Metric Dataframe is of schema [metric, value] and contains metrics - number of rows, number of unique rows,
              number of duplicate rows and percentage of duplicate rows in total.
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

    odf_tmp = idf.drop_duplicates(subset=list_of_cols)
    odf = odf_tmp if treatment else idf

    odf_print = spark.createDataFrame(
        [
            ["rows_count", float(idf.count())],
            ["unique_rows_count", float(odf_tmp.count())],
            ["duplicate_rows", float(idf.count() - odf_tmp.count())],
            ["duplicate_pct", round((idf.count() - odf_tmp.count()) / idf.count(), 4)],
        ],
        schema=["metric", "value"],
    )
    if print_impact:
        print("No. of Rows: " + str(idf.count()))
        print("No. of UNIQUE Rows: " + str(odf_tmp.count()))
        print("No. of Duplicate Rows: " + str(idf.count() - odf_tmp.count()))
        print(
            "Percentage of Duplicate Rows: "
            + str(round((idf.count() - odf_tmp.count()) / idf.count(), 4))
        )

    return odf, odf_print


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
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, rows with high no. of null columns (defined by
                      treatment_threshold argument) are removed from the input dataframe.
    :param treatment_threshold: Defines % of columns allowed to be Null per row and takes value between 0 to 1.
                                If % of null columns is above the threshold for a row, it is removed from the dataframe.
                                There is no row removal if the threshold is 1.0. And if the threshold is 0, all rows with
                                null value are removed.
    :param print_impact: True, False.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after row removal if treated, else original input dataframe.
              Metric Dataframe is of schema [null_cols_count, row_count, row_pct, flagged/treated]. null_cols_count is defined as
              no. of missing columns in a row. row_count is no. of rows with null_cols_count missing columns.
              row_pct is row_count divided by number of rows. flagged/treated is 1 if null_cols_count is more than
              (threshold  X Number of Columns), else 0.
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
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all (non-array) columns for analysis.
                         "missing" (default) can be passed to include only those columns with missing values.
                         One of the usecases where "all" may be preferable over "missing" is when the user wants to save
                         the imputation model for the future use e.g. a column may not have missing value in the training
                         dataset but missing values may possibly appear in the prediction dataset.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, missing values are treated as per treatment_method argument.
    :param treatment_method: "MMM", "row_removal", "column_removal" (more methods to be added soon).
                             MMM (Mean Median Mode) replaces null value by the measure of central tendency (mode for
                             categorical features and mean or median for numerical features).
                             row_removal removes all rows with any missing value.
                             column_removal remove a column if % of rows with missing value is above a threshold (defined
                             by key "treatment_threshold" under treatment_configs argument).
    :param treatment_configs: Takes input in dictionary format.
                              For column_removal treatment, key ‘treatment_threshold’ is provided with a value between 0 to 1.
                              For MMM, arguments corresponding to imputation_MMM function (transformer module) are provided,
                              where each key is an argument from imputation_MMM function.
                              For row_removal, this argument can be skipped.
    :param stats_missing: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                          to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
                          missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param stats_mode: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                       to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
                       mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param print_impact: True,False.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the imputed dataframe if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, missing_count, missing_pct]. missing_count is number of rows
              with null values for an attribute and missing_pct is missing_count divided by number of rows.
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

    if treatment_method not in ("MMM", "row_removal", "column_removal"):
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
    else:
        odf = idf

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
    treatment=False,
    treatment_method="value_replacement",
    pre_existing_model=False,
    model_path="NA",
    output_mode="replace",
    stats_unique={},
    print_impact=False,
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param detection_side: "upper", "lower", "both".
                           "lower" detects outliers in the lower spectrum of the column range, whereas "upper" detects
                           in the upper spectrum. "Both" detects in both upper and lower end of the spectrum.
    :param detection_configs: Takes input in dictionary format with keys representing upper & lower parameter for
                              three outlier detection methodologies.
                              a) Percentile Method: In this methodology, a value higher than a certain (default 0.95)
                              percentile value is considered as an outlier. Similarly, a value lower than a certain
                              (default 0.05) percentile value is considered as an outlier.
                              b) Standard Deviation Method: In this methodology, if a value is certain number of
                              standard deviations (default 3.0) away from the mean, then it is identified as an outlier.
                              c) Interquartile Range (IQR) Method: A value which is below Q1 – k * IQR or
                              above Q3 + k * IQR (default k is 1.5) are identified as outliers, where Q1 is first quartile/
                              25th percentile, Q3 is third quartile/75th percentile and IQR is difference between
                              third quartile & first quartile.
                              If an attribute value is less (more) than its derived lower (upper) bound value,
                              it is considered as outlier by a methodology. A attribute value is considered as outlier
                              if it is declared as outlier by atleast 'min_validation' methodologies (default 2).
    :param treatment: Boolean argument – True or False. If True, outliers are treated as per treatment_method argument.
    :param treatment_method: "null_replacement", "row_removal", "value_replacement".
                             In "null_replacement", outlier values are replaced by null so that it can be imputed by a
                             reliable imputation methodology. In "value_replacement", outlier values are replaced by
                             maximum or minimum permissible value by above methodologies. Lastly in "row_removal", rows
                             are removed if it is found with any outlier.
    :param pre_existing_model: Boolean argument – True or False. True if the model with upper/lower permissible values
                               for each attribute exists already to be used, False otherwise.
    :param model_path: If pre_existing_model is True, this argument is path for the pre-saved model.
                       If pre_existing_model is False, this field can be used for saving the model.
                       Default "NA" means there is neither pre-existing model nor there is a need to save one.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with treated column. “append” option append treated
                        column to the input dataset with a postfix "_outliered" e.g. column X is appended as X_outliered.
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param print_impact: True, False.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the imputed dataframe if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, lower_outliers, upper_outliers]. lower_outliers is no. of outliers
              found in the lower spectrum of the attribute range and upper_outliers is outlier count in the upper spectrum.
    """

    num_cols = attributeType_segregation(idf)[0]
    if len(num_cols) == 0:
        warnings.warn("No Outlier Check - No numerical column(s) to analyse")
        odf = idf
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("lower_outliers", T.StringType(), True),
                T.StructField("upper_outliers", T.StringType(), True),
            ]
        )
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print
    if list_of_cols == "all":
        list_of_cols = num_cols
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

    recast_cols = []
    recast_type = []
    for i in list_of_cols:
        if get_dtype(idf, i).startswith("decimal"):
            idf = idf.withColumn(i, F.col(i).cast(T.DoubleType()))
            recast_cols.append(i)
            recast_type.append(get_dtype(idf, i))

    if pre_existing_model:
        df_model = spark.read.parquet(model_path + "/outlier_numcols")
        params = []
        for i in list_of_cols:
            mapped_value = (
                df_model.where(F.col("attribute") == i)
                .select("parameters")
                .rdd.flatMap(lambda x: x)
                .collect()[0]
            )
            params.append(mapped_value)

        pctile_params = idf.approxQuantile(
            list_of_cols,
            [
                detection_configs.get("pctile_lower", 0.05),
                detection_configs.get("pctile_upper", 0.95),
            ],
            0.01,
        )
        skewed_cols = []
        for i, p in zip(list_of_cols, pctile_params):
            if p[0] == p[1]:
                skewed_cols.append(i)
    else:
        detection_configs["pctile_lower"] = detection_configs["pctile_lower"] or 0.0
        detection_configs["pctile_upper"] = detection_configs["pctile_upper"] or 1.0
        pctile_params = idf.approxQuantile(
            list_of_cols,
            [detection_configs["pctile_lower"], detection_configs["pctile_upper"]],
            0.01,
        )
        skewed_cols = []
        for i, p in zip(list_of_cols, pctile_params):
            if p[0] == p[1]:
                skewed_cols.append(i)

        detection_configs["stdev_lower"] = (
            detection_configs["stdev_lower"] or detection_configs["stdev_upper"]
        )
        detection_configs["stdev_upper"] = (
            detection_configs["stdev_upper"] or detection_configs["stdev_lower"]
        )
        stdev_params = []
        for i in list_of_cols:
            mean, stdev = idf.select(F.mean(i), F.stddev(i)).first()
            stdev_params.append(
                [
                    mean - detection_configs["stdev_lower"] * stdev,
                    mean + detection_configs["stdev_upper"] * stdev,
                ]
            )

        detection_configs["IQR_lower"] = (
            detection_configs["IQR_lower"] or detection_configs["IQR_upper"]
        )
        detection_configs["IQR_upper"] = (
            detection_configs["IQR_upper"] or detection_configs["IQR_lower"]
        )
        quantiles = idf.approxQuantile(list_of_cols, [0.25, 0.75], 0.01)
        IQR_params = [
            [
                e[0] - detection_configs["IQR_lower"] * (e[1] - e[0]),
                e[1] + detection_configs["IQR_upper"] * (e[1] - e[0]),
            ]
            for e in quantiles
        ]
        n = detection_configs["min_validation"]
        params = [
            [
                sorted([x[0], y[0], z[0]], reverse=True)[n - 1],
                sorted([x[1], y[1], z[1]])[n - 1],
            ]
            for x, y, z in list(zip(pctile_params, stdev_params, IQR_params))
        ]

        # Saving model File if required
        if model_path != "NA":
            df_model = spark.createDataFrame(
                zip(list_of_cols, params), schema=["attribute", "parameters"]
            )
            df_model.coalesce(1).write.parquet(
                model_path + "/outlier_numcols", mode="overwrite"
            )

    for i, j in zip(recast_cols, recast_type):
        idf = idf.withColumn(i, F.col(i).cast(j))

    def composite_outlier(*v):
        output = []
        for idx, e in enumerate(v):
            if e is None:
                output.append(None)
                continue
            if detection_side in ("upper", "both"):
                if e > params[idx][1]:
                    output.append(1)
                    continue
            if detection_side in ("lower", "both"):
                if e < params[idx][0]:
                    output.append(-1)
                    continue
            output.append(0)
        return output

    f_composite_outlier = F.udf(composite_outlier, T.ArrayType(T.IntegerType()))

    odf = idf.withColumn("outliered", f_composite_outlier(*list_of_cols))
    odf.persist()
    output_print = []
    for index, i in enumerate(list_of_cols):
        odf = odf.withColumn(i + "_outliered", F.col("outliered")[index])
        output_print.append(
            [
                i,
                odf.where(F.col(i + "_outliered") == -1).count(),
                odf.where(F.col(i + "_outliered") == 1).count(),
            ]
        )

        if treatment & (treatment_method in ("value_replacement", "null_replacement")):
            if skewed_cols:
                warnings.warn(
                    "Columns dropped from outlier treatment due to highly skewed distribution: "
                    + (",").join(skewed_cols)
                )
            if i not in skewed_cols:
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
            else:
                odf = odf.drop(i + "_outliered")

    odf = odf.drop("outliered")

    if treatment & (treatment_method == "row_removal"):
        if skewed_cols:
            warnings.warn(
                "Columns dropped from outlier treatment due to highly skewed distribution: "
                + (",").join(skewed_cols)
            )
        for index, i in enumerate(list_of_cols):
            if i not in skewed_cols:
                odf = odf.where(
                    (F.col(i + "_outliered") == 0) | (F.col(i + "_outliered").isNull())
                ).drop(i + "_outliered")
            else:
                odf = odf.drop(i + "_outliered")

    if not treatment:
        odf = idf

    odf_print = spark.createDataFrame(
        output_print, schema=["attribute", "lower_outliers", "upper_outliers"]
    )
    if print_impact:
        odf_print.show(len(list_of_cols))

    return odf, odf_print


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
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all categorical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, columns with high IDness (defined by
                      treatment_threshold argument) are removed from the input dataframe.
    :param treatment_threshold: Defines acceptable level of IDness (calculated as no. of unique values divided by no. of
                                non-null values) for a column and takes value between 0 to 1. Default threshold
                                of 0.8 can be interpreted as remove column if its unique values count is more than
                                80% of total rows (after excluding null values).
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param print_impact: True,False.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after column removal if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, unique_values, IDness, flagged/treated]. unique_values is no. of distinct
              values in a column, IDness is unique_values divided by no. of non-null values. A column is flagged 1
              if IDness is above the threshold, else 0.
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
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, columns with high biasedness (defined by
                      treatment_threshold argument) are removed from the input dataframe.
    :param treatment_threshold: Defines acceptable level of biasedness (frequency of most-frequently seen value)for
                                a column and takes value between 0 to 1. Default threshold of 0.8 can be interpreted as
                                remove column if the number of rows with most-frequently seen value is more than 80%
                                of total rows (after excluding null values).
    :param stats_mode: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                       to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
                       mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after column removal if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, mode, mode_rows, mode_pct, flagged/treated]. mode is the most frequently seen value,
              mode_rows is number of rows with mode value and mode_pct is number of rows with mode value divided by non-null values.
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
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param detection_type: "auto","manual","both"
    :param invalid_entries: List of values or regex patterns to be classified as invalid.
                            Valid only for "auto" or "both" detection type.
    :param partial_match: Boolean argument – True or False. If True, values with substring same as invalid_entries is declared invalid.
    :param treatment: Boolean argument – True or False. If True, invalid values are replaced by Null.
    :param treatment_method: "MMM", "null_replacement", "column_removal" (more methods to be added soon).
                             MMM (Mean Median Mode) replaces invalid value by the measure of central tendency (mode for
                             categorical features and mean or median for numerical features).
                             null_replacement removes all values with any invalid values as null.
                             column_removal remove a column if % of rows with invalid value is above a threshold (defined
                             by key "treatment_threshold" under treatment_configs argument).
     :param treatment_configs: Takes input in dictionary format.
                              For column_removal treatment, key ‘treatment_threshold’ is provided with a value between 0 to 1.
                              For value replacement, by MMM, arguments corresponding to imputation_MMM function (transformer module) are provided,
                              where each key is an argument from imputation_MMM function.
                              For null_replacement, this argument can be skipped.
    :param stats_missing
    :param stats_unique
    :param stats_mode
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with treated column. “append” option append treated
                        column to the input dataset with a postfix "_invalid" e.g. column X is appended as X_invalid.
    :param print_impact: True, False.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after treatment if applicable, else original input dataframe.
              Metric Dataframe is of schema [attribute, invalid_entries, invalid_count, invalid_pct].
              invalid_entries are all potential invalid values (separated by delimiter pipe “|”), invalid_count is no.
              of rows which are impacted by invalid entries and invalid_pct is invalid_count divided by no of rows.
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
