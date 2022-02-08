# coding=utf-8
import warnings

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.sql import functions as F
from pyspark.sql import types as T

from anovos.shared.utils import transpose_dataframe, attributeType_segregation


def global_summary(spark, idf, list_of_cols="all", drop_cols=[], print_impact=True):
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
    :param print_impact: True, False
    :return: Dataframe [metric, value]
    """
    if list_of_cols == "all":
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")

    row_count = idf.count()
    col_count = len(list_of_cols)
    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))
    numcol_count = len(num_cols)
    catcol_count = len(cat_cols)
    othercol_count = len(other_cols)
    if print_impact:
        print("No. of Rows: %s" % "{0:,}".format(row_count))
        print("No. of Columns: %s" % "{0:,}".format(col_count))
        print("Numerical Columns: %s" % "{0:,}".format(numcol_count))
        if numcol_count > 0:
            print(num_cols)
        print("Categorical Columns: %s" % "{0:,}".format(catcol_count))
        if catcol_count > 0:
            print(cat_cols)
        if othercol_count > 0:
            print("Other Columns: %s" % "{0:,}".format(othercol_count))
            print(other_cols)

    odf = spark.createDataFrame(
        [
            ["rows_count", str(row_count)],
            ["columns_count", str(col_count)],
            ["numcols_count", str(numcol_count)],
            ["numcols_name", ", ".join(num_cols)],
            ["catcols_count", str(catcol_count)],
            ["catcols_name", ", ".join(cat_cols)],
            ["othercols_count", str(othercol_count)],
            ["othercols_name", ", ".join(other_cols)],
        ],
        schema=["metric", "value"],
    )
    return odf


def missingCount_computation(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
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
    :return: Dataframe [attribute, missing_count, missing_pct]
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

    idf_stats = idf.select(list_of_cols).summary("count")
    odf = (
        transpose_dataframe(idf_stats, "summary")
        .withColumn(
            "missing_count", F.lit(idf.count()) - F.col("count").cast(T.LongType())
        )
        .withColumn(
            "missing_pct", F.round(F.col("missing_count") / F.lit(idf.count()), 4)
        )
        .select(F.col("key").alias("attribute"), "missing_count", "missing_pct")
    )
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def nonzeroCount_computation(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe [attribute, nonzero_count, nonzero_pct]
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
            "No Non-Zero Count Computation - No numerical column(s) to analyze"
        )
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("nonzero_count", T.StringType(), True),
                T.StructField("nonzero_pct", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    tmp = idf.select(list_of_cols).fillna(0).rdd.map(lambda row: Vectors.dense(row))
    nonzero_count = Statistics.colStats(tmp).numNonzeros()
    odf = spark.createDataFrame(
        zip(list_of_cols, [int(i) for i in nonzero_count]),
        schema=("attribute", "nonzero_count"),
    ).withColumn("nonzero_pct", F.round(F.col("nonzero_count") / F.lit(idf.count()), 4))
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def measures_of_counts(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
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
    :return: Dataframe [attribute, fill_count, fill_pct, missing_count, missing_pct, nonzero_count, nonzero_pct]
    """
    if list_of_cols == "all":
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError("Invalid input for Column(s)")

    odf = (
        transpose_dataframe(idf.select(list_of_cols).summary("count"), "summary")
        .select(
            F.col("key").alias("attribute"),
            F.col("count").cast(T.LongType()).alias("fill_count"),
        )
        .withColumn("fill_pct", F.round(F.col("fill_count") / F.lit(idf.count()), 4))
        .withColumn(
            "missing_count", F.lit(idf.count()) - F.col("fill_count").cast(T.LongType())
        )
        .withColumn("missing_pct", F.round(1 - F.col("fill_pct"), 4))
        .join(nonzeroCount_computation(spark, idf, num_cols), "attribute", "full_outer")
    )

    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def mode_computation(spark, idf, list_of_cols="all", drop_cols=[], print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe [attribute, mode, mode_rows]
             In case there is tie between multiple values, one value is randomly picked as mode.
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
        warnings.warn("No Mode Computation - No discrete column(s) to analyze")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("mode", T.StringType(), True),
                T.StructField("mode_rows", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    mode = [
        list(
            idf.select(i)
            .dropna()
            .groupby(i)
            .count()
            .orderBy("count", ascending=False)
            .first()
            or [None, None]
        )
        for i in list_of_cols
    ]
    mode = [(str(i), str(j)) for i, j in mode]

    odf = spark.createDataFrame(
        zip(list_of_cols, mode), schema=("attribute", "metric")
    ).select(
        "attribute",
        (F.col("metric")["_1"]).alias("mode"),
        (F.col("metric")["_2"]).cast("long").alias("mode_rows"),
    )

    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def measures_of_centralTendency(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
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
    :return: Dataframe [attribute, mean, median, mode, mode_rows, mode_pct]
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

    odf = (
        transpose_dataframe(
            idf.select(list_of_cols).summary("mean", "50%", "count"), "summary"
        )
        .withColumn("mean", F.round(F.col("mean").cast(T.DoubleType()), 4))
        .withColumn("median", F.round(F.col("50%").cast(T.DoubleType()), 4))
        .withColumnRenamed("key", "attribute")
        .join(mode_computation(spark, idf, list_of_cols), "attribute", "full_outer")
        .withColumn(
            "mode_pct",
            F.round(F.col("mode_rows") / F.col("count").cast(T.DoubleType()), 4),
        )
        .select("attribute", "mean", "median", "mode", "mode_rows", "mode_pct")
    )

    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def uniqueCount_computation(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param print_impact: True, False
    :return: Dataframe [attribute, unique_values]
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
        warnings.warn("No Unique Count Computation - No discrete column(s) to analyze")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("unique_values", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    uniquevalue_count = idf.agg(
        *(F.countDistinct(F.col(i)).alias(i) for i in list_of_cols)
    )
    odf = spark.createDataFrame(
        zip(list_of_cols, uniquevalue_count.rdd.map(list).collect()[0]),
        schema=("attribute", "unique_values"),
    )
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def measures_of_cardinality(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe [attribute, unique_values, IDness]
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
        warnings.warn("No Cardinality Computation - No discrete column(s) to analyze")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("unique_values", T.StringType(), True),
                T.StructField("IDness", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    odf = (
        uniqueCount_computation(spark, idf, list_of_cols)
        .join(
            missingCount_computation(spark, idf, list_of_cols),
            "attribute",
            "full_outer",
        )
        .withColumn(
            "IDness",
            F.round(
                F.col("unique_values") / (F.lit(idf.count()) - F.col("missing_count")),
                4,
            ),
        )
        .select("attribute", "unique_values", "IDness")
    )
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def measures_of_dispersion(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe [attribute, stddev, variance, cov, IQR, range]
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
        warnings.warn("No Dispersion Computation - No numerical column(s) to analyze")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("stddev", T.StringType(), True),
                T.StructField("variance", T.StringType(), True),
                T.StructField("cov", T.StringType(), True),
                T.StructField("IQR", T.StringType(), True),
                T.StructField("range", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    odf = (
        transpose_dataframe(
            idf.select(list_of_cols).summary(
                "stddev", "min", "max", "mean", "25%", "75%"
            ),
            "summary",
        )
        .withColumn("stddev", F.round(F.col("stddev").cast(T.DoubleType()), 4))
        .withColumn("variance", F.round(F.col("stddev") * F.col("stddev"), 4))
        .withColumn("range", F.round(F.col("max") - F.col("min"), 4))
        .withColumn("cov", F.round(F.col("stddev") / F.col("mean"), 4))
        .withColumn("IQR", F.round(F.col("75%") - F.col("25%"), 4))
        .select(
            F.col("key").alias("attribute"), "stddev", "variance", "cov", "IQR", "range"
        )
    )
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def measures_of_percentiles(
    spark, idf, list_of_cols="all", drop_cols=[], print_impact=False
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe [attribute, min, 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, max]
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
        warnings.warn("No Percentiles Computation - No numerical column(s) to analyze")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("min", T.StringType(), True),
                T.StructField("1%", T.StringType(), True),
                T.StructField("5%", T.StringType(), True),
                T.StructField("10%", T.StringType(), True),
                T.StructField("25%", T.StringType(), True),
                T.StructField("50%", T.StringType(), True),
                T.StructField("75%", T.StringType(), True),
                T.StructField("90%", T.StringType(), True),
                T.StructField("95%", T.StringType(), True),
                T.StructField("99%", T.StringType(), True),
                T.StructField("max", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    stats = ["min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]
    odf = transpose_dataframe(
        idf.select(list_of_cols).summary(*stats), "summary"
    ).withColumnRenamed("key", "attribute")
    for i in odf.columns:
        if i != "attribute":
            odf = odf.withColumn(i, F.round(F.col(i).cast("Double"), 4))
    odf = odf.select(["attribute"] + stats)
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def measures_of_shape(spark, idf, list_of_cols="all", drop_cols=[], print_impact=False):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to analyse e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe [attribute, skewness, kurtosis]
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
            "No Skewness/Kurtosis Computation - No numerical column(s) to analyze"
        )
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("skewness", T.StringType(), True),
                T.StructField("kurtosis", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    shapes = []
    for i in list_of_cols:
        s, k = idf.select(F.skewness(i), F.kurtosis(i)).first()
        shapes.append([i, s, k])
    odf = (
        spark.createDataFrame(shapes, schema=("attribute", "skewness", "kurtosis"))
        .withColumn("skewness", F.round(F.col("skewness"), 4))
        .withColumn("kurtosis", F.round(F.col("kurtosis"), 4))
    )
    if print_impact:
        odf.show(len(list_of_cols))
    return odf
