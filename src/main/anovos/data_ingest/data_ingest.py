# coding=utf-8
"""
This module consists of functions to read the dataset as Spark DataFrame, concatenate/join with other functions (if required),
and perform some basic ETL actions such as selecting, deleting, renaming and/or recasting columns. List of functions included in this module are:
- read_dataset
- write_dataset
- concatenate_dataset
- join_dataset
- delete_column
- select_column
- rename_column
- recast_column
"""
import warnings

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql import types as T

from anovos.shared.utils import attributeType_segregation, pairwise_reduce


def read_dataset(
    spark,
    file_path,
    file_type,
    file_configs={},
    parquet_conversion=False,
    intermediate_path="",
    treatment=False,
    id_cols=[],
    threshold_num=50,
    threshold_ratio=0.005,
    threshold_string=0.5,
):
    """
    This function reads the input data path and return a Spark DataFrame. Under the hood, this function is based
    on generic Load functionality of Spark SQL. It can also a schema treatment based on cardinality ratio,
    as well as converting the file format to parquet.

    Parameters
    ----------
    spark
        Spark Session
    file_path
        Path to input data (directory or filename).
        Compatible with local path and s3 path (when running in AWS environment).
    file_type
        "csv", "parquet", "avro", "json".
        Avro data source requires an external package to run, which can be configured with spark-submit
        (--packages org.apache.spark:spark-avro_2.11:2.4.0).
    file_configs
        This optional argument is passed in a dictionary format as key/value pairs
        e.g. {"header": "True","delimiter": "|","inferSchema": "True"} for csv files.
        All the key/value pairs in this argument are passed as options to DataFrameReader,
        which is created using SparkSession.read. (Default value = {})
    parquet_conversion
        This boolean flag provides an option to whether convert the file type to parquet or not.
        Schema treatment is highly recommended when using this method, as the schema when converted will
        be kept the same as the original file, thus might defeat the purpose of converting to parquet.
        (Default value = False)
    intermediate_path
        This argument is passed as an intermediate path to write out parquet file, if parquet_conversion
        is set to True. (Default value = '')
    treatment
        This boolean flag provides an option to whether treat the dataframe schema or not. It might not
        work on parquet filetype, as parquet has its own strict Schema.
        (Default value = False)
    id_cols
        This argument is a list contains all the identifier columns, such as ID, Number, Code, etc
        when treatment is set to True.
        (Default value = [])
    threshold_num
        This argument determines the numerical threshold of cardinality for every column. If the number of
        unique values of a column is larger than threshold_num, its DataType will be treated as Double.
        Otherwise its DataType will be treated as String. (Default value = 50)
    threshold_ratio
        This argument determines the cardinality ratio for every column. The cardinality ratio is defined
        by number of non-null unique values divided by number of non-null total values in a column.
        If the cardinality ratio is larger than threshold_ratio, its DataType will be treated as Double.
        Otherwise its DataType will be treated as String. (Default value = 0.005)
    threshold_string
        This argument determines the string ratio for every column. The string ratio is defined
        by number of non-null string values divided by number of non-null total values in a column.
        If the string ratio is smaller than threshold_ratio, its DataType will be treated as Double.
        Otherwise its DataType will be treated as String. (Default value = 0.5)

    Returns
    -------
    DataFrame

    """
    odf = spark.read.format(file_type).options(**file_configs).load(file_path)
    if parquet_conversion:
        if not intermediate_path:
            raise TypeError("intermediate_path cannot be blank for parquet conversion")
    if treatment:
        if id_cols:
            for col in id_cols:
                if col not in odf.columns:
                    raise TypeError("Invalid input for id_cols: " + col)
        if type(threshold_num) != int:
            raise TypeError("Invalid input for threshold_num")
        if type(threshold_ratio) != float or threshold_ratio < 0 or threshold_ratio > 1:
            raise TypeError("Invalid input for threshold_ratio")
        if (
            type(threshold_string) != float
            or threshold_string < 0
            or threshold_string > 1
        ):
            raise TypeError("Invalid input for threshold_string")
        list_of_cols = list(c for c in odf.columns if c not in id_cols)
        odf = odf.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        odf_recast = odf.select(
            [F.col(c).cast(T.DoubleType()).alias(c) for c in list_of_cols]
        )
        funs_count = [F.count]
        exprs_count = [f(F.col(c)) for f in funs_count for c in list_of_cols]
        list_count_recast = (
            odf_recast.groupby().agg(*exprs_count).rdd.flatMap(lambda x: x).collect()
        )

        list_count = odf.groupby().agg(*exprs_count).rdd.flatMap(lambda x: x).collect()

        funs_distinct = [F.countDistinct]
        exprs_distinct = [f(F.col(c)) for f in funs_distinct for c in list_of_cols]
        list_distinct_count = (
            odf.select(*exprs_distinct).rdd.flatMap(lambda x: x).collect()
        )
        list_ratio = [i / j for i, j in zip(list_distinct_count, list_count)]
        list_cat_ratio = [i / j for i, j in zip(list_count_recast, list_count)]

        list_schema = []
        if id_cols:
            for col in id_cols:
                list_schema.append(T.StructField(str(col), T.StringType(), True))
        for k in range(0, len(list_count)):
            if list_cat_ratio[k] < threshold_string:
                list_schema.append(
                    T.StructField(str(list_of_cols[k]), T.StringType(), True)
                )
            elif list_count[k] <= threshold_num:
                if list_ratio[k] > threshold_ratio:
                    list_schema.append(
                        T.StructField(str(list_of_cols[k]), T.DoubleType(), True)
                    )
                else:
                    list_schema.append(
                        T.StructField(str(list_of_cols[k]), T.StringType(), True)
                    )
            elif (
                list_ratio[k] < threshold_ratio
                or list_distinct_count[k] < threshold_num
            ):
                list_schema.append(
                    T.StructField(str(list_of_cols[k]), T.StringType(), True)
                )

            else:
                list_schema.append(
                    T.StructField(str(list_of_cols[k]), T.DoubleType(), True)
                )
        odf.unpersist()
        full_schema = T.StructType(list_schema)
        odf = (
            spark.read.format(file_type)
            .options(**file_configs)
            .schema(full_schema)
            .load(file_path)
        )
    if parquet_conversion:
        file_path = intermediate_path
        file_type = "parquet"
        odf.write.format(file_type).options(**file_configs).save(
            file_path, mode="overwrite"
        )
        odf = spark.read.format(file_type).options(**file_configs).load(file_path)
    return odf


def write_dataset(idf, file_path, file_type, file_configs={}, column_order=[]):
    """
    This function saves the Spark DataFrame in the user-provided output path. Like read_dataset, this function is
    based on the generic Save functionality of Spark SQL.

    Parameters
    ----------
    idf
        Input Dataframe i.e. Spark DataFrame to be saved
    file_path
        Path to output data (directory or filename). Compatible with local path and s3 path (when running in AWS environment).
    file_type
        "csv", "parquet", "avro", "json".
        Avro data source requires an external package to run, which can be configured with spark-submit
        (--packages org.apache.spark:spark-avro_2.11:2.4.0).
    file_configs
        This argument is passed in dictionary format as key/value pairs. Some of the potential keys are header, delimiter,
        mode, compression, repartition.
        compression options - uncompressed, gzip (doesn't work with avro), snappy (only valid for parquet)
        mode options - error (default), overwrite, append repartition - None (automatic partitioning) or an integer value ()
        e.g. {"header":"True", "delimiter":",",'compression':'snappy','mode':'overwrite','repartition':'10'}.
        All the key/value pairs (except repartition, mode) written in this argument are passed as options to DataFrameWriter is available using
        Dataset.write operator. If the number of repartitions mentioned through this argument is less than the existing
        DataFrame partitions, then the coalesce operation is used instead of the repartition operation to make the
        execution work. This is because the coalesce operation doesn’t require any shuffling like repartition which is known to be an expensive step.
    column_order
        list of columns in the order in which Dataframe is to be written. If None or [] is specified, then the default order is applied.

    """

    if not column_order:
        column_order = idf.columns
    else:
        if len(column_order) != len(idf.columns):
            raise ValueError(
                "Count of column(s) specified in column_order argument do not match Dataframe"
            )
        diff_cols = [x for x in column_order if x not in set(idf.columns)]
        if diff_cols:
            raise ValueError(
                "Column(s) specified in column_order argument not found in Dataframe: "
                + str(diff_cols)
            )

    mode = file_configs["mode"] if "mode" in file_configs else "error"
    repartition = (
        int(file_configs["repartition"]) if "repartition" in file_configs else None
    )

    if repartition is None:
        idf.select(column_order).write.format(file_type).options(**file_configs).save(
            file_path, mode=mode
        )
    else:
        exist_parts = idf.rdd.getNumPartitions()
        req_parts = int(repartition)
        if req_parts > exist_parts:
            idf.select(column_order).repartition(req_parts).write.format(
                file_type
            ).options(**file_configs).save(file_path, mode=mode)
        else:
            idf.select(column_order).coalesce(req_parts).write.format(
                file_type
            ).options(**file_configs).save(file_path, mode=mode)


def concatenate_dataset(*idfs, method_type="name"):
    """
    This function combines multiple dataframes into a single dataframe. A pairwise concatenation is performed on
    the dataframes, instead of adding one dataframe at a time to the bigger dataframe. This function leverages union
    functionality of Spark SQL.

    Parameters
    ----------
    *idfs
        All dataframes to be concatenated (with the first dataframe columns)
    method_type
        "index", "name". This argument needs to be passed as a keyword argument.
        The “index” method concatenates the dataframes by the column index (without shuffling columns).
        If the sequence of column is not fixed among the dataframe, this method should be avoided.
        The “name” method concatenates after shuffling and arranging columns as per the first dataframe order.
        First dataframe passed under idfs will define the final columns in the concatenated dataframe, and
        will throw error if any column in first dataframe is not available in any of other dataframes. (Default value = "name")

    Returns
    -------
    DataFrame
        Concatenated dataframe

    """
    if method_type not in ["index", "name"]:
        raise TypeError("Invalid input for concatenate_dataset method")
    if method_type == "name":
        odf = pairwise_reduce(
            lambda idf1, idf2: idf1.union(idf2.select(idf1.columns)), idfs
        )  # odf = reduce(DataFrame.unionByName, idfs) # only if exact no. of columns
    else:
        odf = pairwise_reduce(DataFrame.union, idfs)
    return odf


def join_dataset(*idfs, join_cols, join_type):
    """
    This function joins multiple dataframes into a single dataframe by joining key column(s). For optimization, Pairwise joining is
    done on the dataframes, instead of joining individual dataframes to the bigger dataframe. This function leverages
    join functionality of Spark SQL.

    Parameters
    ----------
    idfs
        All dataframes to be joined
    join_cols
        Key column(s) to join all dataframes together.
        In case of multiple key columns to join, they can be passed in a list format or
        a string format where different column names are separated by pipe delimiter “|” e.g. "col1|col2".
    join_type
        "inner", “full”, “left”, “right”, “left_semi”, “left_anti”

    Returns
    -------
    DataFrame
        Joined dataframe

    """
    if isinstance(join_cols, str):
        join_cols = [x.strip() for x in join_cols.split("|")]

    list_of_df_cols = [x.columns for x in idfs]
    list_of_all_cols = [x for sublist in list_of_df_cols for x in sublist]
    list_of_nonjoin_cols = [x for x in list_of_all_cols if x not in join_cols]

    if len(list_of_nonjoin_cols) != (
        len(list_of_all_cols) - (len(list_of_df_cols) * len(join_cols))
    ):
        raise ValueError("Specified join_cols do not match all the Input Dataframe(s)")

    if len(list_of_nonjoin_cols) != len(set(list_of_nonjoin_cols)):
        raise ValueError(
            "Duplicate column(s) present in non joining column(s) in Input Dataframe(s)"
        )

    odf = pairwise_reduce(
        lambda idf1, idf2: idf1.join(idf2, join_cols, join_type), idfs
    )
    return odf


def delete_column(idf, list_of_cols, print_impact=False):
    """
    This function is used to delete specific columns from the input data. It is executed using drop functionality
    of Spark SQL. It is advisable to use this function if the number of columns to delete is lesser than the number
    of columns to select; otherwise, it is recommended to use select_column.

    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of columns to delete e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    print_impact
        True, False
        This argument is to compare number of columns before and after the operation.(Default value = False)

    Returns
    -------
    DataFrame
        Dataframe after dropping columns

    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    list_of_cols = list(set(list_of_cols))

    odf = idf.drop(*list_of_cols)

    if print_impact:
        print("Before: \nNo. of Columns- ", len(idf.columns))
        print(idf.columns)
        print("After: \nNo. of Columns- ", len(odf.columns))
        print(odf.columns)
    return odf


def select_column(idf, list_of_cols, print_impact=False):
    """
    This function is used to select specific columns from the input data. It is executed using select operation of
    spark dataframe. It is advisable to use this function if the number of columns to select is lesser than the
    number of columns to drop; otherwise, it is recommended to use delete_column.

    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of columns to select e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    print_impact
        True, False
        This argument is to compare number of columns before and after the operation.(Default value = False)

    Returns
    -------
    DataFrame
        Dataframe with the selected columns

    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    list_of_cols = list(set(list_of_cols))

    odf = idf.select(list_of_cols)

    if print_impact:
        print("Before: \nNo. of Columns-", len(idf.columns))
        print(idf.columns)
        print("\nAfter: \nNo. of Columns-", len(odf.columns))
        print(odf.columns)
    return odf


def rename_column(idf, list_of_cols, list_of_newcols, print_impact=False):
    """
    This function is used to rename the columns of the input data. Multiple columns can be renamed; however,
    the sequence they passed as an argument is critical and must be consistent between list_of_cols and
    list_of_newcols.

    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of old column names e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    list_of_newcols
        List of corresponding new column names e.g., ["newcol1","newcol2"].
        Alternatively, new column names can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "newcol1|newcol2".
        First element in list_of_cols will be original column name, and corresponding first column in list_of_newcols will be new column name.
    print_impact
        True, False
        This argument is to compare column names before and after the operation. (Default value = False)

    Returns
    -------
    DataFrame
        Dataframe with revised column names

    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(list_of_newcols, str):
        list_of_newcols = [x.strip() for x in list_of_newcols.split("|")]

    mapping = dict(zip(list_of_cols, list_of_newcols))
    odf = idf.select([F.col(i).alias(mapping.get(i, i)) for i in idf.columns])

    if print_impact:
        print("Before: \nNo. of Columns- ", len(idf.columns))
        print(idf.columns)
        print("After: \nNo. of Columns- ", len(odf.columns))
        print(odf.columns)
    return odf


def recast_column(idf, list_of_cols, list_of_dtypes, print_impact=False):
    """
    This function is used to modify the datatype of columns. Multiple columns can be cast; however,
    the sequence they passed as argument is critical and needs to be consistent between list_of_cols and
    list_of_dtypes.

    Parameters
    ----------
    idf
        Input Dataframe
    list_of_cols
        List of columns to cast e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    list_of_dtypes
        List of corresponding datatypes e.g., ["type1","type2"].
        Alternatively, datatypes can be specified in a string format,
        where they are separated by pipe delimiter “|” e.g., "type1|type2".
        First element in list_of_cols will column name and corresponding element in list_of_dtypes
        will be new datatypes such as "float", "integer", "long", "string", "double", decimal" etc.
        Datatypes are case insensitive e.g. float or Float are treated as same.
    print_impact
        True, False
        This argument is to compare schema before and after the operation. (Default value = False)

    Returns
    -------
    DataFrame
        Dataframe with revised datatypes

    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(list_of_dtypes, str):
        list_of_dtypes = [x.strip() for x in list_of_dtypes.split("|")]

    odf = idf
    for i, j in zip(list_of_cols, list_of_dtypes):
        odf = odf.withColumn(i, F.col(i).cast(j))

    if print_impact:
        print("Before: ")
        idf.printSchema()
        print("After: ")
        odf.printSchema()
    return odf


def recommend_type(
    spark,
    idf,
    list_of_cols="all",
    drop_cols=[],
    dynamic_threshold=0.01,
    static_threshold=100,
):
    """
    This function is to recommend the form and datatype of columns. Cardinality of each column will be measured,
    then both dynamic_threshold and static_threshold will be used to determine the recommended form and datatype
    for each column.

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe
    list_of_cols
        List of columns to cast e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2". (Default value = 'all')
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of strata_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    dynamic_threshold
        Cardinality threshold to determine columns recommended form and datatype.
        If the column's unique values < column total records * dynamic_threshold, column will be recommended as
        categorical, and in string datatype. Else, column will be recommended as numerical, and in double datatype
        In recommend_type, we will use the general threshold equals to the minimum of dynamic_threshold and
        static_threshold. (Default value = 0.01)
    static_threshold
        Cardinality threshold to determine columns recommended form and datatype.
        If the column's unique values < static_threshold, column will be recommended as
        categorical, and in string datatype. Else, column will be recommended as numerical, and in double datatype
        In recommend_type, we will use the general threshold equals to the minimum of dynamic_threshold and
        static_threshold. (Default value = 100)


    Returns
    -------
    DataFrame
        Dataframe with attributes and their original/recommended form and datatype

    """
    if list_of_cols == "all":
        list_of_cols = idf.columns

    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]

    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")

    if len(list_of_cols) == 0:
        warnings.warn("No recommend_attributeType analysis - No column(s) to analyze")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("original_form", T.StringType(), True),
                T.StructField("original_dataType", T.StringType(), True),
                T.StructField("recommended_form", T.StringType(), True),
                T.StructField("recommended_dataType", T.StringType(), True),
                T.StructField("distinct_value_count", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf

    if type(dynamic_threshold) != float:
        raise TypeError("Invalid input for dynamic_threshold: float type only")

    if dynamic_threshold <= 0 or dynamic_threshold > 1:
        raise TypeError(
            "Invalid input for dynamic_threshold: Value need to be between 0 and 1"
        )

    if type(static_threshold) != int:
        raise TypeError("Invalid input for static_threshold: int type only")

    def min_val(val1, val2):
        if val1 > val2:
            return val2
        else:
            return val1

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)
    rec_num_cols = []
    rec_cat_cols = []
    for col in num_cols:
        if idf.select(col).distinct().na.drop().count() < min_val(
            (dynamic_threshold * idf.select(col).na.drop().count()), static_threshold
        ):
            rec_cat_cols.append(col)

    for col in cat_cols:
        idf_inter = (
            idf.na.drop(subset=col).withColumn(col, idf[col].cast("double")).select(col)
        )
        if (
            idf_inter.distinct().na.drop().count() == idf_inter.distinct().count()
            and idf_inter.distinct().na.drop().count() != 0
        ):
            if idf.select(col).distinct().na.drop().count() >= min_val(
                (dynamic_threshold * idf.select(col).na.drop().count()),
                static_threshold,
            ):
                rec_num_cols.append(col)

    rec_cols = rec_num_cols + rec_cat_cols
    ori_form = []
    ori_type = []
    rec_form = []
    rec_type = []
    num_dist_val = []
    if len(rec_cols) > 0:
        for col in rec_cols:
            if col in rec_num_cols:
                ori_form.append("categorical")
                ori_type.append(idf.select(col).dtypes[0][1])
                rec_form.append("numerical")
                rec_type.append("double")
                num_dist_val.append(idf.select(col).distinct().count())
            else:
                ori_form.append("numerical")
                ori_type.append(idf.select(col).dtypes[0][1])
                rec_form.append("categorical")
                rec_type.append("string")
                num_dist_val.append(idf.select(col).distinct().count())
        odf_rec = spark.createDataFrame(
            zip(rec_cols, ori_form, ori_type, rec_form, rec_type, num_dist_val),
            schema=(
                "attribute",
                "original_form",
                "original_dataType",
                "recommended_form",
                "recommended_dataType",
                "distinct_value_count",
            ),
        )
        return odf_rec
    else:
        warnings.warn("No column type change recommendation is made")
        schema = T.StructType(
            [
                T.StructField("attribute", T.StringType(), True),
                T.StructField("original_form", T.StringType(), True),
                T.StructField("original_dataType", T.StringType(), True),
                T.StructField("recommended_form", T.StringType(), True),
                T.StructField("recommended_dataType", T.StringType(), True),
                T.StructField("distinct_value_count", T.StringType(), True),
            ]
        )
        odf = spark.sparkContext.emptyRDD().toDF(schema)
        return odf
