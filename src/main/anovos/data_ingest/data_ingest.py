# coding=utf-8
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from anovos.shared.utils import pairwise_reduce


def read_dataset(spark, file_path, file_type, file_configs={}):
    """
    :param spark: Spark Session
    :param file_path: Path to input data (directory or filename).
                      Compatible with local path and s3 path (when running in AWS environment).
    :param file_type: "csv", "parquet", "avro".
                      Avro data source requires an external package to run, which can be configured with
                      spark-submit (--packages org.apache.spark:spark-avro_2.11:2.4.0).
    :param file_configs: This argument is passed in a dictionary format as key/value pairs
                         e.g. {"header": "True","delimiter": "|","inferSchema": "True"} for csv files.
                         All the key/value pairs in this argument are passed as options to DataFrameReader,
                         which is created using SparkSession.read.
    :return: Dataframe
    """
    odf = spark.read.format(file_type).options(**file_configs).load(file_path)
    return odf


def write_dataset(idf, file_path, file_type, file_configs={}, column_order=[]):
    """
    :param idf: Input Dataframe
    :param file_path: Path to output data (directory or filename).
                      Compatible with local path and s3 path (when running in AWS environment).
    :param file_type: "csv", "parquet", "avro".
                      Avro data source requires an external package to run, which can be configured with
                      spark-submit (--packages org.apache.spark:spark-avro_2.11:2.4.0).
    :param file_configs: This argument is passed in dictionary format as key/value pairs.
                         Some of the potential keys are header, delimiter, mode, compression, repartition.
                         compression options - uncompressed, gzip (doesn't work with avro), snappy (only valid for parquet)
                         mode options - error (default), overwrite, append
                         repartition - None (automatic partitioning) or an integer value ()
                         e.g. {"header":"True","delimiter":",",'compression':'snappy','mode':'overwrite','repartition':'10'}.
    :param column_order: list of columns in the order in which Dataframe is to be written. If None or [] is specified, then the default order is applied.
    :return: None (Dataframe saved)
    """

    if not column_order:
        column_order = idf.columns
    else:
        if not isinstance(column_order, list):
            raise TypeError("Invalid input type for column_order argument")
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
    :param dfs: All dataframes to be concatenated (with the first dataframe columns)
    :param method_type: "index", "name".
                        This argument needs to be passed as a keyword argument.
                        "index" method concatenates by column index positioning, without shuffling columns.
                        "name" concatenates after shuffling and arranging columns as per the first dataframe.
                        First dataframe passed under idfs will define the final columns in the concatenated dataframe,
                        and will throw error if any column in first dataframe is not available in any of other dataframes.
    :return: Concatenated Dataframe
    """
    if method_type not in ["index", "name"]:
        raise TypeError("Invalid input for concatenate_dataset method")
    if method_type == "name":
        odf = pairwise_reduce(
            lambda idf1, idf2: idf1.union(idf2.select(idf1.columns)), idfs
        )
        # odf = reduce(DataFrame.unionByName, idfs) # only if exact no. of columns
    else:
        odf = pairwise_reduce(DataFrame.union, idfs)
    return odf


def join_dataset(*idfs, join_cols, join_type):
    """
    :param idfs: All dataframes to be joined
    :param join_cols:  Key column(s) to join all dataframes together.
                       In case of multiple columns to join, they can be passed in a list format or
                       a string format where different column names are separated by pipe delimiter “|”.
    :param join_type: "inner", “full”, “left”, “right”, “left_semi”, “left_anti”
    :return: Joined Dataframe
    """
    if isinstance(join_cols, str):
        join_cols = [x.strip() for x in join_cols.split("|")]
    odf = pairwise_reduce(
        lambda idf1, idf2: idf1.join(idf2, join_cols, join_type), idfs
    )
    return odf


def delete_column(idf, list_of_cols, print_impact=False):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to delete e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe after dropping columns
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
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to select e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :return: Dataframe with the selected columns
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
    :param idf: Input Dataframe
    :param list_of_cols: List of old column names e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param list_of_newcols: List of corresponding new column names e.g., ["newcol1","newcol2"].
                            Alternatively, new column names can be specified in a string format,
                            where different column names are separated by pipe delimiter “|” e.g., "newcol1|newcol2".
                            First element in list_of_cols will be original column name,
                            and corresponding first column in list_of_newcols will be new column name.
    :return: Dataframe with revised column names
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
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to cast e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param list_of_dtypes: List of corresponding datatypes e.g., ["type1","type2"].
                           Alternatively, datatypes can be specified in a string format,
                           where they are separated by pipe delimiter “|” e.g., "type1|type2".
                           First element in list_of_cols will column name and corresponding element in list_of_dtypes
                           will be new datatypes such as "float", "integer", "long", "string", "double", decimal" etc.
                           Datatypes are case insensitive e.g. float or Float are treated as same.
    :return: Dataframe with revised datatypes
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
