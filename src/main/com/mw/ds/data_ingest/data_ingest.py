from typing import List, Tuple
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from com.mw.ds.shared.spark import *


def read_dataset(file_path:List[str], file_type: str, file_configs={}):
    """
    :param file_path: Path to input data (directory or filename)
    :param file_type: csv, parquet
    :param file_configs: passing arguments in dict format e.g. {"header": "True", "delimiter": ",","inferSchema": "True"}
    :return: dataframe
    """
    odf = spark.read.format(file_type).options(**file_configs).load(file_path)
    return odf


def write_dataset(idf, file_path, file_type, file_configs={}):
    """
    :param idf: input dataframe
    :param file_path: Path to input data (directory or filename)
    :param file_type: csv, parquet
    :param file_configs: passing arguments in dict format - header, delimiter, mode, compression, repartition
                   compression options - uncompressed, gzip, snappy (only valid for parquet)
                   mode options - error (default), overwrite, append
                   repartition - None or int (no. of part files to generate)
                   {"header":"True","delimiter":",",'compression':'snappy','mode':'overwrite','repartition':'10'}
    :return: None, dataframe saved
    """

    mode = file_configs['mode'] if 'mode' in file_configs else 'error'
    repartition = int(file_configs['repartition']
                      ) if 'repartition' in file_configs else None

    if repartition is None:
        idf.write.format(file_type).options(
            **file_configs).save(file_path, mode=mode)
    else:
        exist_parts = idf.rdd.getNumPartitions()
        req_parts = int(repartition)
        if req_parts > exist_parts:
            idf.repartition(req_parts).write.format(file_type).options(
                **file_configs).save(file_path, mode=mode)
        else:
            idf.coalesce(req_parts).write.format(file_type).options(
                **file_configs).save(file_path, mode=mode)


def pairwise_reduce(op, x):
    while len(x) > 1:
        v = [op(i, j) for i, j in zip(x[::2], x[1::2])]
        if len(x) > 1 and len(x) % 2 == 1:
            v[-1] = op(v[-1], x[-1])
        x = v
    return x[0]


def concatenate_dataset(*idfs, method_type='name'):
    """
    :param dfs: all dataframes to be concatenated (with 1st dataframe columns)
    :param method_type: index (concatenating without shuffling columns) or name (concatenating after shuffling columns)
    :return: Concatenated dataframe 
    """
    from functools import reduce
    from pyspark.sql import DataFrame
    if (method_type not in ['index', 'name']):
        raise TypeError('Invalid input for concatenate_dataset method')
    if method_type == 'name':
        odf = pairwise_reduce(lambda idf1, idf2: idf1.union(
            idf2.select(idf1.columns)), idfs)
        # odf = reduce(DataFrame.unionByName, idfs) # only if exact no. of columns
    else:
        odf = pairwise_reduce(DataFrame.union, idfs)
    return odf


def join_dataset(*idfs, join_cols, join_type):
    """
    :param idfs: all dataframes to be joined
    :param join_cols: joining columns (str separated by | or list of columns) to join all dfs
    :param join_type: inner, full, left, right, left_semi, left_anti
    :return: Joined dataframe
    """
    if isinstance(join_cols, str):
        join_cols = [x.strip() for x in join_cols.split('|')]
    odf = pairwise_reduce(lambda idf1, idf2: idf1.join(
        idf2, join_cols, join_type), idfs)
    return odf


def delete_column(idf, list_of_cols, print_impact=False):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to delete (list or string of col names separated by |)
    :return: dataframe after dropping columns
    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
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
    :param list_of_cols: List of columns to select (list or string of col names separated by |)
    :return: dataframe after selected columns
    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    odf = idf.select(list_of_cols)

    if print_impact:
        print("Before: \nNo. of Columns- ", len(idf.columns))
        print(idf.columns)
        print("After: \nNo. of Columns- ", len(odf.columns))
        print(odf.columns)
    return odf


def rename_column(idf, list_of_cols, list_of_newcols, print_impact=False):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of old column names (list or string of col names separated by |)
    :param list_of_newcols: List of new column names (list or string of col names separated by |)
    :return: dataframe with revised names
    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(list_of_newcols, str):
        list_of_newcols = [x.strip() for x in list_of_newcols.split('|')]

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
    :param list_of_cols: List of column to cast (list or string of col names separated by |)
    :param list_of_dtypes: List of corresponding datatype (list or string of col names separated by |)
                    Float, Integer,Decimal, Long, String etc (case insensitive)
    :return: dataframe with revised datatypes
    """
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(list_of_dtypes, str):
        list_of_dtypes = [x.strip() for x in list_of_dtypes.split('|')]

    odf = idf
    for i, j in zip(list_of_cols, list_of_dtypes):
        odf = odf.withColumn(i, F.col(i).cast(j))

    if print_impact:
        print("Before: ")
        idf.printSchema()
        print("After: ")
        odf.printSchema()
    return odf


class DataIngestion:
    def __init__(self, spark) -> None:
        self.spark = spark

    def read_dataset(self, file_path, file_type, file_configs={}):
        """
        :param file_path: Path to input data (directory or filename)
        :param file_type: csv, parquet
        :param file_configs: passing arguments in dict format e.g. {"header": "True", "delimiter": ",","inferSchema": "True"}
        :return: dataframe
        """
        odf = self.spark.read.format(file_type).options(
            **file_configs).load(file_path)
        return odf

    def infer_schema(self, paths:List[str], file_type: str, file_configs={}):
        return self.read_dataset(paths, file_type, file_configs).schema


    def generate_data(self, inputDf, selectColumns: List[str], castColumns:List[Tuple[str,str]], renameColumns:List[Tuple[str,str]]):
        df = inputDf
        df = self.select_column(df,selectColumns)
        df = self.__recast_column(df,castColumns)
        df = self.rename_column(df,renameColumns)
        return df



    def write_dataset(self, idf, file_path, file_type, file_configs={}):
        """
        :param idf: input dataframe
        :param file_path: Path to input data (directory or filename)
        :param file_type: csv, parquet
        :param file_configs: passing arguments in dict format - header, delimiter, mode, compression, repartition
                    compression options - uncompressed, gzip, snappy (only valid for parquet)
                    mode options - error (default), overwrite, append
                    repartition - None or int (no. of part files to generate)
                    {"header":"True","delimiter":",",'compression':'snappy','mode':'overwrite','repartition':'10'}
        :return: None, dataframe saved
        """
        mode = file_configs['mode'] if 'mode' in file_configs else 'error'
        repartition = int(file_configs['repartition']
                        ) if 'repartition' in file_configs else None

        if repartition is None:
            idf.write.format(file_type).options(
                **file_configs).save(file_path, mode=mode)
        else:
            exist_parts = idf.rdd.getNumPartitions()
            req_parts = int(repartition)
            if req_parts > exist_parts:
                idf.repartition(req_parts).write.format(file_type).options(
                    **file_configs).save(file_path, mode=mode)
            else:
                idf.coalesce(req_parts).write.format(file_type).options(
                    **file_configs).save(file_path, mode=mode) 
    
    def rename_column(self, idf, list_of_cols : List[Tuple[str,str]]):
        """
        :param idf: Input Dataframe
        :param list_of_cols: List of tupple of  old column names and new columns
        :return: dataframe with revised names
        """

        odf = idf
        for column_name, new_column_name in (list_of_cols):
            odf = odf.withColumnRenamed(column_name, new_column_name)
        return odf

        # odf = idf.select([F.col(column_name).alias(new_column_name) for column_name, new_column_name in list_of_cols])
        # return odf  


    def __recast_column(self, df, list_of_cols : List[Tuple[str,str]]):
        """
        :param idf: Input Dataframe
        :param list_of_cols: List of column to cast (list or string of col names separated by |)
        :param list_of_dtypes: List of corresponding datatype (list or string of col names separated by |)
                        Float, Integer,Decimal, Long, String etc (case insensitive)
        :return: dataframe with revised datatypes
        """

        odf = df
        for column_name, new_data_type in (list_of_cols):
            odf = odf.withColumn(column_name, F.col(column_name).cast(new_data_type))
        return odf


    def select_column(self, idf, list_of_cols:List[str]):
        """
        :param idf: Input Dataframe
        :param list_of_cols: List of columns to select (list or string of col names separated by |)
        :return: dataframe after selected columns
        """
        odf = idf.select([x.strip() for x in list_of_cols])

        return odf