# coding=utf-8

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from pyspark.mllib.stat import Statistics
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, ImputerModel, StringIndexer, IndexToString, OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, MinMaxScalerModel, PCA, PCAModel
from pyspark.ml.linalg import DenseVector

from anovos.data_analyzer.stats_generator import missingCount_computation, uniqueCount_computation
from anovos.data_ingest.data_ingest import read_dataset, recast_column
from anovos.shared.utils import attributeType_segregation, get_dtype

from sklearn import cluster
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.validation import column_or_1d
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU

import imp
import os
import joblib
import pickle
import random
import tempfile
import warnings
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
from operator import mod
from typing import Iterable 
from itertools import chain
from matplotlib import pyplot
from dcor import distance_correlation


def attribute_binning(spark, idf, list_of_cols='all', drop_cols=[], method_type="equal_range", bin_size=10,
                      bin_dtype="numerical",
                      pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param bin_method: "equal_frequency", "equal_range".
                        In "equal_range" method, each bin is of equal size/width and in "equal_frequency", each bin has
                        equal no. of rows, though the width of bins may vary.
    :param bin_size: Number of bins.
    :param bin_dtype: "numerical", "categorical".
                      With "numerical" option, original value is replaced with an Integer (1,2,…) and
                      with "categorical" option, original replaced with a string describing min and max value allowed
                      in the bin ("minval-maxval").
    :param pre_existing_model: Boolean argument – True or False. True if binning model exists already, False Otherwise.
    :param model_path: If pre_existing_model is True, this argument is path for referring the pre-saved model.
                       If pre_existing_model is False, this argument can be used for saving the model.
                       Default "NA" means there is neither pre-existing model nor there is a need to save one.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with transformed column. “append” option append transformed
                        column to the input dataset with a postfix "_binned" e.g. column X is appended as X_binned.
    :return: Binned Dataframe
    """

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if len(list_of_cols) == 0:
        warnings.warn("No Binning Performed - No numerical column(s) to transform")
        return idf

    if method_type not in ("equal_frequency", "equal_range"):
        raise TypeError('Invalid input for method_type')
    if bin_size < 2:
        raise TypeError('Invalid input for bin_size')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model:
        df_model = spark.read.parquet(model_path + "/attribute_binning")
        bin_cutoffs = []
        for i in list_of_cols:
            mapped_value = df_model.where(F.col('attribute') == i).select('parameters') \
                .rdd.flatMap(lambda x: x).collect()[0]
            bin_cutoffs.append(mapped_value)
    else:
        if method_type == "equal_frequency":
            pctile_width = 1 / bin_size
            pctile_cutoff = []
            for j in range(1, bin_size):
                pctile_cutoff.append(j * pctile_width)
            bin_cutoffs = idf.approxQuantile(list_of_cols, pctile_cutoff, 0.01)

        else:
            bin_cutoffs = []
            for i in list_of_cols:
                max_val = (idf.select(F.col(i)).groupBy().max().rdd.flatMap(lambda x: x).collect() + [None])[0]
                min_val = (idf.select(F.col(i)).groupBy().min().rdd.flatMap(lambda x: x).collect() + [None])[0]
                bin_cutoff = []
                if max_val:
                    bin_width = (max_val - min_val) / bin_size
                    for j in range(1, bin_size):
                        bin_cutoff.append(min_val + j * bin_width)
                bin_cutoffs.append(bin_cutoff)

        if model_path != "NA":
            df_model = spark.createDataFrame(zip(list_of_cols, bin_cutoffs), schema=['attribute', 'parameters'])
            df_model.write.parquet(model_path + "/attribute_binning", mode='overwrite')

    def bucket_label(value, index):
        if value is None:
            return None
        for j in range(0, len(bin_cutoffs[index])):
            if value <= bin_cutoffs[index][j]:
                if bin_dtype == "numerical":
                    return j + 1
                else:
                    if j == 0:
                        return "<= " + str(round(bin_cutoffs[index][j], 4))
                    else:
                        return str(round(bin_cutoffs[index][j - 1], 4)) + "-" + str(round(bin_cutoffs[index][j], 4))
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

        if idx % 5 == 0:
            odf.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    if output_mode == 'replace':
        for col in list_of_cols:
            odf = odf.drop(col).withColumnRenamed(col + "_binned", col)

    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_binned") for i in list_of_cols]
        uniqueCount_computation(spark, odf, output_cols).show(len(output_cols))
    return odf


def monotonic_binning(spark, idf, list_of_cols='all', drop_cols=[], label_col='label', event_label=1,
                      bin_method="equal_range", bin_size=10, bin_dtype="numerical", output_mode="replace"):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param label_col: Label/Target column
    :param event_label: Value of (positive) event (i.e label 1)
    :param bin_method: "equal_frequency", "equal_range".
                        In "equal_range" method, each bin is of equal size/width and in "equal_frequency", each bin has
                        equal no. of rows, though the width of bins may vary.
    :param bin_size: Default number of bins in case monotonicity is not achieved.
    :param bin_dtype: "numerical", "categorical".
                      With "numerical" option, original value is replaced with an Integer (1,2,…) and
                      with "categorical" option, original replaced with a string describing min and max value allowed
                      in the bin ("minval-maxval").
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with transformed column. “append” option append transformed
                        column to the input dataset with a postfix "_binned" e.g. column X is appended as X_binned.
    :return: Binned Dataframe
    """
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in (drop_cols + [label_col])]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')

    attribute_binning(spark, idf, list_of_cols='all', drop_cols=[], method_type="equal_range", bin_size=10,
                      pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False)

    odf = idf
    for col in list_of_cols:
        n = 20
        r = 0
        while n > 2:
            tmp = attribute_binning(spark, idf, [col], drop_cols=[], method_type=bin_method, bin_size=n,
                                    output_mode='append') \
                .select(label_col, col, col + '_binned') \
                .withColumn(label_col, F.when(F.col(label_col) == event_label, 1).otherwise(0)) \
                .groupBy(col + '_binned').agg(F.avg(col).alias('mean_val'),
                                              F.avg(label_col).alias('mean_label')).dropna()
            r, p = stats.spearmanr(tmp.toPandas()[['mean_val']], tmp.toPandas()[['mean_label']])
            if r == 1.0:
                odf = attribute_binning(spark, odf, [col], drop_cols=[], method_type=bin_method, bin_size=n,
                                        bin_dtype=bin_dtype, output_mode=output_mode)
                break
            n = n - 1
            r = 0
        if r < 1.0:
            odf = attribute_binning(spark, odf, [col], drop_cols=[], method_type=bin_method, bin_size=bin_size,
                                    bin_dtype=bin_dtype, output_mode=output_mode)

    return odf


def cat_to_num_unsupervised(spark, idf, list_of_cols='all', drop_cols=[], method_type=1, index_order='frequencyDesc',
                            pre_existing_model=False, model_path="NA", output_mode='replace', print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of categorical columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all categorical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param method_type: 1 for Label Encoding or 0 for One hot encoding.
                        In label encoding, each categorical value is assigned a unique integer based on alphabetical
                        or frequency ordering (both ascending & descending options are available that can be selected by
                        index_order argument).
                        In one-hot encoding, every unique value in the column will be added in a form of dummy/binary column.
    :param index_order: "frequencyDesc", "frequencyAsc", "alphabetDesc", "alphabetAsc".
                        Valid only for Label Encoding method_type.
    :param pre_existing_model: Boolean argument – True or False. True if encoding model exists already, False Otherwise.
    :param model_path: If pre_existing_model is True, this argument is path for referring the pre-saved model.
                       If pre_existing_model is False, this argument can be used for saving the model.
                       Default "NA" means there is neither pre existing model nor there is a need to save one.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with transformed column. “append” option append transformed
                        column to the input dataset with a postfix "_index" e.g. column X is appended as X_index.
    :return: Encoded Dataframe
    """

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == 'all':
        list_of_cols = cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in cat_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')

    if len(list_of_cols) == 0:
        warnings.warn("No Encoding Computation - No categorical column(s) to transform")
        return idf
    if method_type not in (0, 1):
        raise TypeError('Invalid input for method_type')
    if index_order not in ('frequencyDesc', 'frequencyAsc', 'alphabetDesc', 'alphabetAsc'):
        raise TypeError('Invalid input for Encoding Index Order')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model:
        pipelineModel = PipelineModel.load(model_path + "/cat_to_num_unsupervised/indexer")
    else:
        stages = []
        for i in list_of_cols:
            stringIndexer = StringIndexer(inputCol=i, outputCol=i + '_index',
                                          stringOrderType=index_order, handleInvalid='keep')
            stages += [stringIndexer]
        pipeline = Pipeline(stages=stages)
        pipelineModel = pipeline.fit(idf)

    odf_indexed = pipelineModel.transform(idf)

    if method_type == 0:
        list_of_cols_vec = []
        list_of_cols_idx = []
        for i in list_of_cols:
            list_of_cols_vec.append(i + "_vec")
            list_of_cols_idx.append(i + "_index")
        if pre_existing_model:
            encoder = OneHotEncoderEstimator.load(model_path + "/cat_to_num_unsupervised/encoder")
        else:
            encoder = OneHotEncoderEstimator(inputCols=list_of_cols_idx, outputCols=list_of_cols_vec,
                                             handleInvalid='keep')

        odf_encoded = encoder.fit(odf_indexed).transform(odf_indexed)

        odf = odf_encoded
        selected_cols = odf_encoded.columns
        for i in list_of_cols:
            uniq_cats = idf.select(i).distinct().count()

            def vector_to_array(v):
                v = DenseVector(v)
                new_array = list([int(x) for x in v])
                return new_array

            f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.IntegerType()))

            odf = odf.withColumn("tmp", f_vector_to_array(i + '_vec')) \
                .select(selected_cols + [F.col("tmp")[j].alias(i + "_" + str(j)) for j in range(0, uniq_cats)])
            if output_mode == 'replace':
                selected_cols = [e for e in odf.columns if e not in (i, i + '_vec', i + '_index')]
            else:
                selected_cols = [e for e in odf.columns if e not in (i + '_vec', i + '_index')]
            odf = odf.select(selected_cols)
    else:
        odf = odf_indexed
        for i in list_of_cols:
            odf = odf.withColumn(i + '_index', F.when(F.col(i).isNull(), None)
                                 .otherwise(F.col(i + '_index').cast(T.IntegerType())))
        if output_mode == 'replace':
            for i in list_of_cols:
                odf = odf.drop(i).withColumnRenamed(i + '_index', i)
            odf = odf.select(idf.columns)

    if (pre_existing_model == False) & (model_path != "NA"):
        pipelineModel.write().overwrite().save(model_path + "/cat_to_num_unsupervised/indexer")
        if method_type == 0:
            encoder.write().overwrite().save(model_path + "/cat_to_num_unsupervised/encoder")

    if (print_impact == True) & (method_type == 1):
        print("Before")
        idf.describe().where(F.col('summary').isin('count', 'min', 'max')).show()
        print("After")
        odf.describe().where(F.col('summary').isin('count', 'min', 'max')).show()
    if (print_impact == True) & (method_type == 0):
        print("Before")
        idf.printSchema()
        print("After")
        odf.printSchema()

    return odf


def z_standardization(spark, idf, list_of_cols='all', drop_cols=[], pre_existing_model=False, model_path="NA", 
                      output_mode='replace', print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: List of columns for standarization
    pre_existing_model: Mean/stddev for each feature. True if model files exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  Default NA means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Scaled Dataframe
    '''
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if len(list_of_cols) == 0:
        warnings.warn("No Transformation Performed - Normalization")
        return idf

    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')
    
    if pre_existing_model:
        df_model = spark.read.parquet(model_path+"/z_standardization")
        parameters = []
        for i in list_of_cols:
            mapped_value = df_model.where(F.col('feature') == i).select('parameters').rdd.flatMap(lambda x:x).collect()[0]
            parameters.append(mapped_value)
    else:
        parameters = []
        for i in list_of_cols:
            mean, sttdev = idf.select(F.mean(i), F.stddev(i)).first()
            mean, sttdev = float(mean), float(sttdev)
            parameters.append([mean, sttdev])
    
    odf = idf
    for index, i in enumerate(list_of_cols):
        modify_col = ((i + "_scaled") if (output_mode == "append") else i)
        odf = odf.withColumn(modify_col, (F.col(i) - parameters[index][0])/parameters[index][1])
    
    # Saving Model File if required
    if (pre_existing_model == False) & (model_path != "NA"):
        df_model = spark.createDataFrame(zip(list_of_cols, parameters), schema=['feature', 'parameters'])
        df_model.repartition(1).write.parquet(model_path+"/z_standardization", mode='overwrite')
        
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_scaled") for i in list_of_cols]
        print("Before: ")
        idf.select(list_of_cols).describe().show(10, False)
        print("After: ")
        odf.select(output_cols).describe().show(10, False)
    
    return odf


def IQR_standardization(spark, idf, list_of_cols='all', drop_cols=[], pre_existing_model=False, model_path="NA", 
                        output_mode='replace', print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: List of columns for standarization
    pre_existing_model: 25/50/75 percentile for each feature. True if model files exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  Default NA means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Scaled Dataframe
    '''
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if len(list_of_cols) == 0:
        warnings.warn("No Transformation Performed - Normalization")
        return idf

    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model:
        df_model = spark.read.parquet(model_path+"/IQR_standardization")
        parameters = []
        for i in list_of_cols:
            mapped_value = df_model.where(F.col('feature') == i).select('parameters').rdd.flatMap(lambda x:x).collect()[0]
            parameters.append(mapped_value)
    else:
        parameters = idf.approxQuantile(list_of_cols, [0.25,0.5,0.75], 0.01)
    
    # Note: drop columns with identical 75th and 25th percentiles (o/w all values in odf will be null)
    parameters_, excluded_col = [], []
    for i, param in zip(list_of_cols, parameters):
        if param[0]!=param[2]:
            parameters_.append(param)
        else:
            parameters_.append([0, 0, 1])
            excluded_col.append(i)
    if len(excluded_col) > 0:
        warnings.warn('The original values of the following column(s) are returned because the 75th and 25th percentiles are the same:'+str(excluded_col))
    parameters = parameters_
    
    odf = idf
    for index, i in enumerate(list_of_cols):
        modify_col = ((i + "_scaled") if (output_mode == "append") else i)
        odf = odf.withColumn(modify_col, (F.col(i) - parameters[index][1])/(parameters[index][2] - parameters[index][0]))
    if (pre_existing_model == False) & (model_path != "NA"):
        df_model = spark.createDataFrame(zip(list_of_cols, parameters), schema=['feature', 'parameters'])
        df_model.repartition(1).write.parquet(model_path+"/IQR_standardization", mode='overwrite') 
    
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_scaled") for i in list_of_cols]
        print("Before: ")
        idf.select(list_of_cols).describe().show(10, False)
        print("After: ")
        odf.select(output_cols).describe().show(10, False)
    
    return odf


def normalization(spark, idf, list_of_cols='all', drop_cols=[], pre_existing_model=False, model_path="NA", 
                  output_mode='replace', print_impact=False):
    '''
    idf: Pyspark Dataframe
    list_of_cols: List of columns for normalization
    pre_existing_model: True if normalization/scalar model exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for normalization model. If pre_existing_model is False, 
                this argument can be used for saving the normalization model. 
                Default ("NA") means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Scaled Dataframe
    '''
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if len(list_of_cols) == 0:
        warnings.warn("No Transformation Performed - Normalization")
        return idf

    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')
    
    # Building new scalar model or uploading the existing model
    assembler_norm = VectorAssembler(inputCols=list_of_cols, outputCol="list_of_cols_vector", handleInvalid="keep")
    assembled_norm_data = assembler_norm.transform(idf) 
    if pre_existing_model == True:
        scalerModel = MinMaxScalerModel.load(model_path+"/normalization")
    else:
        scaler = MinMaxScaler(inputCol="list_of_cols_vector", outputCol="list_of_cols_scaled")
        scalerModel = scaler.fit(assembled_norm_data)
    # Applying model
    scaledData = scalerModel.transform(assembled_norm_data)
    # Saving model if required
    if (pre_existing_model == False) & (model_path != "NA"):
        scalerModel.write().overwrite().save(model_path+"/normalization")
    
    # Converting normalization output back into individual features
    def vector_to_array(v):
        return v.toArray().tolist()
    f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.FloatType()))
    odf = scaledData.withColumn("list_of_cols_array", f_vector_to_array('list_of_cols_scaled')).drop(*['list_of_cols_scaled',"list_of_cols_vector"])
    
    odf = odf.select(odf.columns + [(F.when(F.isnan(F.col("list_of_cols_array")[i]),None).otherwise(F.col("list_of_cols_array")[i])).alias(list_of_cols[i]+"_scaled") for i in range(len(list_of_cols))])            .drop("list_of_cols_array")
    
    if output_mode =='replace':
        for i in list_of_cols:
            odf = odf.drop(i).withColumnRenamed(i+"_scaled",i)
            
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_scaled") for i in list_of_cols]
        print("Before: ")
        idf.select(list_of_cols).describe().show(10, False)
        print("After: ")
        odf.select(output_cols).describe().show(10, False)
    
    return odf


def imputation_MMM(spark, idf, list_of_cols="missing", drop_cols=[], method_type="median", pre_existing_model=False,
                   model_path="NA",
                   output_mode="replace", stats_missing={}, stats_mode={}, print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to impute e.g., ["col1","col2"].
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
    :param method_type: "median", "mean" (valid only for for numerical columns attributes).
                         Mode is only option for categorical columns.
    :param pre_existing_model: Boolean argument – True or False. True if imputation model exists already, False otherwise.
    :param model_path: If pre_existing_model is True, this argument is path for referring the pre-saved model.
                       If pre_existing_model is False, this argument can be used for saving the model.
                       Default "NA" means there is neither pre-existing model nor there is a need to save one.
    :param output_mode: "replace", "append".
                         “replace” option replaces original columns with transformed column. “append” option append transformed
                         column to the input dataset with a postfix "_imputed" e.g. column X is appended as X_imputed.
    :param stats_missing: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                          to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
                          missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param stats_mode: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                       to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
                       mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: Imputed Dataframe
    """
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
    else:
        missing_df = read_dataset(spark, **stats_missing).select('attribute', 'missing_count', 'missing_pct')

    missing_cols = missing_df.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()

    if str(pre_existing_model).lower() == 'true':
        pre_existing_model = True
    elif str(pre_existing_model).lower() == 'false':
        pre_existing_model = False
    else:
        raise TypeError('Non-Boolean input for pre_existing_model')

    if (len(missing_cols) == 0) & (pre_existing_model == False) & (model_path == "NA"):
        return idf

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)
    if list_of_cols == 'all':
        list_of_cols = num_cols + cat_cols
    if list_of_cols == "missing":
        list_of_cols = [x for x in missing_cols if x in num_cols + cat_cols]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if len(list_of_cols) == 0:
        warnings.warn("No Imputation performed- No column(s) to impute")
        return idf
    if any(x not in num_cols + cat_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if method_type not in ('mean', 'median'):
        raise TypeError('Invalid input for method_type')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))

    odf = idf
    if len(num_cols) > 0:
        # Checking for Integer/Decimal Type Columns & Converting them into Float/Double Type
        recast_cols = []
        recast_type = []
        for i in num_cols:
            if get_dtype(idf, i) not in ('float', 'double'):
                odf = odf.withColumn(i, F.col(i).cast(T.DoubleType()))
                recast_cols.append(i + "_imputed")
                recast_type.append(get_dtype(idf, i))

        # Building new imputer model or uploading the existing model
        if pre_existing_model == True:
            imputerModel = ImputerModel.load(model_path + "/imputation_MMM/num_imputer-model")
        else:
            imputer = Imputer(strategy=method_type, inputCols=num_cols,
                              outputCols=[(e + "_imputed") for e in num_cols])
            imputerModel = imputer.fit(odf)

        # Applying model
        # odf = recast_column(imputerModel.transform(odf), recast_cols, recast_type)
        odf = imputerModel.transform(odf)
        for i, j in zip(recast_cols, recast_type):
            odf = odf.withColumn(i, F.col(i).cast(j))

        # Saving model if required
        if (pre_existing_model == False) & (model_path != "NA"):
            imputerModel.write().overwrite().save(model_path + "/imputation_MMM/num_imputer-model")

    if len(cat_cols) > 0:
        if pre_existing_model:
            df_model = spark.read.csv(model_path + "/imputation_MMM/cat_imputer", header=True, inferSchema=True)
            parameters = []
            for i in cat_cols:
                mapped_value = \
                    df_model.where(F.col('attribute') == i).select('parameters').rdd.flatMap(lambda x: x).collect()[0]
                parameters.append(mapped_value)
        else:
            if stats_mode == {}:
                parameters = [str((idf.select(i).dropna().groupby(i).count().orderBy("count", ascending=False).first()
                                   or [None])[0]) for i in cat_cols]
            else:
                mode_df = read_dataset(spark, **stats_mode).replace('None', None)
                parameters = [mode_df.where(F.col('attribute') == i).select('mode').rdd.flatMap(list).collect()[0] for i
                              in cat_cols]

        for index, i in enumerate(cat_cols):
            odf = odf.withColumn(i + "_imputed", F.when(F.col(i).isNull(), parameters[index]).otherwise(F.col(i)))

        # Saving model File if required
        if (pre_existing_model == False) & (model_path != "NA"):
            df_model = spark.createDataFrame(zip(cat_cols, parameters), schema=['attribute', 'parameters'])
            df_model.repartition(1).write.csv(model_path + "/imputation_MMM/cat_imputer", header=True, mode='overwrite')

    for i in (num_cols + cat_cols):
        if i not in missing_cols:
            odf = odf.drop(i + "_imputed")
        elif output_mode == 'replace':
            odf = odf.drop(i).withColumnRenamed(i + "_imputed", i)

    if print_impact:
        if output_mode == 'replace':
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(spark, odf, list_of_cols) \
                      .select('attribute', F.col("missing_count").alias("missingCount_after")), 'attribute', 'inner')
        else:
            output_cols = [(i + "_imputed") for i in [e for e in (num_cols + cat_cols) if e in missing_cols]]
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(spark, odf, output_cols) \
                      .withColumnRenamed('attribute', 'attribute_after') \
                      .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")) \
                      .drop('missing_pct'), 'attribute', 'inner')
        odf_print.show(len(list_of_cols))
    return odf


def imputation_sklearn(spark, idf, list_of_cols="missing", drop_cols=[], method_type="KNN", max_size=500000, 
                       emr_mode=False, pre_existing_model=False, model_path="NA", output_mode="replace", 
                       stats_missing={}, print_impact=False):
    
    '''
    idf: Pyspark Dataframe
    method_type: KNN, RBM, regression
    list_of_cols: all, missing (i.e. all feautures with missing values), list of columns (in list format or string separated by |) 
                all is better strategy when training has no missing data but testing/prediction data may have.
                Categorical features are discarded else transform them before this process.
    id_col, label_col: Excluding ID & Label columns from imputation
    max_size: Maximum rows for training the imputer
    pre_existing_model: True if imputer exists already, False Otherwise. 
    model_path: If pre_existing_model is True, this argument is path for imputation model. 
                  If pre_existing_model is False, this argument can be used for saving the imputation model. 
                  Default "NA" means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Imputed Dataframe
    '''
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
    else:
        missing_df = read_dataset(spark, **stats_missing).select('attribute', 'missing_count', 'missing_pct')

    missing_cols = missing_df.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()

    if str(pre_existing_model).lower() == 'true':
        pre_existing_model = True
    elif str(pre_existing_model).lower() == 'false':
        pre_existing_model = False
    else:
        raise TypeError('Non-Boolean input for pre_existing_model')

    if (len(missing_cols) == 0) & (pre_existing_model == False) & (model_path == "NA"):
        return idf
    
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if list_of_cols == "missing":
        list_of_cols = [x for x in missing_cols if x in num_cols]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = sorted(list(set([e for e in list_of_cols if e not in drop_cols])))

    if len(list_of_cols) == 0:
        warnings.warn("No Action Performed - sklearn Imputation")
        return idf
    # Pending: should we allow input column with wrong dtype? (applicable to all functions - currently not allowed)
    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
        
    if method_type not in ('KNN', 'regression'):
        raise TypeError('Invalid input for method_type')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    include_cols = num_cols
    exclude_cols = [e for e in idf.columns if e not in num_cols]
    
    if pre_existing_model:
        if emr_mode:
            bash_cmd = "aws s3 cp " + model_path + "/imputation_sklearn.sav"
            output = subprocess.check_output(['bash', '-c', bash_cmd])
            #imputer = joblib.load("imputation_sklearn.sav")
            imputer = pickle.load(open("imputation_sklearn.sav", 'rb'))
        else: 
            #imputer = joblib.load(model_path + "/imputation_sklearn.sav")
            imputer = pickle.load(open(model_path + "/imputation_sklearn.sav", 'rb'))
        idf_rest = idf
    else:
        sample_ratio = min(1.0,float(max_size)/idf.count())
        # Note: subtract removes duplicated rows also so assigned an ID and drop it later (o/w odf may have less num of rows)
        idf = idf.withColumn('id', F.monotonically_increasing_id())
        idf_model = idf.sample(False, sample_ratio, 0)
        idf_rest = idf.subtract(idf_model)
        idf, idf_model, idf_rest = idf.drop('id'), idf_model.drop('id'), idf_rest.drop('id')
        idf_pd = idf_model.toPandas()
        
        X = idf_pd[include_cols]
        # X = idf_pd.drop(exclude_cols,axis=1)
        # Note: in this case, column order in odf will be wrong if list_of_cols does not follow the original column
        # Y = idf_pd[exclude_cols]

        if method_type == 'KNN':
            imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(X)
        if method_type == 'regression':
            imputer = IterativeImputer()
            imputer.fit(X)
        # Note: removed due to version conflict
        # Imputer (used by boltzmannclean in RBM) was deprecated since sklearn v0.22 but KNNImputer was first introduced in version 0.22
        # if method_type == 'RBM':
        #     import boltzmannclean
        #     imputer = boltzmannclean.train_rbm(X.values, tune_hyperparameters=False)

        if (pre_existing_model == False) & (model_path != "NA"):
            if emr_mode:
                #joblib.dump(imputer, "imputation_sklearn.sav")
                pickle.dump(imputer, open("imputation_sklearn.sav", 'wb'))
                bash_cmd = "aws s3 cp imputation_sklearn.sav " + model_path + "/imputation_sklearn.sav"
                output = subprocess.check_output(['bash', '-c', bash_cmd])
            else:
                #joblib.dump(imputer, model_path + "/imputation_sklearn.sav")
                local_path = model_path  + "/imputation_sklearn.sav"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                #joblib.dump(imputer, local_path)
                pickle.dump(imputer, open(local_path, 'wb'))
        
        pred = imputer.transform(X)
        output = pd.concat([pd.Series(list(pred)),idf_pd], axis=1)
        output.rename(columns={0:'features'}, inplace=True)
        output.features = output.features.map(lambda x: [float(e) for e in x])
        odf_model = spark.createDataFrame(output)
        for index,i in enumerate(include_cols):
            modify_col = ((i + "_imputed") if (output_mode == "append") else i)
            odf_model = odf_model.withColumn(modify_col, F.col('features')[index])
        odf_model = odf_model.drop('features')
        for i in odf_model.columns:
            odf_model = odf_model.withColumn(i,F.when(F.isnan(F.col(i)),None).otherwise(F.col(i)))

    if idf_rest.count() > 0:
        @F.pandas_udf(returnType=T.ArrayType(T.DoubleType()))
        def prediction(*cols):
            X = pd.concat(cols, axis=1)
            return pd.Series(row.tolist() for row in imputer.transform(X))
        odf_rest = idf_rest.withColumn('features',prediction(*include_cols))
        for index,i in enumerate(include_cols):
            modify_col = ((i + "_imputed") if (output_mode == "append") else i)
            odf_rest = odf_rest.withColumn(modify_col, F.col('features')[index])
        odf_rest = odf_rest.drop('features')
        
    if pre_existing_model:
        odf = odf_rest
    elif idf_rest.count() == 0:
        odf = odf_model
    else:
        odf = odf_model.union(odf_rest.select(odf_model.columns))
    
    
    for i in include_cols:
        if (i not in missing_cols) & (output_mode == 'append'):
            odf = odf.drop(i+"_imputed")
    
    if print_impact:
        if output_mode == 'replace':
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(spark, odf, list_of_cols) \
                      .select('attribute', F.col("missing_count").alias("missingCount_after")), 'attribute', 'inner')
        else:
            output_cols = [(i + "_imputed") for i in [e for e in num_cols if e in missing_cols]]
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(spark, odf, output_cols) \
                      .withColumnRenamed('attribute', 'attribute_after') \
                      .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")) \
                      .drop('missing_pct'), 'attribute', 'inner')
        odf_print.show(len(list_of_cols))
    return odf


def imputation_matrixFactorization(spark, idf, list_of_cols="missing", drop_cols=[], id_col="id", output_mode='replace',
                                   stats_missing={}, print_impact=False):
    '''
    idf: Pyspark Dataframe
    list_of_cols: all, missing (i.e. all feautures with missing values), list of columns (in list format or string separated by |) 
                all is better strategy when training has no missing data but testing/prediction data may have.
                Categorical features are discarded else transform them before this process.
    id_col, label_col: Excluding ID & Label columns from imputation
    output_mode: replace or append
    return: Imputed Dataframe
    '''
        
    if output_mode not in ('replace','append'):
        raise TypeError('Invalid input for output_mode')
    
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
    else:
        missing_df = read_dataset(spark, **stats_missing).select('attribute', 'missing_count', 'missing_pct')
    missing_cols = missing_df.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if list_of_cols == "missing":
        list_of_cols = [x for x in missing_cols if x in num_cols]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if (e not in drop_cols) & (e != id_col)]))

    if len(list_of_cols) == 0:
        warnings.warn("No Action Performed - Matrix Factorization Imputation")
        return idf
    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    include_cols = num_cols
    exclude_cols = [e for e in idf.columns if e not in num_cols]
     
    #Create map<key: value>
    key_and_val = F.create_map(list(chain.from_iterable([[F.lit(c), F.col(c)] for c in include_cols])))
    df_flatten = idf.select(id_col, F.explode(key_and_val)).withColumn("key", F.concat(F.col('key'), F.lit("_imputed")))

    #Indexing ID & Key/Feature Column
    id_type = get_dtype(idf,id_col)
    if id_type == 'string':
        id_indexer = StringIndexer().setInputCol(id_col).setOutputCol("IDLabel")
        id_indexer_model = id_indexer.fit(df_flatten)
        df_flatten = id_indexer_model.transform(df_flatten).drop(id_col)
    else:
        df_flatten = df_flatten.withColumnRenamed(id_col,"IDLabel")
    
    indexer = StringIndexer().setInputCol("key").setOutputCol("keyLabel")
    indexer_model = indexer.fit(df_flatten)
    df_encoded = indexer_model.transform(df_flatten).drop('key')
    df_model = df_encoded.where(F.col('value').isNotNull())
    df_test = df_encoded.where(F.col('value').isNull())
    if df_model.select('IDLabel').distinct().count() < df_encoded.select('IDLabel').distinct().count():
        warnings.warn("The returned odf may not be fully imputed because values for all list_of_cols are null for some IDs")
    
    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=20, regParam=0.01, userCol="IDLabel", itemCol="keyLabel", ratingCol="value",
              coldStartStrategy="drop")
    model = als.fit(df_model)
    
    df_pred = model.transform(df_test).drop('value').withColumnRenamed("prediction","value")
    df_encoded_pred = df_model.union(df_pred.select(df_model.columns))
    if id_type == 'string':
        IDlabelReverse = IndexToString().setInputCol("IDLabel").setOutputCol(id_col)
        df_encoded_pred = IDlabelReverse.transform(df_encoded_pred)
    else:
        df_encoded_pred = df_encoded_pred.withColumnRenamed("IDLabel", id_col)
        
    keylabelReverse = IndexToString().setInputCol("keyLabel").setOutputCol("key")
    odf_imputed = keylabelReverse.transform(df_encoded_pred).groupBy(id_col).pivot('key').agg(F.first('value'))                   .select([id_col]+[(i+"_imputed") for i in include_cols if i in missing_cols])
        
    odf = idf.join(odf_imputed,id_col,'left_outer')
    
    for i in num_cols:
        if i not in missing_cols:
            odf = odf.drop(i + "_imputed")
        elif output_mode == 'replace':
            odf = odf.drop(i).withColumnRenamed(i + "_imputed", i)
    
    if print_impact:
        if output_mode == 'replace':
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(spark, odf, list_of_cols) \
                      .select('attribute', F.col("missing_count").alias("missingCount_after")), 'attribute', 'inner')
        else:
            output_cols = [(i + "_imputed") for i in [e for e in num_cols if e in missing_cols]]
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(spark, odf, output_cols) \
                      .withColumnRenamed('attribute', 'attribute_after') \
                      .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")) \
                      .drop('missing_pct'), 'attribute', 'inner')
        odf_print.show(len(list_of_cols))
    return odf


def imputation_custom(spark, idf, list_of_cols="missing", list_of_fills=None, method_type='row_removal', 
                      output_mode="replace", stats_missing={}, print_impact=False):
    # Note: remove input variable drop_cols - may cause confusion in list_of_fills, list_of_cols mapping
    '''
    idf: Pyspark Dataframe
    method: fill_constant or row_removal
    list_of_cols: all, list of columns (in list format or string separated by |)
    list_of_fills: list of constants to be filled for correponding columns in list_of_cols
    output_mode: replace or append
    return: Imputed Dataframe
    '''
    
    if method_type not in ('fill_constant','row_removal'):
        raise TypeError('Invalid input for method_type')
    if output_mode not in ('replace','append'):
        raise TypeError('Invalid input for output_mode')
    
    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
    else:
        missing_df = read_dataset(spark, **stats_missing).select('attribute', 'missing_count', 'missing_pct')
    missing_cols = missing_df.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if list_of_cols == "missing":
        list_of_cols = missing_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    # if isinstance(drop_cols, str):
    #     drop_cols = [x.strip() for x in drop_cols.split('|')]
    # # Note: Function set changes the order of columns
    # list_of_cols = list([e for e in list_of_cols if e not in drop_cols])

    if len(list_of_cols) == 0:
        warnings.warn("No Action Performed - custom Imputation")
        return idf
    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))
    
    if method_type == 'row_removal':
        odf = idf.dropna(subset=list_of_cols)
        
        if print_impact:
            print("Before Count: " + str(idf.count()))
            print("After Count: " + str(odf.count()))
                
    else:
        if isinstance(list_of_fills, str):
            list_of_fills = [x.strip() for x in list_of_fills.split('|')]
        # Allow single value input for numerical fill
        elif isinstance(list_of_fills, int):
            list_of_fills = [list_of_fills] * len(list_of_cols)
        if len(list_of_fills) != len(list_of_cols):
            raise TypeError('Invalid input for list_of_fills')

        odf = idf
        for i in list(zip(list_of_cols,list_of_fills)):
            if i[0] in missing_cols:
                modify_col = ((i[0] + "_imputed") if (output_mode == "append") else i[0])
                if i[0] in num_cols:
                    odf = odf.withColumn(modify_col, F.when(F.col(i[0]).isNull(), i[1]).otherwise(F.col(i[0])))
                if i[0] in cat_cols:
                    odf = odf.withColumn(modify_col, F.when(F.col(i[0]).isNull(), i[1]).otherwise(F.col(i[0])))

        if print_impact:
            if output_mode == 'replace':
                odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                    .join(missingCount_computation(spark, odf, list_of_cols) \
                        .select('attribute', F.col("missing_count").alias("missingCount_after")), 'attribute', 'inner')
            else:
                output_cols = [(i + "_imputed") for i in [e for e in (num_cols + cat_cols) if e in missing_cols]]
                odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                    .join(missingCount_computation(spark, odf, output_cols) \
                        .withColumnRenamed('attribute', 'attribute_after') \
                        .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")) \
                        .drop('missing_pct'), 'attribute', 'inner')
            odf_print.show(len(list_of_cols))
    return odf


def imputation_comparison(spark, idf, list_of_cols="missing", drop_cols=[], id_col="id", null_pct=0.1, 
                          stats_missing={}, print_impact=True):
    '''
    idf: Pyspark Dataframe
    list_of_cols: all, missing (i.e. all feautures with missing values), list of columns (in list format or string separated by |) 
                all is better strategy when training has no missing data but testing/prediction data may have.
                Categorical features are discarded else transform them before this process.
    id_col, label_col: Excluding ID & Label columns from imputation
    null_pct: %row converted into null for test dataset (per col)
    return: Name of Imputation Technique
    ''' 

    if stats_missing == {}:
        missing_df = missingCount_computation(spark, idf)
        # Note: save generated stats_missing result which can be used in method 1~5
        # Pending: path to save the result; delete file after computation?
        missing_df.write.parquet("NA/missing", mode='overwrite')
        stats_missing = {"file_path": "NA/missing", "file_type": "parquet"}
    else:
        missing_df = read_dataset(spark, **stats_missing).select('attribute', 'missing_count', 'missing_pct')
    missing_cols = missing_df.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()
    
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    elif list_of_cols == "missing":
        list_of_cols = [x for x in missing_cols if x in num_cols]
    elif isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|') if x.strip() in idf.columns]
    else:
        list_of_cols = [e for e in list_of_cols if e in idf.columns] 
    list_of_cols = list(set([e for e in list_of_cols if (e not in drop_cols) & (e !=id_col)]))

    if len(list_of_cols) == 0:
        warnings.warn("No Action Performed - Imputation_Comparison")
        return None
    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols
    
    idf_test = idf.dropna().withColumn('index', F.monotonically_increasing_id()).withColumn("index", F.row_number().over(Window.orderBy("index")))
    null_count = int(null_pct*idf_test.count())
    idf_null = idf_test
    for i in list_of_cols:
        null_index = random.sample(range(idf_test.count()), null_count)
        idf_null = idf_null.withColumn(i, F.when(F.col('index').isin(null_index), None).otherwise(F.col(i)))

    idf_null.write.parquet("intermediate_data/imputation_comparison/test_dataset", mode='overwrite')
    idf_null = spark.read.parquet("intermediate_data/imputation_comparison/test_dataset")

    method1 = imputation_MMM(spark, idf_null, list_of_cols=list_of_cols, method_type="mean", stats_missing=stats_missing)
    method2 = imputation_MMM(spark, idf_null, list_of_cols=list_of_cols, method_type="median", stats_missing=stats_missing)
    method3 = imputation_sklearn(spark, idf_null, list_of_cols=list_of_cols, method_type="KNN", stats_missing=stats_missing)
    method4 = imputation_sklearn(spark, idf_null, list_of_cols=list_of_cols, method_type='regression', stats_missing=stats_missing)
    method5 = imputation_matrixFactorization(spark, idf_null, id_col = id_col, list_of_cols=list_of_cols, stats_missing=stats_missing)
    #method6 = imputation_sklearn(idf_null, method_type='RBM', list_of_cols=list_of_cols, id_col = id_col, label_col=label_col)

    rmse_all = []
    method_all = ['MMM-mean','MMM-median','KNN','regression','matrix_factorization'] 
    for index, method in enumerate([method1,method2,method3,method4,method5]):
        rmse=0
        for i in list_of_cols:
            idf_joined = idf_test.select('index',F.col(i).alias('val')).join(method.select('index',F.col(i).alias('pred')),'index','left_outer').dropna()
            idf_joined = recast_column(idf=idf_joined, list_of_cols=['val','pred'], list_of_dtypes=['double', 'double'])
            i_rmse = RegressionEvaluator(metricName="rmse", labelCol="val",predictionCol="pred").evaluate(idf_joined)
            rmse += i_rmse
        rmse_all.append(rmse)
        
    min_index= rmse_all.index(np.min(rmse_all))
    best_method =method_all[min_index]
    
    if print_impact:
        print(list(zip(method_all, rmse_all)))
        print("Best Imputation Method: ", best_method)
    return best_method


def autoencoders_latentFeatures(spark, idf, list_of_cols="all", drop_cols=[], reduction_params=0.5, max_size=500000, 
                                epochs=100, batch_size=256, emr_mode=False, pre_existing_model=False, model_path="NA",
                                standardization_pre_existing_model=False, standardization_model_path="NA",
                                output_mode="replace", print_impact=False, plot_learning_curves=True):
    '''
    idf: Input Dataframe
    list_of_cols: all (numerical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    pre_existing_model: True if model files exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  Default NA means there is neither pre_existing_model nor there is a need to save one.
    output_mode: append or replace
    return: Dataframe
    '''

    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all': 
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    
    if (len(list_of_cols) == 0) | (any(x not in num_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols
    
    n_inputs = len(list_of_cols)
    if reduction_params < 1:
        n_bottleneck = int(reduction_params*n_inputs)
    else:
        n_bottleneck = int(reduction_params)
    
    # Note: standardize input columns before training. Otherwise it could be hard to converge
    idf_standardized = z_standardization(spark, idf, list_of_cols=list_of_cols, pre_existing_model=standardization_pre_existing_model,
                                         model_path=standardization_model_path, output_mode='append')
    list_of_cols_scaled = [i+'_scaled' for i in list_of_cols]
    
    if pre_existing_model:
        if emr_mode:
            bash_cmd = "aws s3 cp " + model_path + "/autoencoders_latentFeatures/encoder.h5"
            output = subprocess.check_output(['bash', '-c', bash_cmd])
            bash_cmd = "aws s3 cp " + model_path + "/autoencoders_latentFeatures/model.h5"
            output = subprocess.check_output(['bash', '-c', bash_cmd])
            encoder = load_model("encoder.h5")
            model = load_model("model.h5")
        else: 
            encoder = load_model(model_path + "/autoencoders_latentFeatures/encoder.h5")
            model = load_model(model_path + "/autoencoders_latentFeatures/model.h5")
    else:
        idf_valid = idf_standardized.select(list_of_cols_scaled).dropna()
        idf_model = idf_valid.sample(False, min(1.0, float(max_size)/idf_valid.count()), 0)
        
        idf_train = idf_model.sample(False, 0.8, 0)
        idf_test = idf_model.subtract(idf_train)
        X_train = idf_train.toPandas()
        X_test = idf_test.toPandas()
             
        visible = Input(shape=(n_inputs,))
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        bottleneck = Dense(n_bottleneck)(e)
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        d = Dense(n_inputs*2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        output = Dense(n_inputs, activation='linear')(d)

        # autoencoder model
        model = Model(inputs=visible, outputs=output)
        encoder = Model(inputs=visible, outputs=bottleneck)
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train, X_train, epochs=int(epochs), batch_size=int(batch_size), verbose=2, 
                  validation_data=(X_test,X_test))
        if plot_learning_curves:
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()
 
        # Saving model if required
        if (pre_existing_model == False) & (model_path != "NA"):
            if emr_mode:
                encoder.save("encoder.h5")
                model.save("model.h5")
                bash_cmd = "aws s3 cp encoder.h5 " + model_path + "/autoencoders_latentFeatures/encoder.h5"
                output = subprocess.check_output(['bash', '-c', bash_cmd])
                bash_cmd = "aws s3 cp model.h5 " + model_path + "/autoencoders_latentFeatures/model.h5"
                output = subprocess.check_output(['bash', '-c', bash_cmd])
            else:
                if not os.path.exists(model_path + "/autoencoders_latentFeatures/"):
                    os.makedirs(model_path + "/autoencoders_latentFeatures/")
                encoder.save(model_path + "/autoencoders_latentFeatures/encoder.h5")
                model.save(model_path + "/autoencoders_latentFeatures/model.h5")
        
        #print(model.predict(X_test))
    
    class ModelWrapperPickable:
        def __init__(self, model):
            self.model = model

        def __getstate__(self):
            model_str = ''
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                tensorflow.keras.models.save_model(self.model, fd.name, overwrite=True)
                model_str = fd.read()
            d = { 'model_str': model_str }
            return d

        def __setstate__(self, state):
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(state['model_str'])
                fd.flush()
                self.model = tensorflow.keras.models.load_model(fd.name)

    model_wrapper= ModelWrapperPickable(encoder)
    #print(model_wrapper.model.predict(X_test))

    def compute_output_pandas_udf(model_wrapper):
        '''Spark pandas udf for model prediction.'''
        @F.pandas_udf(returnType=T.ArrayType(T.DoubleType()))
        def predict_pandas_udf(*cols):
            X = pd.concat(cols, axis=1)
            return pd.Series(row.tolist() for row in model_wrapper.model.predict(X))
        return predict_pandas_udf
    
    """
    # Pending: 2 configs need to be added to anovos.shared.spark configs variabl:
        - 'spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT': '1', 
        - 'spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT': '1'
    """
    if output_mode=="append":
        odf = idf_standardized.withColumn('predicted_output', compute_output_pandas_udf(model_wrapper)(*list_of_cols_scaled))\
            .select(idf.columns+[F.col("predicted_output")[j].alias("latent_" + str(j)) for j in range(0, n_bottleneck)])
    else:
        odf = idf_standardized.withColumn('predicted_output', compute_output_pandas_udf(model_wrapper)(*list_of_cols_scaled))\
            .select([e for e in idf.columns if e not in list_of_cols]+
                    [F.col("predicted_output")[j].alias("latent_" + str(j)) for j in range(0, n_bottleneck)])
    
    # Pending: print impact for generated latent features only or all features?
    if print_impact:
        #odf.describe().show()
        output_cols = ["latent_" + str(j) for j in range(0, n_bottleneck)]
        odf.select(output_cols).describe().show(10, False)
    
    return odf


def PCA_latentFeatures(spark, idf, list_of_cols="all", drop_cols=[], explained_variance_cutoff=0.95, 
                       pre_existing_model=False, model_path="NA", 
                       standardization_pre_existing_model=False, standardization_model_path="NA",
                       output_mode="replace", print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: all (numerical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    pre_existing_model: True if model files exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  Default NA means there is neither pre_existing_model nor there is a need to save one.
    output_mode: append or replace
    return: Dataframe
    '''
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all': 
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    
    if (len(list_of_cols) == 0) | (any(x not in num_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols

    # Note: standardize input columns before training. Otherwise it could be hard to converge
    idf_standardized = z_standardization(spark, idf, list_of_cols=list_of_cols, pre_existing_model=standardization_pre_existing_model,
                                         model_path=standardization_model_path, output_mode='append')
    list_of_cols_scaled = [i+'_scaled' for i in list_of_cols]
    # Note: use the assembled data without nan values for PCA model training
    assembler = VectorAssembler(inputCols=list_of_cols_scaled, outputCol="features", handleInvalid="skip")
    assembled_data = assembler.transform(idf_standardized)
    # Note: use the original assembled data to generate odf
    assembler_orignial = VectorAssembler(inputCols=list_of_cols_scaled, outputCol="features", handleInvalid="keep")
    assembled_data_orignial = assembler_orignial.transform(idf_standardized)

    if pre_existing_model == True:
        pca = PCA.load(model_path+"/PCA_latentFeatures/pca_path")
        pcaModel = PCAModel.load(model_path+"/PCA_latentFeatures/pcaModel_path")
        n = pca.getK()
    else:
        pca = PCA(k=len(list_of_cols_scaled), inputCol='features', outputCol='features_pca')
        pcaModel = pca.fit(assembled_data)
        explained_variance = 0
        for n in range(1,len(list_of_cols)+1):
            explained_variance += pcaModel.explainedVariance[n-1]
            if explained_variance > explained_variance_cutoff:
                break
        
        pca = PCA(k=n, inputCol='features', outputCol='features_pca')
        pcaModel = pca.fit(assembled_data)
        # Saving model if required
        if (pre_existing_model == False) & (model_path != "NA"):
            pcaModel.write().overwrite().save(model_path+"/PCA_latentFeatures/pcaModel_path")
            pca.write().overwrite().save(model_path+"/PCA_latentFeatures/pca_path")
    
    def vector_to_array(v):
            return v.toArray().tolist()
    f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.FloatType()))
    
    if output_mode=="append":
        odf = pcaModel.transform(assembled_data_orignial)\
            .withColumn("features_pca_array", f_vector_to_array('features_pca'))\
            .select(idf.columns+
                    [F.col("features_pca_array")[j].alias("latent_" + str(j)) for j in range(0, n)])\
            .replace(float('nan'), None, subset=["latent_" + str(j) for j in range(0, n)])
    else:
        odf = pcaModel.transform(assembled_data_orignial)\
            .withColumn("features_pca_array", f_vector_to_array('features_pca'))\
            .select([e for e in idf.columns if e not in list_of_cols]+
                    [F.col("features_pca_array")[j].alias("latent_" + str(j)) for j in range(0, n)])\
            .replace(float('nan'), None, subset=["latent_" + str(j) for j in range(0, n)])

    if print_impact:
        print("Explained Variance: ", round(np.sum(pcaModel.explainedVariance[0:n]),4))
        odf.select([e for e in odf.columns if e.startswith('latent_')]).describe().show(10, False)
        
    return odf


def feature_transformation(spark, idf, list_of_cols="all", drop_cols=[], method_type="square_root", boxcox_lambda=None,
                           output_mode="replace",print_impact=False):
    # Note: currently using square_root as the default value for method_type because log & box_cox require positive data
    '''
    idf: Input Dataframe
    list_of_cols: all (numerical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    method_type: "log" or "square_root" or "box_cox"
    output_mode: append or replace
    return: Dataframe
    '''
    
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all': 
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    
    if (len(list_of_cols) == 0) | (any(x not in num_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols
    odf = idf
    col_mins = idf.select([F.min(i) for i in list_of_cols])
    if method_type in ["log", "box_cox"]:
        # Note: raise error when there are non positive value in any column (same as scipy.stats.boxcox for box_cox)
        if any([i <= 0 for i in col_mins.rdd.flatMap(lambda x: x).collect()]):
            col_mins.show(1, False)
            raise ValueError('Data must be positive')
    elif method_type == "square_root":
        if any([i < 0 for i in col_mins.rdd.flatMap(lambda x: x).collect()]):
            col_mins.show(1, False)
            raise ValueError('Data must be non-negative')
    else:
        raise TypeError('Invalid input method_type')

    if method_type == "log":
        for i in list_of_cols:
            modify_col = (i + "_log") if output_mode == 'append' else i
            odf = odf.withColumn(modify_col, F.log(F.col(i)))
            
        if print_impact:
            print("Before:")
            # Note: previously shows idf.select(list_of_cols).show(5)
            idf.select(list_of_cols).describe().show(10, False)
            print("After:")
            if output_mode == 'replace':
                output_cols = list_of_cols
            else:
                output_cols = [(i+"_log") for i in list_of_cols]
            odf.select(output_cols).describe().show(10, False)
    
    if method_type == "square_root":
        for i in list_of_cols:
            modify_col = (i + "_sqrt") if output_mode == 'append' else i
            odf = odf.withColumn(modify_col, F.sqrt(F.col(i)))
        
        if print_impact:
            print("Before:")
            idf.select(list_of_cols).describe().show(10, False)
            print("After:")
            if output_mode == 'replace':
                output_cols = list_of_cols
            else:
                output_cols = [(i+"_sqrt") for i in list_of_cols]
            odf.select(output_cols).describe().show(10, False)
    
    # Note: current logic for box_cox transformation (might need further discussion)
    # if boxcox_lambda is not None, directly use the input value for transformation
    # else, search for the bese lambda for each column and apply the transformation
    if method_type == "box_cox":
        if boxcox_lambda is not None:
            if isinstance(boxcox_lambda, (list, tuple)):
                if len(boxcox_lambda) != len(list_of_cols):
                    raise TypeError('Invalid input for boxcox_lambda')
                elif not all([isinstance(l, (float, int)) for l in boxcox_lambda]):
                    raise TypeError('Invalid input for boxcox_lambda')
                else:
                    boxcox_lambda_list = list(boxcox_lambda)

            elif isinstance(boxcox_lambda, (float, int)):
                boxcox_lambda_list = [boxcox_lambda] * len(list_of_cols)
            else:
                raise TypeError('Invalid input for boxcox_lambda')

        else:
            boxcox_lambda_list = []
            for i in list_of_cols:
                # Note: changed the order a bit so that smaller transformation is preferred (can further change)
                # (Sometimes kolmogorovSmirnovTest gives the same extremely small pVal for all lambdas tested)
                lambdaVal = [1,-1,0.5,-0.5,2,-2,0.25,-0.25,3,-3,4,-4,5,-5]
                # lambdaVal = [-5,-4,-3,-2,-1,-0.5,-0.25,0.25,0.5,1,2,3,4,5]
                best_pVal = 0
                for j in lambdaVal:
                    pVal = Statistics.kolmogorovSmirnovTest(odf.select(F.pow(F.col(i),j)).rdd.flatMap(lambda x:x), "norm").pValue
                    if pVal > best_pVal:
                        best_pVal = pVal
                        best_lambdaVal = j

                pVal = Statistics.kolmogorovSmirnovTest(odf.select(F.log(F.col(i))).rdd.flatMap(lambda x:x), "norm").pValue
                if pVal > best_pVal:
                    best_pVal = pVal
                    best_lambdaVal = 0
                
                boxcox_lambda_list.append(best_lambdaVal)
        
        output_cols = []
        for i, curr_lambdaVal in zip(list_of_cols, boxcox_lambda_list):
            # Pending: previously (i + "_log") was used as the new column name for 0 lambdaVal. 
            # But I feel it might make tracking column names a bit hard? 
            if curr_lambdaVal != 1:
                modify_col = (i + "_bxcx_" + str(curr_lambdaVal)) if output_mode == 'append' else i
                output_cols.append(modify_col)
                if curr_lambdaVal == 0:
                    odf = odf.withColumn(modify_col, F.log(F.col(i)))
                else:
                    odf = odf.withColumn(modify_col, F.pow(F.col(i), curr_lambdaVal))
        if len(output_cols)==0:
            warnings.warn("lambdaVal for all columns are 1 so no transformation is performed so idf is returned")
            return idf
        
        if print_impact:
            print("Transformed Columns: ", list_of_cols)
            print("Best BoxCox Parameter(s): ", boxcox_lambda_list)
            print("Before:")
            # Note: added skewness to summary statistics 
            idf.select(list_of_cols).describe()\
                .unionByName(idf.select([F.skewness(i).alias(i) for i in list_of_cols])\
                                .withColumn('summary', F.lit('skewness')))\
                .show()
            print("After:")
            if output_mode == 'replace':
                odf.select(list_of_cols).describe()\
                    .unionByName(odf.select([F.skewness(i).alias(i) for i in list_of_cols])\
                                    .withColumn('summary', F.lit('skewness'))).show()
            else:
                output_cols = [("`" + i + "`") for i in output_cols]
                odf.select(output_cols).describe()\
                    .unionByName(odf.select([F.skewness(i).alias(i[1:-1]) for i in output_cols])\
                                    .withColumn('summary', F.lit('skewness'))).show()
    
    return odf


def feature_autoLearn(spark, idf, list_of_cols="all", drop_cols=[], label_col='label', event_label=1, 
                      IG_threshold=0, dist_corr_threshold=0.7, stability_threshold=0.8, max_size=500000,
                      emr_mode=False, pre_existing_model=False, model_path="NA", output_mode="replace",
                      print_impact=False):
    # Note: added variables emr_mode, pre_existing_model, model_path
    '''
    idf: Input Dataframe
    list_of_cols: all (numerical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    IG_threshold: Information Gain Value Threshold
    dist_corr_threshold: Distance Correlation Threshold
    stability_threshold= Stability Score Threshold
    output_mode: append or replace
    return: Dataframe
    '''
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all': 
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if (e not in drop_cols) & (e != label_col)]))
    
    if (len(list_of_cols) == 0) | (any(x not in num_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols = attributeType_segregation(idf.select(list_of_cols))[0]
    list_of_cols = num_cols

    if output_mode=="replace":
        fixed_cols = [e for e in idf.columns if e not in list_of_cols]
    else:
        fixed_cols = idf.columns

    # Note: the predict function of sklearn does not support null input so error might be raised later if list_of_cols contains null.
    # Pending: should we support null values?
    if (idf.count()!=idf.dropna(subset=list_of_cols).count()):
        warnings.warn("list_of_cols should not contain None values, otherwise it may cause error.")

    idf = idf.withColumn(label_col+'_converted', F.when(F.col(label_col) == event_label, 1).otherwise(0))

    models, col_pairs = [], []
    if pre_existing_model:
        if emr_mode:
            subprocess.check_output(['bash', '-c', 'rm -r feature_autoLearn; mkdir feature_autoLearn'])
            bash_cmd = "aws s3 cp " + model_path + "/feature_autoLearn feature_autoLearn --recursive"
            subprocess.check_output(['bash', '-c', bash_cmd])
            model_names = [f for f in os.listdir() if '.sav' in f]
            for m in model_names:
                models.append(pickle.load(open(m, 'rb')))
                col_pairs.append(m.split('.sav')[0].split("__and__"))

        else:
            model_names = [f for f in os.listdir(model_path + "/feature_autoLearn") if '.sav' in f]
            for m in model_names:
                m_path = model_path + "/feature_autoLearn/" + m
                models.append(pickle.load(open(m_path, 'rb')))
                col_pairs.append(m.split('.sav')[0].split("__and__"))
        idf_rest = idf
        final_feats_df = spark.read.csv(model_path + "/feature_autoLearn/final_feats", header=True, inferSchema=True)
        # Note: to remove irrelevant columns when list_of_cols is a subset of previously trained columns 
        final_feats = final_feats_df\
            .withColumn('var', F.split('feature', '__and__')).withColumn('var1', F.col('var')[0])\
            .withColumn('var2', F.regexp_replace(F.col('var')[1], '_pred|_err', ''))\
            .where((F.col('var1').isin(list_of_cols)) & (F.col('var2').isin(list_of_cols)))\
            .where(F.col('selected')==True).select('feature').rdd.flatMap(lambda x: x).collect()
        
        if print_impact:
            print("Newly Constructed Features (Final): ", len(final_feats))

    else:
        ## Information Gain Based Filteration (1st Round)
        def IG_feature_selection(X, Y, IG_threshold=0, print_impact=False):

            IG_model = SelectKBest(mutual_info_classif, k='all')
            IG_model.fit_transform(X,Y)
            feats_score = list(IG_model.scores_)
            feats_selected = []
            for index,i in enumerate(X.columns):
                if feats_score[index] > IG_threshold:
                    feats_selected.append(i)
            if print_impact:
                print("Features dropped due to low Information Gain value: ", len(X.columns) - len(feats_selected))
                print("Features after IG based feature selection: ", len(feats_selected))
            return feats_selected
        
        idf = idf.withColumn('id', F.monotonically_increasing_id())
        idf_valid = idf.dropna(subset=list_of_cols+[label_col+'_converted'])
        idf_model = idf_valid.sample(False, min(1.0,float(max_size)/idf_valid.count()), 0)
        idf_rest = idf.subtract(idf_model)
        idf, idf_model, idf_rest = idf.drop('id'), idf_model.drop('id'), idf_rest.drop('id')
        idf_pd = idf_model.toPandas()

        all_feats = [e for e in idf_pd.columns if e in list_of_cols]
        X_init = idf_pd[all_feats]
        Y = idf_pd[label_col+'_converted']
        IG_feats = IG_feature_selection(X_init, Y, IG_threshold=IG_threshold, print_impact=print_impact)

        if len(IG_feats)<=1:
            warnings.warn("No transformation performed - feature_autoLearn")
            return idf.drop(label_col+'_converted')

        ## Feature Generation (Predicted + Error for every feature pair)
        X = idf_pd[IG_feats].astype(float)
        linear_pairs = []
        nonlinear_pairs = []
        for index, i in enumerate(IG_feats):
            for subindex, j in enumerate(IG_feats):
                if index != subindex:
                    if distance_correlation(X[i], X[j])>= dist_corr_threshold:
                        linear_pairs.append([i,j])
                    else:
                        nonlinear_pairs.append([i,j])
        #print(len(linear_pairs),len(nonlinear_pairs))
        # clf_linear = Ridge(alpha=1.0)
        output = Y
        for i,j in linear_pairs:
            clf_linear = Ridge(alpha=1.0)
            col_name_pred, col_name_err = i+"__and__"+j+"_pred", i+"__and__"+j+"_err"
            pred = clf_linear.fit(X[[i]],X[[j]]).predict(X[[i]])
            output = pd.concat([output,X[[j]], pd.Series(list(pred))], axis=1).rename(columns={0:(col_name_pred)})
            output[col_name_pred] = output.apply(lambda x: float(x[col_name_pred][0]), axis=1)
            #output.to_csv("test.csv", header=True, index=False)
            #output = pd.read_csv("test.csv")
            output[col_name_err] = output[j] - output[col_name_pred]
            output.drop(j, axis=1, inplace=True)
            models.append(clf_linear)
            col_pairs.append([i, j])
        #print(output.head())

        # clf_nonlinear = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',kernel_params=None)
        for i,j in nonlinear_pairs:
            clf_nonlinear = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',kernel_params=None)
            col_name_pred, col_name_err = i+"__and__"+j+"_pred", i+"__and__"+j+"_err"
            pred = clf_nonlinear.fit(X[[i]],X[[j]]).predict(X[[i]])
            output = pd.concat([output,X[[j]], pd.Series(list(pred))], axis=1).rename(columns={0:(col_name_pred)})
            output[col_name_pred] = output.apply(lambda x: float(x[col_name_pred][0]), axis=1)
            #output.to_csv("test.csv", header=True, index=False)
            #output = pd.read_csv("test.csv")
            output[col_name_err] = output[j] - output[col_name_pred]
            output.drop(j, axis=1, inplace=True)
            models.append(clf_nonlinear)
            col_pairs.append([i, j])
        #print(output.head())

        ## Stability Check on New Constructed Features 
        # Pending: to add the corresponding scripts but they are not written by us
        # Pending: sometimes not coverged warning is shown but the result looks fine.
        from randomized_lasso import RandomizedLasso
        from stability_selection import StabilitySelection
        X = output.drop(label_col+'_converted', axis=1)
        Y = output[label_col+'_converted']
        stability_model = StabilitySelection(base_estimator=RandomizedLasso(), lambda_name='alpha',
                                            threshold=stability_threshold, verbose=0)
        stable_feats = [i for i,j in zip(X.columns,list(stability_model.fit(X, Y))) if j == True]
        ## Information Gain Based Filteration (2nd Round)
        X_stable = output[stable_feats]
        final_feats = IG_feature_selection(X_stable,Y, IG_threshold=IG_threshold, print_impact=False)

        feats_selected_bool = [True if i in final_feats else False for i in X.columns]
        final_feats_df = spark.createDataFrame(zip(X.columns,feats_selected_bool), schema=['feature', 'selected'])
        odf_model = spark.createDataFrame(pd.concat([idf_pd[fixed_cols],output[final_feats]], axis=1))
        
        # filter out unused models
        models_, col_pairs_ = [], []
        for i in range(len(models)):
            m, [c1, c2] = models[i], col_pairs[i]
            if (c1+"__and__"+c2+"_pred" in final_feats) or (c1+"__and__"+c2+"_err" in final_feats):
                models_.append(m)
                col_pairs_.append([c1, c2])
        models, col_pairs = models_, col_pairs_

        if (pre_existing_model == False) & (model_path != "NA"):
            if emr_mode:
                subprocess.check_output(['bash', '-c', 'rm -r feature_autoLearn; mkdir feature_autoLearn'])
                for m, [c1, c2] in zip(models, col_pairs):
                    model_name = "feature_autoLearn/"+c1+"__and__"+c2+".sav"
                    pickle.dump(m, open(model_name, 'wb'))
                    bash_cmd = "aws s3 cp " + model_name + ' ' + model_path + "/feature_autoLearn/" + model_name
                    subprocess.check_output(['bash', '-c', bash_cmd])
                    #joblib.dump(imputer, "imputation_sklearn.sav")
            else:
                subprocess.check_output(['bash', '-c', 'cd ' + model_path + '; rm -r feature_autoLearn; mkdir feature_autoLearn'])
                for m, [c1, c2] in zip(models, col_pairs):
                    local_path = model_path + "/feature_autoLearn/"+c1+"__and__"+c2+".sav"
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    #joblib.dump(imputer, local_path)
                    pickle.dump(m, open(local_path, 'wb'))
            
            final_feats_df.repartition(1).write.csv(model_path + "/feature_autoLearn/final_feats", header=True, mode='overwrite')
        
        if print_impact:
            print("No. of linear feature pairs: ", len(linear_pairs))
            print("No. of nonlinear feature pairs: ", len(nonlinear_pairs))
            print("Newly Constructed Features: ", len(output.columns) - 1)
            print("Newly Constructed Features (After Stability Check): ", len(stable_feats))
            print("Newly Constructed Features (Final): ", len(final_feats))
    
    if idf_rest.count() > 0:
        def model_prediction(model):
            @F.pandas_udf(returnType=T.ArrayType(T.DoubleType()))
            def prediction(col):
                X = pd.concat([col], axis=1)
                return pd.Series(row.tolist() for row in model.predict(X))        
            return prediction
        
        odf_rest = idf_rest # .select(label_col+'_converted')
        for i in range(len(models)):
            m, [c1, c2] = models[i], col_pairs[i]

            col_name_pred, col_name_err = c1+"__and__"+c2+"_pred", c1+"__and__"+c2+"_err"
            odf_rest = odf_rest.withColumn(col_name_pred, model_prediction(m)(c1)[0])\
                .withColumn(col_name_err, F.col(c2) - F.col(col_name_pred))
            # odf_rest = odf_rest.withColumn(col_name_pred, F.when(F.isnull(c1), None).otherwise(model_prediction(m)(c1)[0]))\
            #     .withColumn(col_name_err, F.when((F.isnull(col_name_pred)) | (F.isnull(c2)), None)\
            #                                .otherwise(F.col(c2) - F.col(col_name_pred)))
        odf_rest = odf_rest.select(fixed_cols + final_feats)

    if pre_existing_model:
        odf = odf_rest
    elif idf_rest.count() == 0:
        odf = odf_model
    else:
        odf = odf_model.unionByName(odf_rest.select(odf_model.columns))
        
    
    idf = idf.drop(label_col+'_converted')
    if print_impact:
        print("Before:")
        idf.show(5)
        print("After:")
        odf.show(5)
    
    return odf


def outlier_categories(spark, idf, list_of_cols='all', drop_cols=[], coverage=1.0, max_category=50,
                       pre_existing_model=False, model_path="NA", output_mode='replace', print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of categorical columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all categorical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param coverage: Defines the minimum % of rows that will be mapped to actual category name and the rest to be mapped
                     to others and takes value between 0 to 1. Coverage of 0.8 can be interpreted as top frequently seen
                     categories are considered till it covers minimum 80% of rows and rest lesser seen values are mapped to others.
    :param max_category: Even if coverage is less, only (max_category - 1) categories will be mapped to actual name and rest to others.
                         Caveat is when multiple categories have same rank, then #categories can be more than max_category.
    :param pre_existing_model: Boolean argument – True or False. True if the model with the outlier/other values
                               for each attribute exists already to be used, False Otherwise.
    :param model_path: If pre_existing_model is True, this argument is path for the pre-saved model.
                       If pre_existing_model is False, this field can be used for saving the model.
                       Default "NA" means there is neither pre-existing model nor there is a need to save one.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with transformed column. “append” option append transformed
                        column to the input dataset with a postfix "_outliered" e.g. column X is appended as X_outliered.
    :return: Dataframe after outlier treatment
    """

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == 'all':
        list_of_cols = cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in cat_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')

    if len(list_of_cols) == 0:
        warnings.warn("No Outlier Categories Computation - No categorical column(s) to transform")
        return idf
    if (coverage <= 0) | (coverage > 1):
        raise TypeError('Invalid input for Coverage Value')
    if max_category < 2:
        raise TypeError('Invalid input for Maximum No. of Categories Allowed')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model == True:
        df_model = spark.read.csv(model_path + "/outlier_categories", header=True, inferSchema=True)
    else:
        for index, i in enumerate(list_of_cols):
            window = Window.partitionBy().orderBy(F.desc('count_pct'))
            df_cats = idf.groupBy(i).count().dropna() \
                .withColumn('count_pct', F.col('count') / F.sum('count').over(Window.partitionBy())) \
                .withColumn('rank', F.rank().over(window)) \
                .withColumn('cumu', F.sum('count_pct').over(window.rowsBetween(Window.unboundedPreceding, 0))) \
                .withColumn('lag_cumu', F.lag('cumu').over(window)).fillna(0) \
                .where(~((F.col('cumu') >= coverage) & (F.col('lag_cumu') >= coverage))) \
                .where(F.col('rank') <= (max_category - 1)) \
                .select(F.lit(i).alias('attribute'), F.col(i).alias('parameters'))
            if index == 0:
                df_model = df_cats
            else:
                df_model = df_model.union(df_cats)

    odf = idf
    for i in list_of_cols:
        parameters = df_model.where(F.col('attribute') == i).select('parameters').rdd.flatMap(lambda x: x).collect()
        if output_mode == 'replace':
            odf = odf.withColumn(i, F.when((F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)).otherwise(
                "others"))
        else:
            odf = odf.withColumn(i + "_outliered",
                                 F.when((F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)).otherwise(
                                     "others"))

    # Saving model File if required
    if (pre_existing_model == False) & (model_path != "NA"):
        df_model.repartition(1).write.csv(model_path + "/outlier_categories", header=True, mode='overwrite')

    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_outliered") for i in list_of_cols]
        uniqueCount_computation(spark, idf, list_of_cols).select('attribute',
                                                                 F.col("unique_values").alias(
                                                                     "uniqueValues_before")).show(
            len(list_of_cols))
        uniqueCount_computation(spark, odf, output_cols).select('attribute',
                                                                F.col("unique_values").alias(
                                                                    "uniqueValues_after")).show(
            len(list_of_cols))

    return odf


def declare_missing(spark, idf, list_of_cols='all', drop_cols=[], 
                    missing_values=['', ' ', 'nan', 'null', 'na', 'n/a', 'NaN', 'none', '?'], 
                    output_mode="replace", print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: all or list of columns (in list format or string separated by |)
    missing_values: list of values to be replaced by null
    output_mode: replace or append
    return: Dataframe
    '''
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    elif isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    
    if (len(list_of_cols) == 0) | (any(x not in idf.columns for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    if isinstance(missing_values, str):
        missing_values = [x.strip() for x in missing_values.split('|')]
    
    odf = idf
    for i in list_of_cols:
        modify_col = ((i + "_missing") if (output_mode == "append") else i)
        odf = odf.withColumn(modify_col, F.when(F.col(i).isin(missing_values), None).otherwise(F.col(i)))
    
    # Note: if output_mode == 'append', drop columns with equal missing_count in idf and odf
    if output_mode == 'append':
        output_cols = [(i+"_missing") for i in list_of_cols]
        idf_missing = missingCount_computation(spark, idf, list_of_cols).select('attribute', F.col("missing_count").alias("missingCount_before"))
        odf_missing = missingCount_computation(spark, odf, output_cols).select('attribute', F.col("missing_count").alias("missingCount_after"))
        odf_print = idf_missing\
                .join(odf_missing.withColumnRenamed('attribute', 'attribute_after') \
                                 .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")), 
                      'attribute', 'inner')
        same_cols = odf_print.where(F.col('missingCount_before')==F.col('missingCount_after'))\
            .select('attribute_after').rdd.flatMap(lambda x: x).collect()
        odf = odf.drop(*same_cols)
        odf_print = odf_print.where(F.col('missingCount_before')!=F.col('missingCount_after'))

    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
            idf_missing = missingCount_computation(spark, idf, list_of_cols).select('attribute', F.col("missing_count").alias("missingCount_before"))
            odf_missing = missingCount_computation(spark, odf, output_cols).select('attribute', F.col("missing_count").alias("missingCount_after"))
            odf_print = idf_missing.join(odf_missing, 'attribute', 'inner')
        odf_print.show(len(list_of_cols))
    
    return odf


def catfeats_basic_cleaning(spark, idf, list_of_cols='all', drop_cols=[], output_mode='replace', print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: all (numerical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    output_mode: append or replace
    return: Cleaned Dataframe
    '''

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == 'all':
        list_of_cols = cat_cols
    elif isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    
    if (len(list_of_cols) == 0) | (any(x not in cat_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    cat_cols = attributeType_segregation(idf.select(list_of_cols))[1]
    list_of_cols = cat_cols
        
    null_vocab = ['',' ','nan', 'null', 'na', 'inf', 'n/a', 'not defined', 'none', 'undefined','blank']
    odf_tmp=idf
    modify_list = []
    for i in list_of_cols:
        modify_col = ((i + "_cleaned") if (output_mode == "append") else i)
        modify_list.append(modify_col)
        # Note: removed regexp_replace part: may lead to misunderstanding? For example Other-relative becomes otherrelative
        odf_tmp = odf_tmp.withColumn(modify_col, F.lower(F.trim(F.col(i))))
        # .withColumn(modify_col, F.regexp_replace(F.col(modify_col),'[ &$;:.,*#@_?%!^()-/\'\"\\\]',''))
    odf = declare_missing(spark, odf_tmp, list_of_cols=modify_list, missing_values=null_vocab)
    
    # Note: if output_mode == 'append', drop columns if col i = col i + "_cleaned" for all rows
    if output_mode == 'replace':
            output_cols = list_of_cols
    else:
        output_cols = []
        for i in list_of_cols:
            unequal_rows = odf.withColumn('col_equality', F.col(i)==F.col(i + "_cleaned"))\
                .where(F.col('col_equality')==False).count()
            if unequal_rows == 0:
                odf = odf.drop(i + "_cleaned")
            else:
                output_cols.append(i + "_cleaned")

    if print_impact:
        idf_missing = missingCount_computation(spark, idf, list_of_cols).select('attribute', F.col("missing_count").alias("missingCount_before"))
        odf_missing = missingCount_computation(spark, odf, output_cols).select('attribute', F.col("missing_count").alias("missingCount_after"))
        
        if output_mode == 'replace':
            odf_print = idf_missing.join(odf_missing, 'attribute', 'inner')
        else:
            odf_print = idf_missing\
                .join(odf_missing\
                      .withColumnRenamed('attribute', 'attribute_after') \
                      .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")) \
                      .drop('missing_pct'), 'attribute', 'inner') 
        odf_print.show(len(output_mode))
    return odf
    

def catfeats_fuzzy_matching(spark, idf, list_of_cols='all', drop_cols=[], basic_cleaning=False, 
                            pre_existing_model=False, model_path="NA",
                            output_mode='replace', print_impact=False):
    # Note: added variables: pre_existing_model, model_path
    '''
    idf: Input Dataframe
    list_of_cols: all (numerical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    basic_cleaning: True or False
    output_mode: append or replace
    return: Cleaned Dataframe
    '''

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == 'all':
        list_of_cols = cat_cols
    elif isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    
    if (len(list_of_cols) == 0) | (any(x not in cat_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    cat_cols = attributeType_segregation(idf.select(list_of_cols))[1]
    list_of_cols = cat_cols

    if basic_cleaning:
        idf_cleaned = catfeats_basic_cleaning(spark, idf, list_of_cols)
    else:
        idf_cleaned = idf
    idf_cleaned.persist()
    odf = idf_cleaned
    unchanged_cols = []
    for i in list_of_cols:
        if pre_existing_model:
            df_clusters = spark.read.parquet(model_path+"/catfeats_fuzzy_matching/"+i)
        else:
            ls = idf_cleaned.select(i).dropna().distinct().rdd.flatMap(lambda x:x).collect()
            distance_array = np.ones((len(ls),(len(ls))))*0
            for k in range(1, len(ls)):
                for j in range(k):
                    s1 = fuzz.token_set_ratio(ls[k],ls[j]) + 0.000000000001
                    s2 = fuzz.partial_ratio(ls[k],ls[j]) + 0.000000000001
                    similarity_score = 2*s1*s2/(s1+s2)
                    # Note: merge categories with >=85 similarity
                    distance = 100 if similarity_score < 85 else 0
                    distance_array[k][j] = distance
                    distance_array[j][k] = distance_array[k][j]
        
            # Note: replaced the AffinityPropagation part by DBSCAN which create clusters with 1 point if all are different
            clusters = cluster.DBSCAN(metric="precomputed", min_samples=1).fit_predict(distance_array)

            lol = list(zip(ls,clusters))
            lol = [[str(e[0]), "cluster"+ str(e[1])] for e in lol]
            df_clusters = spark.createDataFrame(lol, schema = [i,'cluster'])\
                .groupBy('cluster').agg(F.collect_list(i).alias(i))\
                .withColumn('mapped_value', F.col(i)[0]).drop('cluster')
            if len(set(clusters))==len(clusters):
                unchanged_cols.append(i)
            elif print_impact:
                # Note: skip printing impact if the column is unchanged
                df_clusters.show(df_clusters.count(), False)
            df_clusters = df_clusters.withColumn(i, F.explode(i))
            if (pre_existing_model == False) & (model_path != "NA"):
                df_clusters.repartition(1).write.parquet(model_path+"/catfeats_fuzzy_matching/"+i, mode='overwrite') 
        # Note: if existing model is applied on another dataset, some keys might not exist in df_clusters -> use the original value
        odf = odf.join(df_clusters,i,'left_outer')\
            .withColumn('mapped_value', F.coalesce('mapped_value', i))\
            .withColumnRenamed('mapped_value', i+"_fuzzmatched")
        
    if output_mode == 'replace':
        for i in list_of_cols:
            odf = odf.drop(i).withColumnRenamed(i+"_fuzzmatched", i)
    else:
        # Note: remove unchanged columns
        odf = odf.drop(*[(i+"_fuzzmatched") for i in unchanged_cols])
    
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_fuzzmatched") for i in list_of_cols if i not in unchanged_cols]

        idf_unique = uniqueCount_computation(spark, idf, list_of_cols)\
            .select('attribute', F.col("unique_values").alias("uniqueValues_before"))
        odf_unique = uniqueCount_computation(spark, odf, output_cols)\
            .select('attribute', F.col("unique_values").alias("uniqueValues_after"))
        
        if output_mode == 'replace':
            odf_print = idf_unique.join(odf_unique, 'attribute', 'inner')
        else:
            odf_print = idf_unique\
                .join(odf_unique\
                      .withColumnRenamed('attribute', 'attribute_after') \
                      .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-12)")),
                      'attribute', 'inner') 
        odf_print.show(len(output_mode))
    
    idf_cleaned.unpersist()
    return odf


def cat_to_num_supervised(spark, idf, list_of_cols='all', drop_cols=[], label_col="label", event_label=1, nonevent_label=0,
                          seed=0, pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: all (categorical columns except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from analysis
    pre_existing_model: categorical value to numerical value model. True if model files exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  Default NA means there is neither pre_existing_model nor there is a need to save one.
    output_mode: append or replace
    seed: Saving the intermediate data (optimization purpose only)
    return: Dataframe
    '''
    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == 'all':
        list_of_cols = cat_cols
    elif isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    list_of_cols = list(set([e for e in list_of_cols if (e not in drop_cols) & (e != label_col)]))
    
    if (len(list_of_cols) == 0) | (any(x not in cat_cols for x in list_of_cols)):
        raise TypeError('Invalid input for Column(s)')
    
    cat_cols = attributeType_segregation(idf.select(list_of_cols))[1]
    list_of_cols = cat_cols
    event_label, nonevent_label = str(event_label), str(nonevent_label)
    odf = idf

    for index, i in enumerate(list_of_cols):
        # if index > 0:
        #     odf = spark.read.parquet("intermediate_data/df_index" + str(index-1) + "_seed" +str(seed))
        if pre_existing_model == True:
            df_tmp = spark.read.csv(model_path + "/cat_to_num_supervised/" + i, header=True, inferSchema=True)
        else:        
            df_tmp = idf.groupBy(i, label_col).count().groupBy(i).pivot(label_col).sum('count').fillna(0)\
                .withColumn(i + '_value', F.round(F.col(event_label)/(F.col(event_label)+F.col(nonevent_label)), 4))\
                .drop(*[event_label, nonevent_label])
        
        odf = odf.join(df_tmp, i, 'left_outer')
        
        # Saving model File if required
        if (pre_existing_model == False) & (model_path != "NA"):
            df_tmp.repartition(1).write.csv(model_path+ "/cat_to_num_supervised/" + i, header=True, mode='overwrite')

        # Pending: sometimes the loop runs forever so comented this out for now
        # error = 1
        # while error > 0:
        #     try:
        #         odf.write.parquet("intermediate_data/df_index"+str(index)+"_seed"+str(seed))
        #         error = 0
        #     except:
        #         seed += 1
        #         pass
            
    if output_mode =='replace':
        for i in list_of_cols:
            odf = odf.drop(i).withColumnRenamed(i + '_value', i)
        odf = odf.select(idf.columns)
    
    if print_impact:
        if output_mode == 'replace':
                output_cols = list_of_cols
        else:
            output_cols = [(i+"_value") for i in list_of_cols]
        print("Before: ")
        idf.select(list_of_cols).describe().where(F.col('summary').isin('count','min','max')).show()
        print("After: ")
        odf.select(output_cols).describe().where(F.col('summary').isin('count','min','max')).show()
        
    return odf

