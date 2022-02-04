import os

import pytest
from anovos.data_ingest.data_ingest import *
from anovos.data_transformer.transformers import *

sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"

@pytest.mark.usefixtures("spark_session")
#scaling
def test_z_standardization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = z_standardization(spark_session, df)
    odf_stddev_dict = odf.describe().where(F.col("summary") == "stddev").toPandas().to_dict('list')
    assert round(float(odf_stddev_dict['age'][0])) == 1.0
    assert round(float(odf_stddev_dict['fnlwgt'][0])) == 1.0
    assert round(float(odf_stddev_dict['hours-per-week'][0])) == 1.0

def test_IQR_standardization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = IQR_standardization(spark_session, df)
    odf_median_dict = odf.summary().where(F.col("summary") == "50%").toPandas().to_dict('list')
    assert round(float(odf_median_dict['age'][0])) == 0.0
    assert round(float(odf_median_dict['fnlwgt'][0])) == 0.0
    assert round(float(odf_median_dict['hours-per-week'][0])) == 0.0

def test_normalization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = normalization(df)
    odf_min_dict = odf.describe().where(F.col("summary") == "min").toPandas().to_dict('list')
    assert round(float(odf_min_dict['age'][0])) == 0.0
    assert round(float(odf_min_dict['fnlwgt'][0])) == 0.0
    assert round(float(odf_min_dict['hours-per-week'][0])) == 0.0
    odf_max_dict = odf.describe().where(F.col("summary") == "max").toPandas().to_dict('list')
    assert round(float(odf_max_dict['age'][0])) == 1.0
    assert round(float(odf_max_dict['fnlwgt'][0])) == 1.0
    assert round(float(odf_max_dict['hours-per-week'][0])) == 1.0

#imputation
def test_imputation_sklearn(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_sklearn(spark, df, method_type=method_type_test)

def test_imputation_matrixFactorization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_matrixFactorization(spark, df, id_col="ifa")

def test_imputation_custom(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_custom(spark, df, list_of_fills=fills_test, method_type=method_type_test)

def test_imputation_comparison(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    imputation_comparison(spark, df, list_of_cols=test_cols+['platform'], id_col="ifa")

#latent_features
def test_PCA_latentFeatures(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = PCA_latentFeatures(spark, df)

def test_autoencoders_latentFeatures(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = autoencoders_latentFeatures(spark, df, epochs=20, print_impact=True)

#feature_transformation
def test_feature_transformation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = feature_transformation(spark, df, method_type=method_type_test)

def test_declare_missing
    df = read_dataset(spark_session, sample_parquet, "parquet")

def test_cat_to_num_supervised(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")

#autoLearn_catfeats

