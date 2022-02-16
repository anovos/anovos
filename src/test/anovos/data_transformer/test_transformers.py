import os
import pytest
from pytest import approx
from pyspark.sql import functions as F
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.transformers import attribute_binning, monotonic_binning, cat_to_num_unsupervised, cat_to_num_supervised, z_standardization, IQR_standardization, normalization, imputation_MMM, imputation_sklearn, imputation_matrixFactorization, auto_imputation, autoencoder_latentFeatures, PCA_latentFeatures, feature_transformation, boxcox_transformation, outlier_categories
sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"

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
    odf = imputation_sklearn(spark_session, df, method_type="KNN")
    assert odf.where(F.col("age").isNull()).count() == 0
    assert odf.where(F.col("logfnl").isNull()).count() == 0
    assert odf.where(F.col("hours-per-week").isNull()).count() == 0
    assert odf.where(F.col("education").isNull()).count() == 258
    
def test_imputation_matrixFactorization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_matrixFactorization(spark_session, df, id_col="ifa")
    assert odf.where(F.col("age").isNull()).count() == 0
    assert odf.where(F.col("logfnl").isNull()).count() == 0
    assert odf.where(F.col("hours-per-week").isNull()).count() == 0
    assert odf.where(F.col("education").isNull()).count() == 258

def test_imputation_MMM(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_MMM(spark_session, df, method_type="mode")
    assert odf.where(F.col("age").isNull()).count() == 0
    assert odf.where(F.col("logfnl").isNull()).count() == 0
    assert odf.where(F.col("hours-per-week").isNull()).count() == 0
    assert odf.where(F.col("education").isNull()).count() == 0
    assert odf.where(F.col("race").isNull()).count() == 0
    assert odf.where(F.col("relationship").isNull()).count() == 0
    
def test_auto_imputation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = auto_imputation(spark_session, df, id_col="ifa")
    assert odf[0].where(F.col("age").isNull()).count() == 0
    assert odf[0].where(F.col("logfnl").isNull()).count() == 0
    assert odf[0].where(F.col("hours-per-week").isNull()).count() == 0
    assert odf[0].where(F.col("education").isNull()).count() == 0
    assert odf[0].where(F.col("race").isNull()).count() == 0
    assert odf[0].where(F.col("relationship").isNull()).count() == 0

#latent_features
def test_PCA_latentFeatures(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = PCA_latentFeatures(spark_session, df, list_of_cols=["age","fnlwgt","logfnl","education-num","hours-per-week"],explained_variance_cutoff=0.3)
    assert len(odf.columns) < len(df.columns)
    assert len(odf.columns) == 13
    odf = PCA_latentFeatures(spark_session, df, list_of_cols=["age","fnlwgt","logfnl","education-num","hours-per-week"],explained_variance_cutoff=0.3,output_mode="append")
    assert len(odf.columns) > len(df.columns)
    assert len(odf.columns) == 18
    assert odf.where(F.col("education").isNull()).count() == 91
    assert odf.where(F.col("race").isNull()).count() == 58
    assert odf.where(F.col("latent_0").isNull()).count() == 0

def test_autoencoder_latentFeatures(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = autoencoder_latentFeatures(spark_session, df, list_of_cols=["age","fnlwgt","logfnl","education-num","hours-per-week"] ,epochs=20, reduction_params=0.5)
    assert len(odf.columns) < len(df.columns)
    assert len(odf.columns) == 14
    odf = autoencoder_latentFeatures(spark_session, df, list_of_cols=["age","fnlwgt","logfnl","education-num","hours-per-week"] ,epochs=20, reduction_params=0.5, output_mode= "append")
    assert len(odf.columns) > len(df.columns)
    assert len(odf.columns) == 19    
    assert odf.where(F.col("latent_0").isNull()).count() == 0
    assert odf.where(F.col("latent_1").isNull()).count() == 0

#feature_transformation
def test_feature_transformation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = feature_transformation(df)
    odf_pd=odf.where(F.col("ifa") == "27520a").toPandas()
    assert approx(odf_pd["age"][0])==7.14142842854285
    assert approx(odf_pd["fnlwgt"][0])==399.6936326738268
    assert approx(odf_pd["hours-per-week"][0])==4.47213595499958
    
def test_boxcox_transformation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = boxcox_transformation(df,drop_cols=["capital-gain","capital-loss"], boxcox_lambda=0.5)
    odf_pd=odf.where(F.col("ifa") == "27520a").toPandas()
    assert approx(odf_pd["age"][0])==7.14142842854285
    assert approx(odf_pd["fnlwgt"][0])==399.6936326738268
    assert approx(odf_pd["hours-per-week"][0])==4.47213595499958 

def test_outlier_categories(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf= outlier_categories(spark_session, df, max_category=15)
    assert odf.select("workclass").distinct().count() == 10
    assert odf.select("education").distinct().count() == 16
    assert odf.select("native-country").distinct().count() == 15
    assert odf.select("occupation").distinct().count() == 16
    assert odf.select("sex").distinct().count() == 4
    assert odf.select("race").distinct().count() == 10
    assert odf.select("marital-status").distinct().count() == 8

#binning
def test_attribute_binning(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = attribute_binning(spark_session, df, bin_size=20)
    odf_min_dict = odf.describe().where(F.col("summary") == "min").toPandas().to_dict('list')
    odf_max_dict = odf.describe().where(F.col("summary") == "max").toPandas().to_dict('list')
    assert round(float(odf_min_dict['age'][0])) == 1
    assert round(float(odf_min_dict['fnlwgt'][0])) == 1
    assert round(float(odf_min_dict['hours-per-week'][0])) == 1
    assert round(float(odf_max_dict['age'][0])) == 20
    assert round(float(odf_max_dict['fnlwgt'][0])) == 20
    assert round(float(odf_max_dict['hours-per-week'][0])) == 20
    
def test_monotonic_binning(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = monotonic_binning(spark_session, df, label_col="income", event_label="<=50K", bin_method="equal_range", bin_size=10)
    odf_min_dict = odf.describe().where(F.col("summary") == "min").toPandas().to_dict('list')
    odf_max_dict = odf.describe().where(F.col("summary") == "max").toPandas().to_dict('list')
    assert round(float(odf_min_dict['age'][0])) == 1
    assert round(float(odf_min_dict['fnlwgt'][0])) == 1
    assert round(float(odf_min_dict['hours-per-week'][0])) == 1
    assert round(float(odf_max_dict['age'][0])) == 10
    assert round(float(odf_max_dict['fnlwgt'][0])) == 10
    assert round(float(odf_max_dict['hours-per-week'][0])) == 10

 #categorical_to_numerical_transformation   
def test_cat_to_num_unsupervised(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf =cat_to_num_unsupervised(spark_session, df, drop_cols=["ifa"], method_type=1, index_order="frequencyDesc", cardinality_threshold=100)
    odf_min_dict = odf.describe().where(F.col("summary") == "min").toPandas().to_dict('list')
    assert round(float(odf_min_dict['workclass'][0])) == 0
    assert round(float(odf_min_dict['marital-status'][0])) == 0
    assert round(float(odf_min_dict['relationship'][0])) == 0
    
    odf =cat_to_num_unsupervised(spark_session, df, drop_cols=["ifa"],method_type=0, cardinality_threshold=100)
    odf_min_dict = odf.describe().where(F.col("summary") == "min").toPandas().to_dict('list')
    odf_max_dict = odf.describe().where(F.col("summary") == "max").toPandas().to_dict('list')
    assert round(float(odf_min_dict['relationship_0'][0])) == 0
    assert round(float(odf_min_dict['race_7'][0])) == 0
    assert round(float(odf_min_dict['marital-status_1'][0])) == 0
    assert round(float(odf_min_dict['sex_1'][0])) == 0
    assert round(float(odf_min_dict['occupation_12'][0])) == 0
    assert round(float(odf_max_dict['relationship_0'][0])) == 1
    assert round(float(odf_max_dict['race_7'][0])) == 1
    assert round(float(odf_max_dict['marital-status_1'][0])) == 1
    assert round(float(odf_max_dict['sex_1'][0])) == 1
    assert round(float(odf_max_dict['occupation_12'][0])) == 1
    
def test_cat_to_num_supervised(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf= cat_to_num_supervised(spark_session,df,drop_cols=["ifa"],label_col="income",event_label="<=50K")
    df_workclass_private=df.where(F.col("workclass")=="Private").select("income").toPandas().value_counts()
    assert round(odf.where(F.col("ifa")=="27520a").toPandas()["workclass"][0]) == round(df_workclass_private[0]/(df_workclass_private[0]+df_workclass_private[1]))
    df_workclass_local_gov=df.where(F.col("workclass")=="Local-gov").select("income").toPandas().value_counts()
    assert round(odf.where(F.col("ifa")=="6144a").toPandas()["workclass"][0]) == round(df_workclass_local_gov[0]/(df_workclass_local_gov[0]+df_workclass_local_gov[1]))
    df_workclass_federal_gov=df.where(F.col("workclass")=="Federal-gov").select("income").toPandas().value_counts()
    assert round(odf.where(F.col("ifa")=="23710a").toPandas()["workclass"][0]) == round(df_workclass_federal_gov[0]/(df_workclass_federal_gov[0]+df_workclass_federal_gov[1]))