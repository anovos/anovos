import pytest
import os
from com.mw.ds.data_analyzer.association_evaluator import *

sample_parquet = "./data/test_dataset/part-00000-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
sample_avro = "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
sample_output_path = "./data/tmp/output/"

@pytest.mark.usefixtures("spark_session")

# def correlation_matrix(idf, list_of_cols='all', drop_cols=[], plot=False):
# def variable_clustering(idf, list_of_cols='all', drop_cols=[], sample_size=100000, plot=False):


def test_IV_calculation(spark_session):
    test_df = spark.read.parquet(sample_parquet)
    test_df = test_df.withColumn('label', F.when(F.col('income') == '<=50K', F.lit(0.0)).when(F.col('income') == '>50K', F.lit(1.0))).drop('income')
    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['age'][0] == 28
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['sex'][0] == 'Male'  
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['education'][0] == '11th' 

    result_df = IV_calculation(test_df,drop_cols = ['ifa'])

    assert result_df.count() == 15
    assert result_df.where(F.col("feature") == "relationship").toPandas().to_dict('list')['iv'][0] == 1.6205   
    assert result_df.where(F.col("feature") == "marital-status").toPandas().to_dict('list')['iv'][0] == 1.3929
    assert result_df.where(F.col("feature") == "age").toPandas().to_dict('list')["iv"][0] == 1.1891
    assert result_df.where(F.col("feature") == "occupation").toPandas().to_dict('list')['iv'][0] == 0.7467   
    assert result_df.where(F.col("feature") == "education").toPandas().to_dict('list')['iv'][0] == 0.7459
    assert result_df.where(F.col("feature") == "education-num").toPandas().to_dict('list')['iv'][0] == 0.7095 
    assert result_df.where(F.col("feature") == "hours-per-week").toPandas().to_dict('list')['iv'][0] == 0.4441
    assert result_df.where(F.col("feature") == "capital-gain").toPandas().to_dict('list')['iv'][0] == 0.3179 
    assert result_df.where(F.col("feature") == "sex").toPandas().to_dict('list')['iv'][0] == 0.3106 
    assert result_df.where(F.col("feature") == "workclass").toPandas().to_dict('list')['iv'][0] == 0.1669
    
def test_IG_calculation(spark_session):
    test_df = spark.read.parquet(sample_parquet)
    test_df = test_df.withColumn('label', F.when(F.col('income') == '<=50K', F.lit(0.0)).when(F.col('income') == '>50K', F.lit(1.0))).drop('income')
    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['age'][0] == 28
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['sex'][0] == 'Male'  
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['education'][0] == '11th' 

    result_df1 = IG_calculation(test_df,drop_cols = ['ifa'])

    assert result_df1.count() == 15
    assert result_df1.where(F.col("feature") == "relationship").toPandas().to_dict('list')['ig'][0] == 0.1702   
    assert result_df1.where(F.col("feature") == "marital-status").toPandas().to_dict('list')['ig'][0] == 0.1583
    assert result_df1.where(F.col("feature") == "age").toPandas().to_dict('list')["ig"][0] == 0.0943
    assert result_df1.where(F.col("feature") == "occupation").toPandas().to_dict('list')['ig'][0] == 0.0917   
    assert result_df1.where(F.col("feature") == "education").toPandas().to_dict('list')['ig'][0] == 0.0873
    assert result_df1.where(F.col("feature") == "education-num").toPandas().to_dict('list')['ig'][0] == 0.0888 
    assert result_df1.where(F.col("feature") == "hours-per-week").toPandas().to_dict('list')['ig'][0] == 0.0552
    assert result_df1.where(F.col("feature") == "capital-gain").toPandas().to_dict('list')['ig'][0] == 0.0431
    assert result_df1.where(F.col("feature") == "sex").toPandas().to_dict('list')['ig'][0] == 0.0379 
    assert result_df1.where(F.col("feature") == "workclass").toPandas().to_dict('list')['ig'][0] == 0.0223
    
def test_variable_clustering(spark_session):
    test_df = spark.read.parquet(sample_parquet)
    test_df = test_df.withColumn('label', F.when(F.col('income') == '<=50K', F.lit(0.0)).when(F.col('income') == '>50K', F.lit(1.0))).drop('income')
    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['age'][0] == 28
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['sex'][0] == 'Male'  
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['education'][0] == '11th' 

    result_df2 = variable_clustering(test_df,drop_cols = ['ifa','label'])

    assert result_df2.count() == 15
    assert len(result_df2.columns) == 3
    assert result_df2.where((F.col("cluster")==0) & (F.col("feature") == "relationship")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.3409127531278682 
    assert result_df2.where((F.col("cluster")==0) & (F.col("feature") == "sex")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.33779295118072383
    assert result_df2.where((F.col("cluster")==0) & (F.col("feature") == "marital-status")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.46933342119492977
    assert result_df2.where((F.col("cluster")==0) & (F.col("feature") == "hours-per-week")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.8105795691359969
    assert result_df2.where((F.col("cluster")==1) & (F.col("feature") == "fnlwgt")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.22615883103056889
    assert result_df2.where((F.col("cluster")==1) & (F.col("feature") == "logfnl")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.22574711203494083
    assert result_df2.where((F.col("cluster")==2) & (F.col("feature") == "capital-loss")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.9184927063017515
    assert result_df2.where((F.col("cluster")==2) & (F.col("feature") == "education-num")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.348300651814405
    assert result_df2.where((F.col("cluster")==2) & (F.col("feature") == "occupation")).toPandas().to_dict('list')['RS_Ratio'][0] == 0.3690423679907467
    



    
