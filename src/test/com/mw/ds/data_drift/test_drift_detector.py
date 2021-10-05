import pytest
import os
from com.mw.ds.data_drift.drift_detector import *

sample_parquet1 = "./data/test_dataset/part-00000-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_parquet2 = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
sample_avro = "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
sample_output_path = "./data/tmp/output/"

@pytest.mark.usefixtures("spark_session")

def test_drift_statistics(spark_session):
    test_df = spark.read.parquet(sample_parquet1)
    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['age'][0] == 28
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['sex'][0] == 'Male'  
    assert test_df.where(F.col("ifa") == "4062a").toPandas().to_dict('list')['education'][0] == '11th'
    
    test_df1 = spark.read.parquet(sample_parquet2)
    assert test_df1.where(F.col("ifa") == "27520a").count() == 1
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['sex'][0] == 'Male'  
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == "HS-grad" 
    
    #testing using PSI

    result_df = drift_statistics(test_df,test_df1,drop_cols = ['income'],threshold=0.5)

    assert result_df.count() == 16
    assert result_df.where(F.col("attribute") == "workclass").toPandas().to_dict('list')['PSI'][0] == 0.0016 
    assert result_df.where(F.col("attribute") == "workclass").toPandas().to_dict('list')['flagged'][0] == 0   
    assert result_df.where(F.col("attribute") == "sex").toPandas().to_dict('list')['PSI'][0] == 0.0
    assert result_df.where(F.col("attribute") == "sex").toPandas().to_dict('list')['flagged'][0] == 0
    assert result_df.where(F.col("attribute") == "education").toPandas().to_dict('list')["PSI"][0] == 0.0021
    assert result_df.where(F.col("attribute") == "education").toPandas().to_dict('list')["flagged"][0] == 0
    assert result_df.where(F.col("attribute") == "native-country").toPandas().to_dict('list')['PSI'][0] == 0.0043   
    assert result_df.where(F.col("attribute") == "native-country").toPandas().to_dict('list')['flagged'][0] == 0
    assert result_df.where(F.col("attribute") == "occupation").toPandas().to_dict('list')['PSI'][0] == 0.0019 
    assert result_df.where(F.col("attribute") == "occupation").toPandas().to_dict('list')['flagged'][0] == 0
    assert result_df.where(F.col("attribute") == "ifa").toPandas().to_dict('list')['PSI'][0] == 0.6122 
    assert result_df.where(F.col("attribute") == "ifa").toPandas().to_dict('list')['flagged'][0] == 1 
    assert result_df.where(F.col("attribute") == "age").toPandas().to_dict('list')['PSI'][0] == 8.0E-4 
    assert result_df.where(F.col("attribute") == "age").toPandas().to_dict('list')['flagged'][0] == 0
    
    #testing using JSD
    result_df1 = drift_statistics(test_df,test_df1,drop_cols = ['income'],method_type='JSD',bin_size=110,threshold = 0.05)
    
    assert result_df1.count() == 16
    assert result_df1.where(F.col("attribute") == "logfnl").toPandas().to_dict('list')['JSD'][0] == 0.0015 
    assert result_df1.where(F.col("attribute") == "logfnl").toPandas().to_dict('list')['flagged'][0] == 0   
    assert result_df1.where(F.col("attribute") == "sex").toPandas().to_dict('list')['JSD'][0] == 0.0
    assert result_df1.where(F.col("attribute") == "sex").toPandas().to_dict('list')['flagged'][0] == 0
    assert result_df1.where(F.col("attribute") == "education").toPandas().to_dict('list')["JSD"][0] == 3.0E-4
    assert result_df1.where(F.col("attribute") == "education").toPandas().to_dict('list')["flagged"][0] == 0
    assert result_df1.where(F.col("attribute") == "ifa").toPandas().to_dict('list')['JSD'][0] == 0.0758   
    assert result_df1.where(F.col("attribute") == "ifa").toPandas().to_dict('list')['flagged'][0] == 1
    
    #testing using HD
    result_df2 = drift_statistics(test_df,test_df1,drop_cols=['income'],method_type='HD',bin_size=110,threshold = 0.033)
    
    assert result_df2.count() == 16
    assert result_df2.where(F.col("attribute") == "logfnl").toPandas().to_dict('list')['HD'][0] == 0.0395 
    assert result_df2.where(F.col("attribute") == "logfnl").toPandas().to_dict('list')['flagged'][0] == 1   
    assert result_df2.where(F.col("attribute") == "hours-per-week").toPandas().to_dict('list')['HD'][0] == 0.0346
    assert result_df2.where(F.col("attribute") == "hours-per-week").toPandas().to_dict('list')['flagged'][0] == 1
    assert result_df2.where(F.col("attribute") == "education").toPandas().to_dict('list')["HD"][0] == 0.016
    assert result_df2.where(F.col("attribute") == "education").toPandas().to_dict('list')["flagged"][0] == 0
    assert result_df2.where(F.col("attribute") == "race").toPandas().to_dict('list')['HD'][0] == 0.0126   
    assert result_df2.where(F.col("attribute") == "race").toPandas().to_dict('list')['flagged'][0] == 0
    
    #testing using KS
    result_df3 = drift_statistics(test_df,test_df1,drop_cols=['income','ifa'],method_type='KS',bin_method='equal_frequency',bin_size=110,threshold=0.01)
    
    assert result_df3.count() == 15
    assert result_df3.where(F.col("attribute") == "fnlwgt").toPandas().to_dict('list')['KS'][0] == 0.0117 
    assert result_df3.where(F.col("attribute") == "fnlwgt").toPandas().to_dict('list')['flagged'][0] == 1   
    assert result_df3.where(F.col("attribute") == "hours-per-week").toPandas().to_dict('list')['KS'][0] == 0.0115
    assert result_df3.where(F.col("attribute") == "hours-per-week").toPandas().to_dict('list')['flagged'][0] == 1
    assert result_df3.where(F.col("attribute") == "occupation").toPandas().to_dict('list')["KS"][0] == 0.0086
    assert result_df3.where(F.col("attribute") == "occupation").toPandas().to_dict('list')["flagged"][0] == 0
    assert result_df3.where(F.col("attribute") == "capital-gain").toPandas().to_dict('list')['KS'][0] == 0.0023   
    assert result_df3.where(F.col("attribute") == "capital-gain").toPandas().to_dict('list')['flagged'][0] == 0


    