import pytest
import os
from com.mw.ds.data_analyzer.stats_generator import *

sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
sample_avro = "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
sample_output_path = "./data/tmp/output/"

@pytest.mark.usefixtures("spark_session")

def test_missingCount_computation(spark_session):
    test_df = spark_session.createDataFrame(
        [
            ('27520a', 51, 'HS-grad'),
            ('10a', 42, 'Postgrad'),
            ('11a', 55, None),
            ('1100b', 23, 'HS-grad')
        ],
        ['ifa', 'age', 'education']
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert test_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51   
    assert test_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad' 

    result_df = missingCount_computation(test_df)

    assert result_df.count() == 3
    assert result_df.where(F.col("feature") == "education").toPandas().to_dict('list')['missing_count'][0] == 1   
    assert result_df.where(F.col("feature") == "education").toPandas().to_dict('list')['missing_pct'][0] == 0.25   


def test_uniqueCount_computation(spark_session):
    test_df1 = spark_session.createDataFrame(
        [
            ('27520a', 51, 'HS-grad'),
            ('10a', 42, 'Postgrad'),
            ('11a', 55, None),
            ('1100b', 23, 'HS-grad')
        ],
        ['ifa', 'age', 'education']
    )
    assert test_df1.where(F.col("ifa") == "27520a").count() == 1
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51   
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad'
    
    result_df1 = uniqueCount_computation(test_df1)
    assert result_df1.count() == 3
    assert result_df1.where(F.col("feature") == "education").toPandas().to_dict('list')['unique_values'][0] == 2 
    assert result_df1.where(F.col("feature") == "age").toPandas().to_dict('list')['unique_values'][0] == 4   

def test_mode_computation(spark_session):
    test_df2 = spark_session.createDataFrame(
        [
            ('27520a', 51, 'HS-grad'),
            ('10a', 42, 'Postgrad'),
            ('11a', 55, None),
            ('1100b', 23, 'HS-grad')
        ],
        ['ifa', 'age', 'education']
    )
    assert test_df2.where(F.col("ifa") == "27520a").count() == 1
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51   
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad'
    
    result_df2 = mode_computation(test_df2)
    assert result_df2.count() == 3
    assert result_df2.where(F.col("feature") == "education").toPandas().to_dict('list')['mode'][0] == 'HS-grad' 
    assert result_df2.where(F.col("feature") == "education").toPandas().to_dict('list')['mode_pct'][0] == 0.6667  
    
# def nonzeroCount_computation(idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def measures_of_centralTendency(idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def measures_of_cardinality(idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def measures_of_dispersion(idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def measures_of_percentiles(idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def measures_of_counts (idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def measures_of_shape(idf, list_of_cols='all', drop_cols=[], print_impact=False):
# def global_summary(idf, list_of_cols='all', drop_cols=[], print_impact=True):
# def descriptive_stats(idf, list_of_cols='all', drop_cols=[], print_impact=True):
