import pytest
import os
from com.mw.ds.data_analyzer.quality_checker import *

sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
sample_avro = "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
sample_output_path = "./data/tmp/output/"

@pytest.mark.usefixtures("spark_session")

# def IDness_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=1.0, print_impact=False):
# def biasedness_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=1.0, print_impact=False):
# def outlier_detection(idf, list_of_cols='all', drop_cols=[], detection_side='upper', 
# def imputation_MMM(idf, list_of_cols="missing", drop_cols=[], method_type="median",
# def null_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_method='row_removal', 



def test_rows_wMissingFeats(spark_session):
    test_df = spark_session.createDataFrame(
        [
            ('27520a', 51, 9000, 'HS-grad'),
            ('10a', 42, 7000,'Postgrad'),
            ('11a', 35, None, None),
            ('1100b', 23, 6000, 'HS-grad')
        ],
        ['ifa', 'age', 'income','education']
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert test_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert test_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['income'][0] == 9000  
    assert test_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad' 

    result_df = rows_wMissingFeats(test_df,treatment = True,treatment_threshold=0.4)

    assert result_df[0].count() == 3
    assert result_df[1].where(F.col("null_feats_count") == 0).toPandas().to_dict('list')['row_count'][0] == 3   
    assert result_df[1].where(F.col("null_feats_count") == 0).toPandas().to_dict('list')['row_pct'][0] == 0.75
    assert result_df[1].where(F.col("null_feats_count") == 0).toPandas().to_dict('list')['flagged'][0] == 0
    assert result_df[1].where(F.col("null_feats_count") == 2).toPandas().to_dict('list')['row_count'][0] == 1   
    assert result_df[1].where(F.col("null_feats_count") == 2).toPandas().to_dict('list')['row_pct'][0] == 0.25
    assert result_df[1].where(F.col("null_feats_count") == 2).toPandas().to_dict('list')['flagged'][0] == 1 
    
def test_duplicate_detection(spark_session):
    test_df1 = spark_session.createDataFrame(
        [
            ('27520a', 51, 9000, 'HS-grad'),
            ('10a', 42, 7000,'Postgrad'),
            ('10a', 42, 7000,'Postgrad'),
            ('11a', 35, None, None),
            ('1100b', 23, 6000, 'HS-grad')
        ],
        ['ifa', 'age', 'income','education']
    )
    assert test_df1.where(F.col("ifa") == "27520a").count() == 1
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['income'][0] == 9000  
    assert test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad' 

    result_df1 = duplicate_detection(test_df1,treatment = True)

    assert result_df1[0].count() == 4
    assert result_df1[1].where(F.col("metric") == "rows_count").toPandas().to_dict('list')['value'][0] == 5   
    assert result_df1[1].where(F.col("metric") == "unique_rows_count").toPandas().to_dict('list')['value'][0] == 4
    
def test_invalidEntries_detection(spark_session):
    test_df2 = spark_session.createDataFrame(
        [
            ('27520a', 51, 9000, 'HS-grad'),
            ('10a', 42, 7000,'Postgrad'),
            ('10a', 9999, 7000,'Postgrad'),
            ('11a', 35, None, ':'),
            ('1100b', 23, 6000, 'HS-grad')
        ],
        ['ifa', 'age', 'income','education']
    )
    assert test_df2.where(F.col("ifa") == "27520a").count() == 1
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['income'][0] == 9000  
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad' 

    result_df2 = invalidEntries_detection(test_df2,treatment = True)

    assert result_df2[0].count() == 5
    assert result_df2[1].where(F.col("feature") == "age").toPandas().to_dict('list')['invalid_count'][0] == 1 
    assert result_df2[1].where(F.col("feature") == "age").toPandas().to_dict('list')['invalid_pct'][0] == 0.2
    assert result_df2[1].where(F.col("feature") == "education").toPandas().to_dict('list')['invalid_count'][0] == 1 
    assert result_df2[1].where(F.col("feature") == "education").toPandas().to_dict('list')['invalid_pct'][0] == 0.2
    
    



