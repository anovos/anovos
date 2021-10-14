import os

import pytest
from com.mw.ds.data_ingest.data_ingest import *

sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
sample_avro = "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
sample_output_path = "./data/tmp/output/"


@pytest.mark.usefixtures("spark_session")
# class TestDataIngest(object):
def test_read_dataset():
    df = read_dataset(sample_parquet, "parquet")
    assert df.where(F.col("ifa") == "27520a").count() == 1
    assert df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad'
    # df2 = read_dataset(sample_avro, "avro")
    # assert df2.where(F.col("ifa") == "10a").count() == 1
    # assert df2.where(F.col("ifa") == "10a").toPandas().to_dict('list')['age'][0] == 42


def test_write_dataset():
    df = read_dataset(sample_parquet, "parquet")
    write_dataset(df, sample_output_path, "parquet")
    assert os.path.isfile(sample_output_path + "_SUCCESS")


def test_concatenate_dataset(spark_session):
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

    idfs = [test_df]
    idfs.append(test_df2)
    concat_df = concatenate_dataset(*idfs, method_type='name')
    assert concat_df.where(F.col("ifa") == "27520a").count() == 2
    assert concat_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert concat_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][1] == 51
    assert concat_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad'
    assert concat_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][1] == 'HS-grad'


def test_join_dataset(spark_session):
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
    test_df2 = spark_session.createDataFrame(
        [
            ('27520a', 177, 'female'),
            ('10a', 182, 'male'),
            ('11a', 155, 'female'),
            ('1100b', 191, 'male')
        ],
        ['ifa', 'height', 'gender']
    )
    assert test_df2.where(F.col("ifa") == "27520a").count() == 1
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['height'][0] == 177
    assert test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['gender'][0] == 'female'

    idfs = [test_df]
    idfs.append(test_df2)
    join_df = join_dataset(*idfs, join_cols='ifa', join_type='inner')
    assert join_df.where(F.col("ifa") == "27520a").count() == 1
    assert join_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['age'][0] == 51
    assert join_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['height'][0] == 177
    assert join_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['education'][0] == 'HS-grad'
    assert join_df.where(F.col("ifa") == "27520a").toPandas().to_dict('list')['gender'][0] == 'female'

###   def read_dataset(file_path, file_type, file_configs={}):
###   def write_dataset(idf, file_path, file_type, file_configs={}):
###   def concatenate_dataset(*idfs,method_type='name'):
###   def join_dataset(*idfs,join_cols,join_type):
#   def delete_column(idf,list_of_cols, print_impact=False):
#   def select_column(idf,list_of_cols, print_impact=False):
#   def rename_column(idf,list_of_cols, list_of_newcols, print_impact=False):
#   def recast_column(idf, list_of_cols, list_of_dtypes, print_impact=False):
#
