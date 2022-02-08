import os

import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from anovos.data_ingest.data_ingest import (
    read_dataset,
    write_dataset,
    concatenate_dataset,
    join_dataset,
    delete_column,
    select_column,
    rename_column,
    recast_column,
)

sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = (
    "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
)
sample_avro = (
    "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
)
sample_output_path = "./data/tmp/output/"


@pytest.mark.usefixtures("spark_session")
def test_read_dataset(spark_session: SparkSession):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    assert df.where(F.col("ifa") == "27520a").count() == 1
    assert df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0] == 51
    assert (
        df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["education"][0]
        == "HS-grad"
    )


def test_write_dataset(spark_session: SparkSession):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    write_dataset(df, sample_output_path, "parquet", {"mode": "overwrite"})
    assert os.path.isfile(sample_output_path + "_SUCCESS")
    write_dataset(df, sample_output_path, "parquet", {"mode": "overwrite"}, None)
    assert os.path.isfile(sample_output_path + "_SUCCESS")
    write_dataset(df, sample_output_path, "parquet", {"mode": "overwrite"}, [])
    assert os.path.isfile(sample_output_path + "_SUCCESS")
    column_order1 = [
        "ifa",
        "age",
        "workclass",
        "fnlwgt",
        "logfnl",
        "education",
        "education-num",
        "marital-status",
        "income",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]
    write_dataset(
        df,
        file_path=sample_output_path,
        file_type="parquet",
        file_configs={"mode": "overwrite"},
        column_order=column_order1,
    )
    assert os.path.isfile(sample_output_path + "_SUCCESS")
    column_order2 = df.columns
    write_dataset(
        df, sample_output_path, "parquet", {"mode": "overwrite"}, column_order2
    )
    assert os.path.isfile(sample_output_path + "_SUCCESS")


def test_concatenate_dataset(spark_session: SparkSession):
    test_df = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )
    test_df2 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df2.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df2.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    idfs = [test_df, test_df2]
    concat_df = concatenate_dataset(*idfs, method_type="name")
    assert concat_df.where(F.col("ifa") == "27520a").count() == 2
    assert (
        concat_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        concat_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][1]
        == 51
    )
    assert (
        concat_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )
    assert (
        concat_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][1]
        == "HS-grad"
    )


def test_join_dataset(spark_session: SparkSession):
    test_df = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )
    test_df2 = spark_session.createDataFrame(
        [
            ("27520a", 177, "female"),
            ("10a", 182, "male"),
            ("11a", 155, "female"),
            ("1100b", 191, "male"),
        ],
        ["ifa", "height", "gender"],
    )
    assert test_df2.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["height"][0]
        == 177
    )
    assert (
        test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["gender"][0]
        == "female"
    )

    idfs = [test_df, test_df2]
    join_df = join_dataset(*idfs, join_cols="ifa", join_type="inner")
    assert join_df.where(F.col("ifa") == "27520a").count() == 1
    assert (
        join_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        join_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["height"][0]
        == 177
    )
    assert (
        join_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )
    assert (
        join_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["gender"][0]
        == "female"
    )


def test_delete_column(spark_session: SparkSession):
    test_df = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "HS-grad"),
            ("11a", 35, None, "HS-grad"),
            ("1100g", 33, 7500, "matric"),
            ("11d", 45, 9500, "HS-grad"),
            ("1100b", 23, None, "matric"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df = delete_column(test_df, list_of_cols=["income"])
    assert result_df.count() == 6
    assert len(result_df.columns) == 3

    list_of_columns = result_df.columns
    assert "income" not in list_of_columns
    assert "age" in list_of_columns
    assert "education" in list_of_columns
    assert "ifa" in list_of_columns
    assert (
        result_df.where(F.col("ifa") == "10a").toPandas().to_dict("list")["age"][0]
        == 42
    )


def test_select_column(spark_session: SparkSession):
    test_df = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "HS-grad"),
            ("11a", 35, None, "HS-grad"),
            ("1100g", 33, 7500, "matric"),
            ("11d", 45, 9500, "HS-grad"),
            ("1100b", 23, None, "matric"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df1 = select_column(test_df, list_of_cols=["ifa", "income"])
    assert result_df1.count() == 6
    assert len(result_df1.columns) == 2
    assert result_df1.select("ifa", "income")

    list_of_columns = result_df1.columns
    assert "age" not in list_of_columns
    assert "education" not in list_of_columns
    assert "income" in list_of_columns
    assert (
        result_df1.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["income"][0]
        == 9000
    )


def test_rename_column(spark_session: SparkSession):
    test_df = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "HS-grad"),
            ("11a", 35, None, "HS-grad"),
            ("1100g", 33, 7500, "matric"),
            ("11d", 45, 9500, "HS-grad"),
            ("1100b", 23, None, "matric"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df2 = rename_column(
        test_df, list_of_cols=["ifa", "income"], list_of_newcols=["id", "new_income"]
    )
    assert result_df2.count() == 6
    assert len(result_df2.columns) == 4
    list_of_columns = result_df2.columns
    assert "ifa" not in list_of_columns
    assert "income" not in list_of_columns

    assert (
        result_df2.where(F.col("id") == "27520a")
        .toPandas()
        .to_dict("list")["new_income"][0]
        == 9000
    )
    assert (
        result_df2.where(F.col("id") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )


def test_recast_column(spark_session: SparkSession):
    test_df1 = spark_session.createDataFrame(
        [
            ("27520a", "51", 9000, "HS-grad"),
            ("10a", "42", 7000, "HS-grad"),
            ("11a", "35", None, "HS-grad"),
            ("1100g", "33", 7500, "matric"),
            ("11d", "45", 9500, "HS-grad"),
            ("1100b", "23", None, "matric"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df1.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == "51"
    )
    assert (
        test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df1.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df3 = recast_column(test_df1, list_of_cols=["age"], list_of_dtypes=["long"])
    assert result_df3.count() == 6
    assert len(result_df3.columns) == 4
    assert (
        result_df3.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        result_df3.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )
