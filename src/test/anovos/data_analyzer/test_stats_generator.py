import pytest
from pyspark.sql import functions as F

from anovos.data_analyzer.stats_generator import (
    missingCount_computation,
    uniqueCount_computation,
    mode_computation,
    nonzeroCount_computation,
    measures_of_centralTendency,
    measures_of_cardinality,
    measures_of_dispersion,
    measures_of_counts,
    measures_of_shape,
    global_summary,
    measures_of_percentiles,
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
def test_missingCount_computation(spark_session):
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

    result_df = missingCount_computation(spark_session, test_df)

    assert result_df.count() == 3
    assert (
        result_df.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["missing_count"][0]
        == 1
    )
    assert (
        result_df.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["missing_pct"][0]
        == 0.25
    )


def test_uniqueCount_computation(spark_session):
    test_df1 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df1.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df1.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df1 = uniqueCount_computation(spark_session, test_df1)
    assert result_df1.count() == 3
    assert (
        result_df1.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["unique_values"][0]
        == 2
    )
    assert (
        result_df1.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["unique_values"][0]
        == 4
    )


def test_mode_computation(spark_session):
    test_df2 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("13a", 42, "HS-grad"),
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

    result_df2 = mode_computation(spark_session, test_df2)
    assert result_df2.count() == 3
    assert (
        result_df2.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode"][0]
        == "HS-grad"
    )
    assert (
        result_df2.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode_rows"][0]
        == 3
    )
    assert (
        result_df2.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["mode"][0]
        == "42"
    )
    assert (
        result_df2.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["mode_rows"][0]
        == 2
    )


def test_nonzeroCount_computation(spark_session):
    test_df3 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "Postgrad"),
            ("11a", 0, None, None),
            ("1100b", 23, 6000, "HS-grad"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df3.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df3.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df3.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df3.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df3 = nonzeroCount_computation(spark_session, test_df3)
    assert result_df3.count() == 2
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["nonzero_count"][0]
        == 3
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["nonzero_pct"][0]
        == 0.75
    )
    assert (
        result_df3.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["nonzero_count"][0]
        == 3
    )
    assert (
        result_df3.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["nonzero_pct"][0]
        == 0.75
    )


def test_measures_of_centralTendency(spark_session):
    test_df4 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df4.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df4.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df4.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df4 = measures_of_centralTendency(spark_session, test_df4)
    assert result_df4.count() == 3
    assert (
        result_df4.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode"][0]
        == "HS-grad"
    )
    assert (
        result_df4.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["mean"][0]
        == 42.75
    )
    assert (
        result_df4.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["median"][0]
        == 42.0
    )
    assert (
        result_df4.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode_pct"][0]
        == 0.6667
    )


def test_measures_of_cardinality(spark_session):
    test_df5 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df5.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df5.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df5.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df5 = measures_of_cardinality(spark_session, test_df5)
    assert result_df5.count() == 3
    assert (
        result_df5.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["unique_values"][0]
        == 2
    )
    assert (
        result_df5.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["unique_values"][0]
        == 4
    )
    assert (
        result_df5.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["IDness"][0]
        == 1.0
    )
    assert (
        result_df5.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["IDness"][0]
        == 0.6667
    )


def test_measures_of_dispersion(spark_session):
    test_df6 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df6.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df6.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df6.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df6 = measures_of_dispersion(spark_session, test_df6)
    assert result_df6.count() == 1
    assert (
        result_df6.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["stddev"][0]
        == 14.2449
    )
    assert (
        result_df6.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["variance"][0]
        == 202.9172
    )
    assert (
        result_df6.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["cov"][0]
        == 0.3332
    )
    assert (
        result_df6.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["IQR"][0]
        == 28.0
    )
    assert (
        result_df6.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["range"][0]
        == 32.0
    )


# def measures_of_percentiles(idf, list_of_cols='all', drop_cols=[], print_impact=False):
def test_measures_of_counts(spark_session):
    test_df7 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df7.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df7.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df7.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df7 = measures_of_counts(spark_session, test_df7)
    assert result_df7.count() == 3
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["fill_count"][0]
        == 4
    )
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["fill_pct"][0]
        == 1.0
    )
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["missing_count"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["missing_pct"][0]
        == 0.0
    )
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["nonzero_count"][0]
        == 4
    )
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["nonzero_pct"][0]
        == 1.0
    )


def test_measures_of_shape(spark_session):
    test_df8 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df8.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df8.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df8.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df8 = measures_of_shape(spark_session, test_df8)
    assert result_df8.count() == 1
    assert (
        result_df8.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["skewness"][0]
        == -0.7063
    )
    assert (
        result_df8.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["kurtosis"][0]
        == -1.0646
    )


def test_global_summary(spark_session):
    test_df9 = spark_session.createDataFrame(
        [
            ("27520a", 51, "HS-grad"),
            ("10a", 42, "Postgrad"),
            ("11a", 55, None),
            ("1100b", 23, "HS-grad"),
        ],
        ["ifa", "age", "education"],
    )
    assert test_df9.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df9.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df9.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df9 = global_summary(spark_session, test_df9)
    assert result_df9.count() == 8
    assert (
        result_df9.where(F.col("metric") == "rows_count")
        .toPandas()
        .to_dict("list")["value"][0]
        == "4"
    )
    assert (
        result_df9.where(F.col("metric") == "columns_count")
        .toPandas()
        .to_dict("list")["value"][0]
        == "3"
    )
    assert (
        result_df9.where(F.col("metric") == "numcols_count")
        .toPandas()
        .to_dict("list")["value"][0]
        == "1"
    )
    assert (
        result_df9.where(F.col("metric") == "numcols_name")
        .toPandas()
        .to_dict("list")["value"][0]
        == "age"
    )
    assert (
        result_df9.where(F.col("metric") == "catcols_count")
        .toPandas()
        .to_dict("list")["value"][0]
        == "2"
    )


def test_measures_of_percentiles(spark_session):
    test_df10 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "HS-grad"),
            ("11a", 35, None, "HS-grad"),
            ("1100g", 33, 7500, "matric"),
            ("11d", 45, 9500, "HS-grad"),
            ("1100b", 23, 6000, "matric"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df10.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df10.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df10.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df10.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df10 = measures_of_percentiles(spark_session, test_df10)
    assert result_df10.count() == 2
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["min"][0]
        == 6000.0
    )
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["10%"][0]
        <= 6000.0
    )
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["25%"][0]
        <= 7000.0
    )
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["50%"][0]
        <= 7500.0
    )
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["75%"][0]
        <= 9000.0
    )
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["90%"][0]
        <= 9500.0
    )
    assert (
        result_df10.where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["max"][0]
        == 9500.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["min"][0]
        == 23.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["10%"][0]
        <= 23.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["25%"][0]
        <= 33.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["50%"][0]
        <= 35.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["75%"][0]
        <= 45.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["90%"][0]
        <= 51.0
    )
    assert (
        result_df10.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["max"][0]
        == 51.0
    )
