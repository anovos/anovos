import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from anovos.data_analyzer.quality_checker import (
    duplicate_detection,
    invalidEntries_detection,
    IDness_detection,
    biasedness_detection,
    nullColumns_detection,
    nullRows_detection,
    outlier_detection,
)
from anovos.data_transformer.transformers import imputation_MMM

sample_parquet = "./data/test_dataset/part-00000-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = (
    "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
)
sample_avro = (
    "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
)
sample_output_path = "./data/tmp/output/"


@pytest.mark.usefixtures("spark_session")
def test_nullRows_detection(spark_session: SparkSession):
    test_df = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "Postgrad"),
            ("11a", 35, None, None),
            ("1100b", 23, 6000, "HS-grad"),
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

    result_df = nullRows_detection(
        spark_session, test_df, treatment=True, treatment_threshold=0.4
    )

    assert result_df[0].count() == 3
    assert (
        result_df[1]
        .where(F.col("null_cols_count") == 0)
        .toPandas()
        .to_dict("list")["row_count"][0]
        == 3
    )
    assert (
        result_df[1]
        .where(F.col("null_cols_count") == 0)
        .toPandas()
        .to_dict("list")["row_pct"][0]
        == 0.75
    )
    assert (
        result_df[1]
        .where(F.col("null_cols_count") == 0)
        .toPandas()
        .to_dict("list")["treated"][0]
        == 0
    )
    assert (
        result_df[1]
        .where(F.col("null_cols_count") == 2)
        .toPandas()
        .to_dict("list")["row_count"][0]
        == 1
    )
    assert (
        result_df[1]
        .where(F.col("null_cols_count") == 2)
        .toPandas()
        .to_dict("list")["row_pct"][0]
        == 0.25
    )
    assert (
        result_df[1]
        .where(F.col("null_cols_count") == 2)
        .toPandas()
        .to_dict("list")["treated"][0]
        == 1
    )


def test_duplicate_detection(spark_session: SparkSession):
    test_df1 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "Postgrad"),
            ("10a", 42, 7000, "Postgrad"),
            ("11a", 35, None, None),
            ("1100b", 23, 6000, "HS-grad"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df1.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df1.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
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

    result_df1 = duplicate_detection(spark_session, test_df1, treatment=True)

    assert result_df1[0].count() == 4
    assert (
        result_df1[1]
        .where(F.col("metric") == "rows_count")
        .toPandas()
        .to_dict("list")["value"][0]
        == 5
    )
    assert (
        result_df1[1]
        .where(F.col("metric") == "unique_rows_count")
        .toPandas()
        .to_dict("list")["value"][0]
        == 4
    )
    assert (
        result_df1[1]
        .where(F.col("metric") == "duplicate_rows")
        .toPandas()
        .to_dict("list")["value"][0]
        == 1
    )
    assert (
        result_df1[1]
        .where(F.col("metric") == "duplicate_pct")
        .toPandas()
        .to_dict("list")["value"][0]
        == 0.20
    )


def test_invalidEntries_detection(spark_session: SparkSession):
    test_df2 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "Postgrad"),
            ("10a", 9999, 7000, "Postgrad"),
            ("11a", 35, None, ":"),
            ("1100b", 23, 6000, "HS-grad"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df2.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df2.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df2.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df2 = invalidEntries_detection(spark_session, test_df2, treatment=True)

    assert result_df2[0].count() == 5
    assert (
        result_df2[1]
        .where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["invalid_count"][0]
        == 1
    )
    assert (
        result_df2[1]
        .where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["invalid_pct"][0]
        == 0.2
    )
    assert (
        result_df2[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["invalid_count"][0]
        == 1
    )
    assert (
        result_df2[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["invalid_pct"][0]
        == 0.2
    )


def test_IDness_detection(spark_session: SparkSession):
    test_df3 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "Postgrad"),
            ("11a", 35, None, "graduate"),
            ("1100b", 23, 6000, "matric"),
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

    result_df3 = IDness_detection(
        spark_session,
        test_df3,
        drop_cols=["ifa"],
        treatment=False,
        treatment_threshold=1.0,
    )

    assert len(result_df3[0].columns) == 4
    assert (
        result_df3[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["unique_values"][0]
        == 4
    )
    assert (
        result_df3[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["IDness"][0]
        == 1.0
    )
    assert (
        result_df3[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["flagged"][0]
        == 1
    )

    result_df3 = IDness_detection(
        spark_session,
        test_df3,
        drop_cols=["ifa"],
        treatment=True,
        treatment_threshold=1.0,
    )

    assert len(result_df3[0].columns) == 1
    assert (
        result_df3[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["unique_values"][0]
        == 4
    )
    assert (
        result_df3[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["IDness"][0]
        == 1.0
    )
    assert (
        result_df3[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["treated"][0]
        == 1
    )


def test_biasedness_detection(spark_session: SparkSession):
    test_df4 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "HS-grad"),
            ("11a", 35, None, "HS-grad"),
            ("11d", 45, 9500, "HS-grad"),
            ("1100b", 23, 6000, "matric"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df4.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df4.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df4.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df4.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df4 = biasedness_detection(
        spark_session, test_df4, treatment=False, treatment_threshold=0.8
    )

    assert len(result_df4[0].columns) == 4
    assert (
        result_df4[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode"][0]
        == "HS-grad"
    )
    assert (
        result_df4[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode_pct"][0]
        == 0.8
    )
    assert (
        result_df4[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["flagged"][0]
        == 1
    )

    result_df4 = biasedness_detection(
        spark_session, test_df4, treatment=True, treatment_threshold=0.8
    )

    assert len(result_df4[0].columns) == 3
    assert (
        result_df4[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode"][0]
        == "HS-grad"
    )
    assert (
        result_df4[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["mode_pct"][0]
        == 0.8
    )
    assert (
        result_df4[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["treated"][0]
        == 1
    )


def test_imputation_MMM(spark_session: SparkSession):
    test_df5 = spark_session.createDataFrame(
        [
            ("27520a", 51, 8000, "HS-grad"),
            ("10a", 42, 7000, "HS-grad"),
            ("10b", 34, 6000, "grad"),
            ("10c", 29, 9000, "HS-grad"),
            ("11a", 35, None, None),
            ("1100b", 23, 9000, "Postgrad"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df5.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df5.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df5.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 8000
    )
    assert (
        test_df5.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df5 = imputation_MMM(spark_session, test_df5)

    assert result_df5.count() == 6
    assert (
        result_df5.where(F.col("ifa") == "11a").toPandas().to_dict("list")["income"][0]
        == 8000
    )
    assert (
        result_df5.where(F.col("ifa") == "11a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )


def test_nullColumns_detection(spark_session: SparkSession):
    test_df6 = spark_session.createDataFrame(
        [
            ("27520a", 51, 9000, "HS-grad"),
            ("10a", 42, 7000, "Postgrad"),
            ("11a", 35, None, None),
            ("1100b", 23, 6000, "HS-grad"),
        ],
        ["ifa", "age", "income", "education"],
    )
    assert test_df6.where(F.col("ifa") == "27520a").count() == 1
    assert (
        test_df6.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["age"][0]
        == 51
    )
    assert (
        test_df6.where(F.col("ifa") == "27520a").toPandas().to_dict("list")["income"][0]
        == 9000
    )
    assert (
        test_df6.where(F.col("ifa") == "27520a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "HS-grad"
    )

    result_df6 = nullColumns_detection(spark_session, test_df6, treatment=True)

    assert len(result_df6[0].columns) == 4
    assert result_df6[0].count() == 3
    assert (
        result_df6[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["missing_count"][0]
        == 1
    )
    assert (
        result_df6[1]
        .where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["missing_pct"][0]
        == 0.25
    )
    assert (
        result_df6[1]
        .where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["missing_count"][0]
        == 1
    )
    assert (
        result_df6[1]
        .where(F.col("attribute") == "income")
        .toPandas()
        .to_dict("list")["missing_pct"][0]
        == 0.25
    )


def test_outlier_detection(spark_session: SparkSession):
    test_df7 = spark_session.read.parquet(sample_parquet)
    test_df7 = test_df7.withColumn(
        "label",
        F.when(F.col("income") == "<=50K", F.lit(0.0)).when(
            F.col("income") == ">50K", F.lit(1.0)
        ),
    ).drop("income")
    assert test_df7.where(F.col("ifa") == "4062a").count() == 1
    assert (
        test_df7.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["age"][0]
        == 28
    )
    assert (
        test_df7.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["sex"][0]
        == "Male"
    )
    assert (
        test_df7.where(F.col("ifa") == "4062a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "11th"
    )

    result_df7 = outlier_detection(spark_session, test_df7, drop_cols=["ifa", "label"])[
        1
    ]

    assert result_df7.count() == 7
    assert len(result_df7.columns) == 3
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 87
    )
    assert (
        result_df7.where(F.col("attribute") == "fnlwgt")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "fnlwgt")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 518
    )
    assert (
        result_df7.where(F.col("attribute") == "logfnl")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "logfnl")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 15
    )
    assert (
        result_df7.where(F.col("attribute") == "education-num")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "education-num")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 955
    )
    assert (
        result_df7.where(F.col("attribute") == "capital-loss")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "capital-loss")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 790
    )
    assert (
        result_df7.where(F.col("attribute") == "hours-per-week")
        .toPandas()
        .to_dict("list")["lower_outliers"][0]
        == 0
    )
    assert (
        result_df7.where(F.col("attribute") == "hours-per-week")
        .toPandas()
        .to_dict("list")["upper_outliers"][0]
        == 515
    )
