import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from anovos.data_analyzer.association_evaluator import (
    IG_calculation,
    IV_calculation,
    correlation_matrix,
    variable_clustering,
)

from anovos.shared.utils import attributeType_segregation

sample_parquet = "./data/test_dataset/part-00000-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"
sample_csv = (
    "./data/test_dataset/part-00000-8beb3930-8a44-4b7b-906b-a6deca466d9f-c000.csv"
)
sample_avro = (
    "./data/test_dataset/part-00000-f12ee684-956d-487d-b781-fb99af447b34-c000.avro"
)
sample_output_path = "./data/tmp/output/"


@pytest.mark.usefixtures("spark_session")
def test_IV_calculation(spark_session: SparkSession):
    test_df = spark_session.read.parquet(sample_parquet)
    test_df = test_df.withColumn(
        "label",
        F.when(F.col("income") == "<=50K", F.lit(0.0)).when(
            F.col("income") == ">50K", F.lit(1.0)
        ),
    ).drop("income")
    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["age"][0]
        == 28
    )
    assert (
        test_df.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["sex"][0]
        == "Male"
    )
    assert (
        test_df.where(F.col("ifa") == "4062a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "11th"
    )

    result_df = IV_calculation(spark_session, test_df, drop_cols=["ifa"])

    assert result_df.count() == 15
    assert (
        round(
            result_df.where(F.col("attribute") == "relationship")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 1.6208
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "marital-status")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 1.3929
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "age")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 1.1891
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "occupation")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.7686
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "education")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.7525
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "education-num")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.7095
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "hours-per-week")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.4441
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "capital-gain")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.3184
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "sex")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.3111
    )
    assert (
        round(
            result_df.where(F.col("attribute") == "workclass")
            .toPandas()
            .to_dict("list")["iv"][0],
            4,
        )
        == 0.1686
    )


def test_IG_calculation(spark_session: SparkSession):
    test_df = spark_session.read.parquet(sample_parquet)
    test_df = test_df.withColumn(
        "label",
        F.when(F.col("income") == "<=50K", F.lit(0.0)).when(
            F.col("income") == ">50K", F.lit(1.0)
        ),
    ).drop("income")
    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["age"][0]
        == 28
    )
    assert (
        test_df.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["sex"][0]
        == "Male"
    )
    assert (
        test_df.where(F.col("ifa") == "4062a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "11th"
    )

    result_df1 = IG_calculation(spark_session, test_df, drop_cols=["ifa"])

    assert result_df1.count() == 15
    assert (
        round(
            result_df1.where(F.col("attribute") == "relationship")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.1702
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "marital-status")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.1608
    )

    assert (
        round(
            result_df1.where(F.col("attribute") == "age")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0944
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "occupation")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0916
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "education")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0938
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "education-num")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0887
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "hours-per-week")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0549
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "capital-gain")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0434
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "sex")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0379
    )
    assert (
        round(
            result_df1.where(F.col("attribute") == "workclass")
            .toPandas()
            .to_dict("list")["ig"][0],
            4,
        )
        == 0.0223
    )


def test_variable_clustering(spark_session: SparkSession):
    test_df = spark_session.read.parquet(sample_parquet)
    test_df = test_df.withColumn(
        "label",
        F.when(F.col("income") == "<=50K", F.lit(0.0)).when(
            F.col("income") == ">50K", F.lit(1.0)
        ),
    ).drop("income")
    test_df = test_df.withColumn("engagement", F.lit(1))

    assert test_df.where(F.col("ifa") == "4062a").count() == 1
    assert (
        test_df.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["age"][0]
        == 28
    )
    assert (
        test_df.where(F.col("ifa") == "4062a").toPandas().to_dict("list")["sex"][0]
        == "Male"
    )
    assert (
        test_df.where(F.col("ifa") == "4062a")
        .toPandas()
        .to_dict("list")["engagement"][0]
        == 1
    )
    assert (
        test_df.where(F.col("ifa") == "4062a")
        .toPandas()
        .to_dict("list")["education"][0]
        == "11th"
    )

    result_df2 = variable_clustering(spark_session, test_df, drop_cols=["ifa", "label"])

    assert result_df2.count() == 15
    assert len(result_df2.columns) == 3
    assert (
            result_df2.where(
                (F.col("cluster") == 0) & (F.col("attribute") == "relationship")
            )
            .toPandas()
            .to_dict("list")["RS_Ratio"][0]
            == 0.3409
    )
    assert (
            result_df2.where((F.col("cluster") == 0) & (F.col("attribute") == "sex"))
            .toPandas()
            .to_dict("list")["RS_Ratio"][0]
            == 0.3378
    )
    assert (
            result_df2.where(
                (F.col("cluster") == 0) & (F.col("attribute") == "marital-status")
            )
            .toPandas()
            .to_dict("list")["RS_Ratio"][0]
            == 0.4693
    )
    assert (
            result_df2.where(
                (F.col("cluster") == 0) & (F.col("attribute") == "hours-per-week")
            )
            .toPandas()
            .to_dict("list")["RS_Ratio"][0]
            == 0.8106
    )
    assert (
        result_df2.where(
            (F.col("cluster") == 2) & (F.col("attribute") == "capital-loss")
        )
        .toPandas()
        .to_dict("list")["RS_Ratio"][0]
        == 0.9185
    )
    assert (
            result_df2.where(
                (F.col("cluster") == 2) & (F.col("attribute") == "education-num")
            )
            .toPandas()
            .to_dict("list")["RS_Ratio"][0]
            == 0.3483
    )
    assert (
            result_df2.where((F.col("cluster") == 2) & (F.col("attribute") == "occupation"))
            .toPandas()
            .to_dict("list")["RS_Ratio"][0]
            == 0.3690
    )


def test_correlation_matrix(spark_session: SparkSession):
    test_df = spark_session.read.parquet(sample_parquet)

    num_cols, cat_cols, other_cols = attributeType_segregation(test_df)

    result_df3 = correlation_matrix(
        spark_session, test_df, list_of_cols=num_cols, drop_cols=[]
    )

    assert result_df3.count() == 7
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["age"][0]
        == 1.0
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["capital-gain"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["capital-loss"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["education-num"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "education-num")
        .toPandas()
        .to_dict("list")["capital-gain"][0]
        <= 0.2
    )
    assert (
        result_df3.where(F.col("attribute") == "education-num")
        .toPandas()
        .to_dict("list")["capital-loss"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "fnlwgt")
        .toPandas()
        .to_dict("list")["logfnl"][0]
        <= 0.95
    )
    assert (
        result_df3.where(F.col("attribute") == "hours-per-week")
        .toPandas()
        .to_dict("list")["education-num"][0]
        <= 0.2
    )
    assert (
        result_df3.where(F.col("attribute") == "logfnl")
        .toPandas()
        .to_dict("list")["education-num"][0]
        <= 0.1
    )
