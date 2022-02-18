import pytest
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from anovos.data_analyzer.association_evaluator import (
    IV_calculation,
    IG_calculation,
    variable_clustering,
    correlation_matrix,
)

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
        result_df.where(F.col("attribute") == "relationship")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 1.6205
    )
    assert (
        result_df.where(F.col("attribute") == "marital-status")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 1.3929
    )
    assert (
        result_df.where(F.col("attribute") == "age").toPandas().to_dict("list")["iv"][0]
        == 1.1891
    )
    assert (
        result_df.where(F.col("attribute") == "occupation")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 0.7467
    )
    assert (
        result_df.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 0.7459
    )
    assert (
        result_df.where(F.col("attribute") == "education-num")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 0.7095
    )
    assert (
        result_df.where(F.col("attribute") == "hours-per-week")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 0.4441
    )
    assert (
        result_df.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 0.3179
    )
    assert (
        result_df.where(F.col("attribute") == "sex").toPandas().to_dict("list")["iv"][0]
        == 0.3106
    )
    assert (
        result_df.where(F.col("attribute") == "workclass")
        .toPandas()
        .to_dict("list")["iv"][0]
        == 0.1669
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
        result_df1.where(F.col("attribute") == "relationship")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.1702
    )
    assert (
        result_df1.where(F.col("attribute") == "marital-status")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.1583
    )
    assert (
        result_df1.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0943
    )
    assert (
        result_df1.where(F.col("attribute") == "occupation")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0917
    )
    assert (
        result_df1.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0873
    )
    assert (
        result_df1.where(F.col("attribute") == "education-num")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0888
    )
    assert (
        result_df1.where(F.col("attribute") == "hours-per-week")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0552
    )
    assert (
        result_df1.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0431
    )
    assert (
        result_df1.where(F.col("attribute") == "sex")
        .toPandas()
        .to_dict("list")["ig"][0]
        == 0.0379
    )
    assert (
        result_df1.where(F.col("attribute") == "workclass")
        .toPandas()
        .to_dict("list")["ig"][0]
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
        result_df2.where((F.col("cluster") == 1) & (F.col("attribute") == "fnlwgt"))
        .toPandas()
        .to_dict("list")["RS_Ratio"][0]
        == 0.2262
    )
    assert (
        result_df2.where((F.col("cluster") == 1) & (F.col("attribute") == "logfnl"))
        .toPandas()
        .to_dict("list")["RS_Ratio"][0]
        == 0.2257
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

    result_df3 = correlation_matrix(spark_session, test_df, drop_cols=["ifa"])

    assert result_df3.count() == 16
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["age"][0]
        == 1.0
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["native-country"][0]
        <= 0.2
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["capital-gain"][0]
        <= 0.25
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["capital-loss"][0]
        <= 0.25
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["education"][0]
        <= 0.4
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["education-num"][0]
        <= 0.4
    )
    assert (
        result_df3.where(F.col("attribute") == "age")
        .toPandas()
        .to_dict("list")["fnlwgt"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["capital-gain"][0]
        == 1.0
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["native-country"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["age"][0]
        <= 0.2
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["capital-loss"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["education"][0]
        <= 0.3
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["education-num"][0]
        <= 0.4
    )
    assert (
        result_df3.where(F.col("attribute") == "capital-gain")
        .toPandas()
        .to_dict("list")["fnlwgt"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["education"][0]
        == 1.0
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["native-country"][0]
        <= 0.45
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["age"][0]
        <= 0.4
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["capital-loss"][0]
        <= 0.2
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["capital-gain"][0]
        <= 0.3
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["education-num"][0]
        <= 1.0
    )
    assert (
        result_df3.where(F.col("attribute") == "education")
        .toPandas()
        .to_dict("list")["fnlwgt"][0]
        <= 0.1
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["label"][0]
        == 1.0
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["native-country"][0]
        <= 0.2
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["age"][0]
        <= 0.4
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["capital-loss"][0]
        <= 0.3
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["capital-gain"][0]
        <= 0.4
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["education"][0]
        <= 4.0
    )
    assert (
        result_df3.where(F.col("attribute") == "label")
        .toPandas()
        .to_dict("list")["fnlwgt"][0]
        <= 0.1
    )
