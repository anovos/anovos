import findspark
import pytest
from packaging import version

findspark.init()

import pyspark

if version.parse(pyspark.__version__) < version.parse("3.0.0"):
    SPARK_JARS_PACKAGES = ["io.github.histogrammar:histogrammar_2.11:1.0.20",
                           "io.github.histogrammar:histogrammar-sparksql_2.11:1.0.20",
                           "org.apache.spark:spark-avro_2.11:2.4.0"]
else:
    SPARK_JARS_PACKAGES = ["io.github.histogrammar:histogrammar_2.12:1.0.20",
                           "io.github.histogrammar:histogrammar-sparksql_2.12:1.0.20",
                           "org.apache.spark:spark-avro_2.12:3.1.2"]

from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    spark_builder = SparkSession.builder.master("local[*]").appName("anovos_test")
    spark_builder.config('spark.jars.packages', ','.join(list(SPARK_JARS_PACKAGES)))
    spark = spark_builder.getOrCreate()
    return spark
