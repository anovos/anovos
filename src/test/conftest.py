import pytest
from pyspark.sql import SparkSession

from anovos.shared.spark import init_spark, SPARK_JARS_PACKAGES


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    _spark, _spark_context, _sql_context = init_spark(app_name="anovos_test")
    _spark.config("spark.jars.packages", ",".join(list(SPARK_JARS_PACKAGES)))
    return _spark
