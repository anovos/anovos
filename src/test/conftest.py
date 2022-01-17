import pathlib
import sys

import findspark
import pytest

findspark.init()
from pyspark.sql import SparkSession

SRC_DIR = pathlib.Path(__file__).parent.parent / 'main'
sys.path.insert(0, str(SRC_DIR.absolute()))

from anovos.shared.spark import SPARK_JARS_PACKAGES


@pytest.fixture(scope="session")
def spark_session():
    spark_builder = SparkSession.builder.master("local[*]").appName("anovos_test")
    spark_builder.config('spark.jars.packages', ','.join(list(SPARK_JARS_PACKAGES)))
    spark = spark_builder.getOrCreate()
    return spark
