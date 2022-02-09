import pathlib
import sys

import pytest
from pyspark.sql import SparkSession

SRC_DIR = pathlib.Path(__file__).parent.parent / "main"
sys.path.insert(0, str(SRC_DIR.absolute()))

from anovos.shared.spark import init_spark, SPARK_JARS_PACKAGES


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    configs = {
        "app_name": "Anovos_test_pipeline",
        "jars_packages": SPARK_JARS_PACKAGES,
        "py_files": [],
        "spark_config": {
            "spark.sql.session.timeZone": "GMT",
            "spark.python.profile": "false",
        },
    }
    _spark, _spark_context, _sql_context = init_spark(**configs)
    return _spark
