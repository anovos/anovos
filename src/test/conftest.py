import pathlib
import sys

import pytest

SRC_DIR = pathlib.Path(__file__).parent.parent / "main"
sys.path.insert(0, str(SRC_DIR.absolute()))

from anovos.shared.spark import SPARK_JARS_PACKAGES, init_spark


@pytest.fixture(scope="module")
def spark_session():
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
