from os import environ

import __main__
import findspark
from loguru import logger
from packaging import version

findspark.init()
import pyspark
from pyspark.sql import SparkSession, SQLContext

if version.parse(pyspark.__version__) < version.parse("3.0.0"):
    SPARK_JARS_PACKAGES = [
        "io.github.histogrammar:histogrammar_2.11:1.0.20",
        "io.github.histogrammar:histogrammar-sparksql_2.11:1.0.20",
        "org.apache.spark:spark-avro_2.11:" + str(pyspark.__version__),
    ]
else:
    SPARK_JARS_PACKAGES = [
        "io.github.histogrammar:histogrammar_2.12:1.0.20",
        "io.github.histogrammar:histogrammar-sparksql_2.12:1.0.20",
        "org.apache.spark:spark-avro_2.12:" + str(pyspark.__version__),
    ]


def init_spark(
    app_name="anovos",
    master="local[*]",
    jars_packages=None,
    py_files=None,
    spark_config=None,
):
    """

    Parameters
    ----------
    app_name
        Name of Spark app. (Default value = "anovos")
    master
        Cluster connection details
        Defaults to local[*] which means to run Spark locally with as many worker threads
        as logical cores on the machine.
    jars_packages
        List of Spark JAR package names. (Default value = None)
    py_files
        List of files to send to Spark cluster (master and workers). (Default value = None)
    spark_config
        Dictionary of config key-value pairs. (Default value = None)

    Returns
    -------

    """
    logger.info(f"Getting spark session, context and sql context app_name: {app_name}")

    # detect execution environment
    flag_repl = not (hasattr(__main__, "__file__"))
    flag_debug = "DEBUG" in environ.keys()

    if not (flag_repl or flag_debug):
        spark_builder = SparkSession.builder.appName(app_name)
    else:
        spark_builder = SparkSession.builder.master(master).appName(app_name)

    if jars_packages is not None and jars_packages:
        spark_jars_packages = ",".join(list(jars_packages))
        spark_builder.config("spark.jars.packages", spark_jars_packages)

    if py_files is not None and py_files:
        spark_files = ",".join(list(py_files))
        spark_builder.config("spark.files", spark_files)

    if spark_config is not None and spark_config:
        for key, val in spark_config.items():
            spark_builder.config(key, val)

    _spark = spark_builder.getOrCreate()
    _spark_context = _spark.sparkContext
    _sql_context = SQLContext(_spark_context)

    return _spark, _spark_context, _sql_context


configs = {
    "app_name": "Anovos_pipeline",
    "jars_packages": SPARK_JARS_PACKAGES,
    "py_files": [],
    "spark_config": {
        "spark.python.profile": "false",
        "spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT": "1",
        "spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT": "1",
        "spark.sql.session.timeZone": "GMT",
        "spark.python.profile": "false",
    },
}

spark, sc, sqlContext = init_spark(**configs)
