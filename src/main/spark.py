import __main__
from os import environ
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession

def init_spark(app_name='mw_ml_ingest', master='local[*]', jar_packages=[],
                py_files=[], spark_configs={}):
    
    """
    :param app_name: Name of Spark app.
    :param master: Cluster connection details (defaults to local[*])
                    local[*] - Run Spark locally with as many worker threads as logical cores on your machine.
    :param jar_packages: List of Spark JAR package names.
    :param files: List of files to send to Spark cluster (master and workers).
    :param spark_config: Dictionary of config key-value pairs.
    :return: A tuple of references to the Spark Session, Spark Context & SQL Context.
    """
    
    # detect execution environment
    flag_repl = not(hasattr(__main__, '__file__'))
    flag_debug = 'DEBUG' in environ.keys()

    if not (flag_repl or flag_debug):
        spark_builder = SparkSession.builder.appName(app_name)
    else:
        spark_builder = SparkSession.builder.master(master).appName(app_name)

    # create Spark JAR packages string
    spark_jars_packages = ','.join(list(jar_packages))
    spark_builder.config('spark.jars.packages', spark_jars_packages)

    spark_files = ','.join(list(py_files))
    spark_builder.config('spark.files', spark_files)

    # add other config params
    for key, val in spark_configs.items():
        spark_builder.config(key, val)

    # create spark session and contexts
    spark = spark_builder.getOrCreate()
    sc = spark.sparkContext 
    sqlContext = SQLContext(sc)
    
    return spark, sc, sqlContext


configs = {'app_name': 'MW_ML_pipeline', 
           'jar_packages': ["io.github.histogrammar:histogrammar_2.11:1.0.20",
                            "io.github.histogrammar:histogrammar-sparksql_2.11:1.0.20"], 
           'py_files': [], 
           'spark_configs': {'spark.sql.session.timeZone': 'GMT',
                            'spark.python.profile': 'true'}}

spark, sc, sqlContext = init_spark(**configs)