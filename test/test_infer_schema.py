

import findspark
findspark.init('/home/ie_khing/Downloads/spark-2.4.8-bin-hadoop2.7/')
from pyspark.sql import SparkSession

from com.mw.ds.data_ingest import data_ingest


def func(x):
    return x + 1


def test_answer():
   
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .config("spark.master", "local") \
        .getOrCreate()

    data_ingest.DataIngestion(spark)



  