

import findspark
from com.mw.ds.data_ingest.data_ingest import DataIngestion
findspark.init('/home/ie_khing/Downloads/spark-2.4.8-bin-hadoop2.7/')
from pyspark.sql import SparkSession


def test_answer():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.master", "local") \
        .getOrCreate()
    
    data_ingest = DataIngestion(spark)
    schema = data_ingest.infer_schema(["data/income_dataset/parquet"],"parquet", {"header" : "True"})
    print(schema.json())



