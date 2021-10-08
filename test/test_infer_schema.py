

import findspark
from com.mw.ds.data_ingest.data_ingest import DataIngestion
findspark.init('/home/ie_khing/Downloads/spark-2.4.8-bin-hadoop2.7/')
from pyspark.sql import SparkSession


def test_infer_schema():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.master", "local") \
        .getOrCreate()
    
    data_ingest = DataIngestion(spark)
    schema = data_ingest.infer_schema(["data/income_dataset/parquet"],"parquet", {"header" : "True"})
    print(schema.json())


def test_generate_data():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.master", "local") \
        .getOrCreate()
    
    data_ingest = DataIngestion(spark)
    df = data_ingest.read_dataset(["data/test_dataset/csv"],"csv", {"header" : "True"})    
    df = data_ingest.generate_data (inputDf = df, 
    selectColumns = ["bundle", "category","platform_category", "asn", "global_category"], 
    castColumns = [("platform_category", "Integer")], 
    renameColumns = [("bundle", "new_bundle"), ("asn", "new_asn")] )
    df.show(n=10, truncate=False)



