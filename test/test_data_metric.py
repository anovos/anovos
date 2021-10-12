
from pyspark.sql import SparkSession
import findspark
from com.mw.ds.data_analyzer.metric import DataMetric
from com.mw.ds.data_ingest.data_ingest import DataIngestion
findspark.init('/home/ie_khing/Downloads/spark-2.4.8-bin-hadoop2.7/')


def test_infer_schema():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.master", "local") \
        .getOrCreate()
    data_ingest = DataIngestion(spark)
    df = data_ingest.read_dataset(
        ["data/test_dataset/csv"], "csv", {"header": "True"})
    data_metric = DataMetric(spark)
    all_metric = data_metric.generate_all_metric(df)
    for key, value in all_metric.items():
        print(key)
        if value is not None :
            value.show(10)


    # schema = data_ingest.infer_schema(["data/income_dataset/parquet"],"parquet", {"header" : "True"})
    # print(schema.json())


# def test_generate_data():
#     spark = SparkSession \
#         .builder \
#         .appName("Python Spark SQL basic example") \
#         .config("spark.master", "local") \
#         .getOrCreate()

#     data_ingest = DataIngestion(spark)
#     df = data_ingest.read_dataset(["data/test_dataset/csv"],"csv", {"header" : "True"})
#     df = data_ingest.generate_data (inputDf = df,
#     selectColumns = ["bundle", "category","platform_category", "asn", "global_category"],
#     castColumns = [("platform_category", "Integer")],
#     renameColumns = [("bundle", "new_bundle"), ("asn", "new_asn")] )


#     df.show(n=10, truncate=False)
