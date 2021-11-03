
from pyspark.sql import SparkSession
import findspark
import json
from com.mw.ds.data_analyzer.metric import DataMetric
from com.mw.ds.data_ingest.data_ingest import DataIngestion


findspark.init('/home/ie_khing/Downloads/spark-2.4.8-bin-hadoop2.7/')

def test_calc_metrics_for_csv():
    spark = SparkSession \
        .builder \
        .appName("test_calc_metrics_for_csv") \
        .config("spark.master", "local") \
        .getOrCreate()
    data_ingest = DataIngestion(spark)
    df = data_ingest.read_dataset(
        ["data/test_dataset/csv"], "csv", {"header": "True"})
    data_metric = DataMetric(spark)
    all_metric = data_metric.generate_all_metric_in_json(df)
    print(all_metric)
    
    # print(all_metric)


    # for key, value in all_metric.items():
    #     print(key)
    #     if value is not None :
    #         print(value)



def test_calc_metrics_for_parquet():
    spark = SparkSession \
        .builder \
        .appName("test_calc_metrics_for_parquet") \
        .config("spark.master", "local") \
        .getOrCreate()
    data_ingest = DataIngestion(spark)
    df = data_ingest.read_dataset(
        ["data/test_dataset/part-00000-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"], "parquet", {"header": "True"})
    data_metric = DataMetric(spark)
    all_metric = data_metric.generate_all_metric_in_json(df)
    print(all_metric)

    # for key, value in all_metric.items():
    #     print(key)
    #     if value is not None :
    #         print(value)
    #         #value.show(10)

    # schema = data_ingest.infer_schema(["data/income_dataset/parquet"],"parquet", {"header" : "True"})
    # print(schema.json())



