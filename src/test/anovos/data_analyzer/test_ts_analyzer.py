import os
import pytest
from pytest import approx
from pyspark.sql import functions as F
import datetime
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_ingest.ts_auto_detection import regex_date_time_parser,ts_loop_cols_pre,ts_preprocess
from anovos.data_analyzer.ts_analyzer import ts_processed_feats, ts_eligiblity_check, ts_viz_data, ts_analyzer, daypart_cat
sample_csv = "examples/data/time_series_data/csv/productivity.csv"


#ts_processed_feats
def test_ts_processed_feats(spark_session):
    df = read_dataset(spark_session, sample_csv, "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
    ts_preprocess_odf = ts_preprocess(spark_session, idf=df, id_col='STATE', output_path="unit_testing/output/timeseries/ts_preprocess/", tz_offset="local", run_type="local")
    odf = ts_processed_feats(idf=ts_preprocess_odf , col="YR", id_col="STATE", tz="local", cnt_row = 816, cnt_unique_id = 48)
    odf_pd=odf.toPandas()
    assert odf_pd["YR_hour"][0] == 0 
    assert odf_pd["YR_minute"][0] == 0 
    assert odf_pd["YR_second"][0] == 0 
    assert odf_pd["YR_dayofmonth"][0] == 1
    assert odf_pd["YR_weekofyear"][0] == 1
    assert odf_pd["YR_dayofyear"][0] == 1
    assert odf_pd["YR_month"][0] == 1
    assert odf_pd["YR_year"][0] == 1970 
    assert odf_pd["YR_quarter"][0] == 1 
    assert odf_pd["yyyymmdd_col"][0] == datetime.date(1970, 1, 1)
    assert odf_pd["daypart_cat"][0] == "late_hours"
    assert odf_pd["week_cat"][0] == "weekday" 

#ts_eligiblity_check
def test_ts_eligiblity_check(spark_session):
	df = read_dataset(spark_session, sample_csv, "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
	ts_preprocess_odf = ts_preprocess(spark_session, idf=df, id_col='STATE', output_path="unit_testing/output/timeseries/ts_preprocess/", tz_offset="local", run_type="local")
	ts_processed_feats_odf = ts_processed_feats(idf=ts_preprocess_odf , col="YR", id_col="STATE", tz="local", cnt_row=816, cnt_unique_id=48)
	odf1= ts_eligiblity_check(spark_session, ts_processed_feats_odf, id_col="STATE", opt=1, tz_offset="local")
	assert odf1.where(odf1["attribute"]=="id_date_pair")["min"][0] == 17
	assert odf1.where(odf1["attribute"]=="id_date_pair")["max"][0] == 17
	assert odf1.where(odf1["attribute"]=="date_id_pair")["min"][1] == 48
	assert odf1.where(odf1["attribute"]=="date_id_pair")["max"][1] == 48

	odf2= ts_eligiblity_check(spark_session, ts_processed_feats_odf, id_col="STATE", opt=2, tz_offset="local")
	assert odf2["count_unique_dates"][0]==17
	assert odf2["min_date"][0]== datetime.date(1970, 1, 1)
	assert odf2["max_date"][0]== datetime.date(1986, 1, 1)
	assert odf2["date_diff"][0]== 5844
	assert odf2["mean"][0]== 365.25
	assert odf2["variance"][0]== 0.2
	assert odf2["stdev"][0]== 0.447
#ts_viz_data
def test_ts_viz_data(spark_session):
	df = read_dataset(spark_session, sample_csv, "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
	ts_preprocess_odf = ts_preprocess(spark_session, idf=df, id_col='STATE', output_path="unit_testing/output/timeseries/ts_preprocess/", tz_offset="local", run_type="local")
	ts_processed_feats_odf = ts_processed_feats(idf=ts_preprocess_odf , col="YR", id_col="STATE", tz="local", cnt_row=816, cnt_unique_id=48)
	odf=ts_viz_data(idf=ts_processed_feats_odf,x_col="YR",y_col="HWY",id_col="STATE",tz_offset="local",output_type="daily",n_cat=10)
	assert len(odf)==17
	assert odf["YR"][0] == datetime.date(1970, 1, 1)
	assert approx(odf["min"][0]) == approx(1827.14)
	assert approx(odf["max"][0]) == approx(42961.31)
	assert approx(odf["mean"][0]) == approx(9048.108125)
	assert approx(odf["median"][0]) == approx(7281.470)

	odf=ts_viz_data(idf=ts_processed_feats_odf,x_col="YR",y_col="P_CAP",id_col="STATE",tz_offset="local",output_type="daily",n_cat=20)
	assert len(odf)==17
	assert approx(odf["min"][0]) == approx(2627.12)
	assert approx(odf["max"][0]) == approx(128545.36)
	assert approx(odf["mean"][0]) == approx(20859.230417)
	assert approx(odf["median"][0]) == approx(14880.590)

	odf=ts_viz_data(idf=ts_processed_feats_odf,x_col="YR",y_col="HWY",id_col="STATE",tz_offset="local",output_type="weekly",n_cat=10)
	assert len(odf) == 7
	assert odf["dow"][0] == 1
	assert approx(odf["min"][0]) == approx(1847.43)
	assert approx(odf["max"][0]) == approx(46472.40)
	assert approx(odf["mean"][0]) == approx(10509.723229)
	assert approx(odf["median"][0]) == approx(8036.180)

	odf=ts_viz_data(idf=ts_processed_feats_odf,x_col="YR",y_col="HWY",id_col="STATE",tz_offset="local",output_type="hourly",n_cat=10)
	assert len(odf) == 1
	assert odf["daypart_cat"][0] == "late_hours"

#ts_analyzer
def test_ts_analyzer(spark_session):
	df = read_dataset(spark_session, sample_csv, "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
	ts_preprocess_odf = ts_preprocess(spark_session, idf=df, id_col='STATE', output_path="unit_testing/output/timeseries/ts_preprocess/", tz_offset="local", run_type="local")
	ts_analyzer(spark_session,idf=ts_preprocess_odf,id_col="STATE",max_days="3600",output_path="unit_testing/output/timeseries/ts_analyzer/daily/",output_type="daily",tz_offset="local",run_type="local")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/daily/" + "YR_EMP_daily.csv")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/daily/" + "YR_HWY_daily.csv")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/daily/" + "YR_P_CAP_daily.csv")
	odf_YR_P_CAP = read_dataset(spark_session, "unit_testing/output/timeseries/ts_analyzer/daily/YR_P_CAP_daily.csv", "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
	assert odf_YR_P_CAP.count()==17
	odf_YR_P_CAP_pd=odf_YR_P_CAP.toPandas()
	assert approx(odf_YR_P_CAP_pd["min"][0])==approx(2627.12)
	assert approx(odf_YR_P_CAP_pd["max"][0])==approx(128545.36)
	assert approx(odf_YR_P_CAP_pd["mean"][0])==approx(20859.230417)
	assert approx(odf_YR_P_CAP_pd["median"][0])==approx(14880.590)


	ts_analyzer(spark_session,idf=ts_preprocess_odf,id_col="STATE",max_days="3600",output_path="unit_testing/output/timeseries/ts_analyzer/weekly/",output_type="weekly",tz_offset="local",run_type="local")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/weekly/" + "YR_EMP_weekly.csv")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/weekly/" + "YR_HWY_weekly.csv")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/weekly/" + "YR_P_CAP_weekly.csv")
	odf_YR_HWY = read_dataset(spark_session, "unit_testing/output/timeseries/ts_analyzer/weekly/YR_HWY_weekly.csv", "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
	assert odf_YR_HWY.count() == 7
	odf_YR_HWY_pd= odf_YR_HWY.toPandas()
	assert odf_YR_HWY_pd["dow"][0] == 1
	assert approx(odf_YR_HWY_pd["min"][0])==approx(1847.43)
	assert approx(odf_YR_HWY_pd["max"][0])==approx(46472.40)
	assert approx(odf_YR_HWY_pd["mean"][0])==approx(10509.723229)
	assert approx(odf_YR_HWY_pd["median"][0])==approx(8036.180)


	ts_analyzer(spark_session,idf=ts_preprocess_odf,id_col="STATE",max_days="1200",output_path="unit_testing/output/timeseries/ts_analyzer/hourly/",output_type="hourly",tz_offset="local",run_type="local")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/hourly/" + "YR_EMP_hourly.csv")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/hourly/" + "YR_HWY_hourly.csv")
	assert os.path.isfile("unit_testing/output/timeseries/ts_analyzer/hourly/" + "YR_P_CAP_hourly.csv")
	odf_YR_HWY = read_dataset(spark_session, "unit_testing/output/timeseries/ts_analyzer/hourly/YR_HWY_hourly.csv", "csv",file_configs = {"header": "True", "delimiter": "," ,"inferSchema": "True"})
	assert odf_YR_HWY.count() == 1
	odf_YR_HWY_pd=odf_YR_HWY.toPandas()
	assert odf_YR_HWY_pd["daypart_cat"][0] == "late_hours"