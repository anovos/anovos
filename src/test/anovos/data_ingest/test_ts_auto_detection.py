import datetime
import os

import pytest
from pyspark.sql import functions as F
from pytest import approx

from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_ingest.ts_auto_detection import (
    regex_date_time_parser,
    ts_loop_cols_pre,
    ts_preprocess,
)

sample_csv = "examples/data/time_series_data/csv/productivity.csv"

# ts_loop_cols_pre
def test_ts_loop_cols_pre(spark_session):
    df = read_dataset(
        spark_session,
        sample_csv,
        "csv",
        file_configs={"header": "True", "delimiter": ",", "inferSchema": "True"},
    )
    odf = ts_loop_cols_pre(df, "STATE")
    assert odf[0][0] == "STATE"
    assert odf[0][1] == "YR"
    assert odf[0][2] == "P_CAP"
    assert odf[1][0] == "NA"
    assert odf[1][1] == "int_c"
    assert odf[1][2] == "NA"
    assert odf[2][0] == 14
    assert odf[2][1] == 4
    assert odf[2][2] == 9


# regex_date_time_parser
def test_regex_date_time_parser(spark_session):
    df = read_dataset(
        spark_session,
        sample_csv,
        "csv",
        file_configs={"header": "True", "delimiter": ",", "inferSchema": "True"},
    )
    odf = regex_date_time_parser(
        spark_session,
        idf=df,
        id_col="STATE",
        col="YR",
        tz="local",
        val_unique_cat=4,
        trans_cat="int_c",
        save_output=None,
    )
    odf_pd = odf.toPandas()
    assert len(odf.columns) == 10
    assert df.select("YR").dtypes[0][1] == "int"
    assert odf.select("YR").dtypes[0][1] == "date"
    assert odf_pd["YR"][0] == datetime.date(1970, 1, 1)

    odf = regex_date_time_parser(
        spark_session,
        idf=df,
        id_col="STATE",
        col="YR",
        tz="local",
        val_unique_cat=4,
        trans_cat="int_c",
        save_output=None,
        output_mode="append",
    )
    odf_pd = odf.toPandas()
    len(odf.columns) == 11
    odf.select("YR_ts").dtypes[0][1] == "date"
    odf_pd["YR_ts"][0] == datetime.date(1970, 1, 1)

    odf = regex_date_time_parser(
        spark_session,
        idf=df,
        id_col="STATE",
        col="YR",
        tz="local",
        val_unique_cat=4,
        trans_cat="int_c",
        save_output="unit_testing/output/timeseries/regex_date_time_parser/",
    )
    assert os.path.isfile(
        "unit_testing/output/timeseries/regex_date_time_parser/" + "_SUCCESS"
    )


# ts_preprocess
def test_ts_preprocess(spark_session):
    df = read_dataset(
        spark_session,
        sample_csv,
        "csv",
        file_configs={"header": "True", "delimiter": ",", "inferSchema": "True"},
    )
    odf = ts_preprocess(
        spark_session,
        idf=df,
        id_col="STATE",
        output_path="unit_testing/output/timeseries/ts_preprocess/",
        tz_offset="local",
        run_type="local",
    )
    assert odf.count() == 816
    assert odf.select("STATE").distinct().count() == 48
    odf_pd = odf.toPandas()
    assert len(odf.columns) == 10
    assert df.select("YR").dtypes[0][1] == "int"
    assert odf.select("YR").dtypes[0][1] == "date"
    assert odf_pd["YR"][0] == datetime.date(1970, 1, 1)
