import pytest
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.geospatial import (
    centroid,
    weighted_centroid,
    rog_calculation,
)

path = "./data/test_dataset/geo_data/sample_geo_data.csv"
path_null = "./data/test_dataset/geo_data/null_sample_geo_data.csv"
path_invalid = "./data/test_dataset/geo_data/invalid_sample_geo_data.csv"


@pytest.fixture
def df(spark_session):
    return read_dataset(spark_session, path, "csv", file_configs={"header": "True"})


@pytest.fixture
def null_df(spark_session):
    return read_dataset(
        spark_session, path_null, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def invalid_df(spark_session):
    return read_dataset(
        spark_session, path_invalid, "csv", file_configs={"header": "True"}
    )


def test_centroid(df, null_df, invalid_df):
    odf = centroid(df, id_col="id", lat_col="latitude", long_col="longitude")
    null_odf = centroid(null_df, id_col="id", lat_col="latitude", long_col="longitude")
    invalid_odf = centroid(
        invalid_df, id_col="id", lat_col="latitude", long_col="longitude"
    )

    assert df.count() == 1000
    assert null_df.count() == 1000
    assert invalid_df.count() == 1000

    assert df.collect()[0][0] == "1"
    assert df.collect()[0][1] == "-82.32652528778522"
    assert df.collect()[0][2] == "-126.64267115740057"
    assert null_df.collect()[0][0] == "1"
    assert null_df.collect()[0][1] == "-82.32652528778522"
    assert null_df.collect()[0][2] == "-126.64267115740057"
    assert invalid_df.collect()[0][0] == "1"
    assert invalid_df.collect()[0][1] == "-82.32652528778522"
    assert invalid_df.collect()[0][2] == "-126.64267115740057"

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert odf.collect()[0][0] == "296"
    assert odf.collect()[0][1] == -27.637183976372057
    assert odf.collect()[0][2] == -120.569067380193
    assert null_odf.collect()[0][0] == "296"
    assert null_odf.collect()[0][1] == -27.637183976372057
    assert null_odf.collect()[0][2] == -120.569067380193
    assert invalid_odf.collect()[0][0] == "296"
    assert invalid_odf.collect()[0][1] == -27.637183976372057
    assert invalid_odf.collect()[0][2] == -120.569067380193


def test_weighted_centroid(df, null_df, invalid_df):
    odf = weighted_centroid(df, id_col="id", lat_col="latitude", long_col="longitude")
    null_odf = weighted_centroid(
        null_df, id_col="id", lat_col="latitude", long_col="longitude"
    )
    invalid_odf = weighted_centroid(
        invalid_df, id_col="id", lat_col="latitude", long_col="longitude"
    )

    assert df.count() == 1000
    assert null_df.count() == 1000
    assert invalid_df.count() == 1000

    assert df.collect()[0][0] == "1"
    assert df.collect()[0][1] == "-82.32652528778522"
    assert df.collect()[0][2] == "-126.64267115740057"
    assert null_df.collect()[0][0] == "1"
    assert null_df.collect()[0][1] == "-82.32652528778522"
    assert null_df.collect()[0][2] == "-126.64267115740057"
    assert invalid_df.collect()[0][0] == "1"
    assert invalid_df.collect()[0][1] == "-82.32652528778522"
    assert invalid_df.collect()[0][2] == "-126.64267115740057"

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert odf.collect()[0][0] == "296"
    assert odf.collect()[0][1] == -54.96660520402398
    assert odf.collect()[0][2] == -113.80808799888887
    assert null_odf.collect()[0][0] == "296"
    assert null_odf.collect()[0][1] == -34.07695747993671
    assert null_odf.collect()[0][2] == -109.93653864023601
    assert invalid_odf.collect()[0][0] == "296"
    assert invalid_odf.collect()[0][1] == -15.524187758097781
    assert invalid_odf.collect()[0][2] == -139.32366574495202


def test_rog_calculation(df, null_df, invalid_df):
    odf = rog_calculation(df, id_col="id", lat_col="latitude", long_col="longitude")
    null_odf = rog_calculation(
        null_df, id_col="id", lat_col="latitude", long_col="longitude"
    )
    invalid_odf = rog_calculation(
        invalid_df, id_col="id", lat_col="latitude", long_col="longitude"
    )

    assert df.count() == 1000
    assert null_df.count() == 1000
    assert invalid_df.count() == 1000

    assert df.collect()[0][0] == "1"
    assert df.collect()[0][1] == "-82.32652528778522"
    assert df.collect()[0][2] == "-126.64267115740057"
    assert null_df.collect()[0][0] == "1"
    assert null_df.collect()[0][1] == "-82.32652528778522"
    assert null_df.collect()[0][2] == "-126.64267115740057"
    assert invalid_df.collect()[0][0] == "1"
    assert invalid_df.collect()[0][1] == "-82.32652528778522"
    assert invalid_df.collect()[0][2] == "-126.64267115740057"

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert odf.collect()[0][0] == "296"
    assert odf.collect()[0][1] == 0.17222900688648224
    assert null_odf.collect()[0][0] == "296"
    assert null_odf.collect()[0][1] == 0.17222900688648224
    assert invalid_odf.collect()[0][0] == "296"
    assert invalid_odf.collect()[0][1] == 0.17222900688648224
