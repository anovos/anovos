import pytest
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.geospatial import (
    centroid,
    weighted_centroid,
    rog_calculation,
    reverse_geocode,
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

    assert int(df.collect()[0][0]) == 1
    assert int(float(df.collect()[0][1])) == -82
    assert int(float(df.collect()[0][2])) == -126
    assert int(null_df.collect()[0][0]) == 1
    assert int(float(null_df.collect()[0][1])) == -82
    assert int(float(null_df.collect()[0][2])) == -126
    assert int(invalid_df.collect()[0][0]) == 1
    assert int(float(invalid_df.collect()[0][1])) == -82
    assert int(float(invalid_df.collect()[0][2])) == -126

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert int(odf.collect()[0][0]) == 296
    assert int(odf.collect()[0][1]) == -27
    assert int(odf.collect()[0][2]) == -120
    assert int(null_odf.collect()[0][0]) == 296
    assert int(null_odf.collect()[0][1]) == -27
    assert int(null_odf.collect()[0][2]) == -120
    assert int(invalid_odf.collect()[0][0]) == 296
    assert int(invalid_odf.collect()[0][1]) == -27
    assert int(invalid_odf.collect()[0][2]) == -120


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

    assert int(df.collect()[0][0]) == 1
    assert int(float(df.collect()[0][1])) == -82
    assert int(float(df.collect()[0][2])) == -126
    assert int(null_df.collect()[0][0]) == 1
    assert int(float(null_df.collect()[0][1])) == -82
    assert int(float(null_df.collect()[0][2])) == -126
    assert int(invalid_df.collect()[0][0]) == 1
    assert int(float(invalid_df.collect()[0][1])) == -82
    assert int(float(invalid_df.collect()[0][2])) == -126

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert int(odf.collect()[0][0]) == 296
    assert int(odf.collect()[0][1]) == -54
    assert int(odf.collect()[0][2]) == -113
    assert int(null_odf.collect()[0][0]) == 296
    assert int(null_odf.collect()[0][1]) == -34
    assert int(null_odf.collect()[0][2]) == -109
    assert int(invalid_odf.collect()[0][0]) == 296
    assert int(invalid_odf.collect()[0][1]) == -15
    assert int(invalid_odf.collect()[0][2]) == -139


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

    assert int(df.collect()[0][0]) == 1
    assert int(float(df.collect()[0][1])) == -82
    assert int(float(df.collect()[0][2])) == -126
    assert int(null_df.collect()[0][0]) == 1
    assert int(float(null_df.collect()[0][1])) == -82
    assert int(float(null_df.collect()[0][2])) == -126
    assert int(invalid_df.collect()[0][0]) == 1
    assert int(float(invalid_df.collect()[0][1])) == -82
    assert int(float(invalid_df.collect()[0][2])) == -126

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert int(odf.collect()[0][0]) == 296
    assert round(odf.collect()[0][1], 2) == 0.17
    assert int(null_odf.collect()[0][0]) == 296
    assert round(null_odf.collect()[0][1], 2) == 0.17
    assert int(invalid_odf.collect()[0][0]) == 296
    assert round(invalid_odf.collect()[0][1], 2) == 0.17


def test_reverse_geocoding(df, null_df, invalid_df):
    odf = reverse_geocode(df, lat_col="latitude", long_col="longitude")
    null_odf = reverse_geocode(null_df, lat_col="latitude", long_col="longitude")
    invalid_odf = reverse_geocode(invalid_df, lat_col="latitude", long_col="longitude")

    assert df.count() == 1000
    assert null_df.count() == 1000
    assert invalid_df.count() == 1000

    assert int(df.collect()[0][0]) == 1
    assert int(float(df.collect()[0][1])) == -82
    assert int(float(df.collect()[0][2])) == -126
    assert int(null_df.collect()[0][0]) == 1
    assert int(float(null_df.collect()[0][1])) == -82
    assert int(float(null_df.collect()[0][2])) == -126
    assert int(invalid_df.collect()[0][0]) == 1
    assert int(float(invalid_df.collect()[0][1])) == -82
    assert int(float(invalid_df.collect()[0][2])) == -126

    assert odf.count() == 1000
    assert null_odf.count() == 811
    assert invalid_odf.count() == 549

    assert int(odf.collect()[0][0]) == -82
    assert int(odf.collect()[0][1], 2) == -126
    assert int(null_odf.collect()[0][0]) == -82
    assert int(null_odf.collect()[0][1], 2) == -126
    assert int(invalid_odf.collect()[0][0]) == -82
    assert int(invalid_odf.collect()[0][1], 2) == -126
