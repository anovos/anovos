import pytest
import geojson
from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.geospatial import (
    geo_format_latlon,
    geo_format_cartesian,
    geo_format_geohash,
    location_distance,
    geohash_precision_control,
    location_in_polygon,
    location_in_country,
    centroid,
    weighted_centroid,
    rog_calculation,
    reverse_geocoding,
)

path = "./data/test_dataset/geo_data/sample_geo_data_two_latlon.csv"
path2 = "./data/test_dataset/geo_data/sample_geo_data.csv"
path_null = "./data/test_dataset/geo_data/null_sample_geo_data_two_latlon.csv"
path_null2 = "./data/test_dataset/geo_data/null_sample_geo_data.csv"
path_invalid = "./data/test_dataset/geo_data/invalid_sample_geo_data_two_latlon.csv"
path_invalid2 = "./data/test_dataset/geo_data/invalid_sample_geo_data.csv"
path_dms = "./data/test_dataset/geo_data/sample_geo_data_two_latlon_dms.parquet"
path_rad = "./data/test_dataset/geo_data/sample_geo_data_two_latlon_radian.csv"
path_cart = "./data/test_dataset/geo_data/sample_geo_data_two_latlon_cartesian.csv"
path_hash = "./data/test_dataset/geo_data/sample_geo_data_two_latlon_geohash.csv"
path_africa = "./data/test_dataset/geo_data/africa.geojson"
countries = "./data/country_polygons.geojson"


@pytest.fixture
def df(spark_session):
    return read_dataset(spark_session, path, "csv", file_configs={"header": "True"})


@pytest.fixture
def df2(spark_session):
    return read_dataset(spark_session, path2, "csv", file_configs={"header": "True"})


@pytest.fixture
def null_df(spark_session):
    return read_dataset(
        spark_session, path_null, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def null_df2(spark_session):
    return read_dataset(
        spark_session, path_null2, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def invalid_df(spark_session):
    return read_dataset(
        spark_session, path_invalid, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def invalid_df2(spark_session):
    return read_dataset(
        spark_session, path_invalid2, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def dms_df(spark_session):
    return read_dataset(
        spark_session, path_dms, "parquet", file_configs={"header": "True"}
    )


@pytest.fixture
def rad_df(spark_session):
    return read_dataset(spark_session, path_rad, "csv", file_configs={"header": "True"})


@pytest.fixture
def cart_df(spark_session):
    return read_dataset(
        spark_session, path_cart, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def hash_df(spark_session):
    return read_dataset(
        spark_session, path_hash, "csv", file_configs={"header": "True"}
    )


@pytest.fixture
def africa():
    with open(path_africa) as f:
        return geojson.load(f)


def test_geo_format_latlon(df, null_df, invalid_df, dms_df, rad_df):
    odf1 = geo_format_latlon(
        df, ["lat1", "lat2"], ["lon1", "lon2"], "dd", "dms", output_mode="replace"
    )
    odf2 = geo_format_latlon(
        df, ["lat1", "lat2"], ["lon1", "lon2"], "dd", "radian", output_mode="replace"
    )
    odf3 = geo_format_latlon(
        df, ["lat1", "lat2"], ["lon1", "lon2"], "dd", "cartesian", output_mode="replace"
    )
    odf4 = geo_format_latlon(
        df, ["lat1", "lat2"], ["lon1", "lon2"], "dd", "geohash", output_mode="replace"
    )
    null_odf1 = geo_format_latlon(
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "dms",
        output_mode="replace",
    )
    null_odf2 = geo_format_latlon(
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "radian",
        output_mode="replace",
    )
    null_odf3 = geo_format_latlon(
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "cartesian",
        output_mode="replace",
    )
    null_odf4 = geo_format_latlon(
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "geohash",
        output_mode="replace",
    )
    invalid_odf1 = geo_format_latlon(
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "dms",
        output_mode="replace",
    )
    invalid_odf2 = geo_format_latlon(
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "radian",
        output_mode="replace",
    )
    invalid_odf3 = geo_format_latlon(
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "cartesian",
        output_mode="replace",
    )
    invalid_odf4 = geo_format_latlon(
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "dd",
        "geohash",
        output_mode="replace",
    )
    dms_odf1 = geo_format_latlon(
        dms_df,
        ["lat_dms1", "lat_dms2"],
        ["lon_dms1", "lon_dms2"],
        "dms",
        "dd",
        output_mode="replace",
    )
    dms_odf2 = geo_format_latlon(
        dms_df,
        ["lat_dms1", "lat_dms2"],
        ["lon_dms1", "lon_dms2"],
        "dms",
        "radian",
        output_mode="replace",
    )
    dms_odf3 = geo_format_latlon(
        dms_df,
        ["lat_dms1", "lat_dms2"],
        ["lon_dms1", "lon_dms2"],
        "dms",
        "cartesian",
        output_mode="replace",
    )
    dms_odf4 = geo_format_latlon(
        dms_df,
        ["lat_dms1", "lat_dms2"],
        ["lon_dms1", "lon_dms2"],
        "dms",
        "geohash",
        output_mode="replace",
    )
    rad_odf1 = geo_format_latlon(
        rad_df,
        ["lat_rad1", "lat_rad2"],
        ["lon_rad1", "lon_rad2"],
        "radian",
        "dd",
        output_mode="replace",
    )
    rad_odf2 = geo_format_latlon(
        rad_df,
        ["lat_rad1", "lat_rad2"],
        ["lon_rad1", "lon_rad2"],
        "radian",
        "dms",
        output_mode="replace",
    )
    rad_odf3 = geo_format_latlon(
        rad_df,
        ["lat_rad1", "lat_rad2"],
        ["lon_rad1", "lon_rad2"],
        "radian",
        "cartesian",
        output_mode="replace",
    )
    rad_odf4 = geo_format_latlon(
        rad_df,
        ["lat_rad1", "lat_rad2"],
        ["lon_rad1", "lon_rad2"],
        "radian",
        "geohash",
        output_mode="replace",
    )

    assert int(odf1.collect()[0][0]) == 1
    assert int(odf1.collect()[0][1][0]) == -83
    assert int(odf1.collect()[0][1][1]) == 40
    assert int(odf1.collect()[0][1][2]) == 24
    assert int(odf1.collect()[0][2][0]) == -127
    assert int(odf1.collect()[0][2][1]) == 21
    assert int(odf1.collect()[0][2][2]) == 26
    assert int(odf1.collect()[0][3][0]) == 62
    assert int(odf1.collect()[0][3][1]) == 24
    assert int(odf1.collect()[0][3][2]) == 12
    assert int(odf1.collect()[0][4][0]) == 105
    assert int(odf1.collect()[0][4][1]) == 7
    assert int(odf1.collect()[0][4][2]) == 56

    assert int(odf2.collect()[0][0]) == 1
    assert int(odf2.collect()[0][1]) == -1
    assert int(odf2.collect()[0][2]) == -2
    assert int(odf2.collect()[0][3]) == 1
    assert int(odf2.collect()[0][4]) == 1

    assert int(odf3.collect()[0][0]) == 1
    assert int(odf3.collect()[0][1]) == -507719
    assert int(odf3.collect()[0][2]) == -682582
    assert int(odf3.collect()[0][3]) == -6313957
    assert int(odf3.collect()[0][4]) == -770447
    assert int(odf3.collect()[0][5]) == 2848990
    assert int(odf3.collect()[0][6]) == 5646186

    assert int(odf4.collect()[0][0]) == 1
    assert odf4.collect()[0][1] == "11mgwhvt"
    assert odf4.collect()[0][2] == "y74g025n"

    assert int(null_odf1.collect()[0][0]) == 1
    assert int(null_odf1.collect()[0][1][0]) == -83
    assert int(null_odf1.collect()[0][1][1]) == 40
    assert int(null_odf1.collect()[0][1][2]) == 24
    assert int(null_odf1.collect()[0][2][0]) == -127
    assert int(null_odf1.collect()[0][2][1]) == 21
    assert int(null_odf1.collect()[0][2][2]) == 26
    assert int(null_odf1.collect()[0][3][0]) == 62
    assert int(null_odf1.collect()[0][3][1]) == 24
    assert int(null_odf1.collect()[0][3][2]) == 12
    assert int(null_odf1.collect()[0][4][0]) == 105
    assert int(null_odf1.collect()[0][4][1]) == 7
    assert int(null_odf1.collect()[0][4][2]) == 56

    assert int(null_odf2.collect()[0][0]) == 1
    assert int(null_odf2.collect()[0][1]) == -1
    assert int(null_odf2.collect()[0][2]) == -2
    assert int(null_odf2.collect()[0][3]) == 1
    assert int(null_odf2.collect()[0][4]) == 1

    assert int(null_odf3.collect()[0][0]) == 1
    assert int(null_odf3.collect()[0][1]) == -507719
    assert int(null_odf3.collect()[0][2]) == -682582
    assert int(null_odf3.collect()[0][3]) == -6313957
    assert int(null_odf3.collect()[0][4]) == -770447
    assert int(null_odf3.collect()[0][5]) == 2848990
    assert int(null_odf3.collect()[0][6]) == 5646186

    assert int(null_odf4.collect()[0][0]) == 1
    assert null_odf4.collect()[0][1] == "11mgwhvt"
    assert null_odf4.collect()[0][2] == "y74g025n"

    assert int(invalid_odf1.collect()[0][0]) == 1
    assert int(invalid_odf1.collect()[0][1][0]) == -83
    assert int(invalid_odf1.collect()[0][1][1]) == 40
    assert int(invalid_odf1.collect()[0][1][2]) == 24
    assert int(invalid_odf1.collect()[0][2][0]) == -127
    assert int(invalid_odf1.collect()[0][2][1]) == 21
    assert int(invalid_odf1.collect()[0][2][2]) == 26
    assert int(invalid_odf1.collect()[0][3][0]) == 62
    assert int(invalid_odf1.collect()[0][3][1]) == 24
    assert int(invalid_odf1.collect()[0][3][2]) == 12
    assert int(invalid_odf1.collect()[0][4][0]) == 105
    assert int(invalid_odf1.collect()[0][4][1]) == 7
    assert int(invalid_odf1.collect()[0][4][2]) == 56

    assert int(invalid_odf2.collect()[0][0]) == 1
    assert int(invalid_odf2.collect()[0][1]) == -1
    assert int(invalid_odf2.collect()[0][2]) == -2
    assert int(invalid_odf2.collect()[0][3]) == 1
    assert int(invalid_odf2.collect()[0][4]) == 1

    assert int(invalid_odf3.collect()[0][0]) == 1
    assert int(invalid_odf3.collect()[0][1]) == -507719
    assert int(invalid_odf3.collect()[0][2]) == -682582
    assert int(invalid_odf3.collect()[0][3]) == -6313957
    assert int(invalid_odf3.collect()[0][4]) == -770447
    assert int(invalid_odf3.collect()[0][5]) == 2848990
    assert int(invalid_odf3.collect()[0][6]) == 5646186

    assert int(invalid_odf4.collect()[0][0]) == 1
    assert invalid_odf4.collect()[0][1] == "11mgwhvt"
    assert invalid_odf4.collect()[0][2] == "y74g025n"

    assert int(dms_odf1.collect()[0][0]) == 1
    assert int(dms_odf1.collect()[0][1]) == -82
    assert int(dms_odf1.collect()[0][2]) == -126
    assert int(dms_odf1.collect()[0][3]) == 62
    assert int(dms_odf1.collect()[0][4]) == 105

    assert int(dms_odf2.collect()[0][0]) == 1
    assert int(dms_odf2.collect()[0][1]) == -1
    assert int(dms_odf2.collect()[0][2]) == -2
    assert int(dms_odf2.collect()[0][3]) == 1
    assert int(dms_odf2.collect()[0][4]) == 1

    assert int(dms_odf3.collect()[0][0]) == 1
    assert int(dms_odf3.collect()[0][1]) == -507719
    assert int(dms_odf3.collect()[0][2]) == -682582
    assert int(dms_odf3.collect()[0][3]) == -6313957
    assert int(dms_odf3.collect()[0][4]) == -770447
    assert int(dms_odf3.collect()[0][5]) == 2848990
    assert int(dms_odf3.collect()[0][6]) == 5646186

    assert int(dms_odf4.collect()[0][0]) == 1
    assert dms_odf4.collect()[0][1] == "11mgwhvt"
    assert dms_odf4.collect()[0][2] == "y74g025n"

    assert int(rad_odf1.collect()[0][0]) == 1
    assert int(rad_odf1.collect()[0][1]) == -82
    assert int(rad_odf1.collect()[0][2]) == -126
    assert int(rad_odf1.collect()[0][3]) == 62
    assert int(rad_odf1.collect()[0][4]) == 105

    assert int(rad_odf2.collect()[0][0]) == 1
    assert int(rad_odf2.collect()[0][1][0]) == -83
    assert int(rad_odf2.collect()[0][1][1]) == 40
    assert int(rad_odf2.collect()[0][1][2]) == 24
    assert int(rad_odf2.collect()[0][2][0]) == -127
    assert int(rad_odf2.collect()[0][2][1]) == 21
    assert int(rad_odf2.collect()[0][2][2]) == 26
    assert int(rad_odf2.collect()[0][3][0]) == 62
    assert int(rad_odf2.collect()[0][3][1]) == 24
    assert int(rad_odf2.collect()[0][3][2]) == 12
    assert int(rad_odf2.collect()[0][4][0]) == 105
    assert int(rad_odf2.collect()[0][4][1]) == 7
    assert int(rad_odf2.collect()[0][4][2]) == 56

    assert int(rad_odf3.collect()[0][0]) == 1
    assert int(rad_odf3.collect()[0][1]) == -507719
    assert int(rad_odf3.collect()[0][2]) == -682582
    assert int(rad_odf3.collect()[0][3]) == -6313957
    assert int(rad_odf3.collect()[0][4]) == -770447
    assert int(rad_odf3.collect()[0][5]) == 2848990
    assert int(rad_odf3.collect()[0][6]) == 5646186

    assert int(rad_odf4.collect()[0][0]) == 1
    assert rad_odf4.collect()[0][1] == "11mgwhvt"
    assert rad_odf4.collect()[0][2] == "y74g025n"


def test_geoformat_cartesian(cart_df):
    odf1 = geo_format_cartesian(
        cart_df, ["x1", "x2"], ["y1", "y2"], ["z1", "z2"], "dd", output_mode="replace"
    )
    odf2 = geo_format_cartesian(
        cart_df, ["x1", "x2"], ["y1", "y2"], ["z1", "z2"], "dms", output_mode="replace"
    )
    odf3 = geo_format_cartesian(
        cart_df,
        ["x1", "x2"],
        ["y1", "y2"],
        ["z1", "z2"],
        "radian",
        output_mode="replace",
    )
    odf4 = geo_format_cartesian(
        cart_df,
        ["x1", "x2"],
        ["y1", "y2"],
        ["z1", "z2"],
        "geohash",
        output_mode="replace",
    )

    assert int(odf1.collect()[0][0]) == 1
    assert int(odf1.collect()[0][1]) == -82
    assert int(odf1.collect()[0][2]) == -126
    assert int(odf1.collect()[0][3]) == 62
    assert int(odf1.collect()[0][4]) == 105

    assert int(odf2.collect()[0][0]) == 1
    assert int(odf2.collect()[0][1][0]) == -83
    assert int(odf2.collect()[0][1][1]) == 40
    assert int(odf2.collect()[0][1][2]) == 24
    assert int(odf2.collect()[0][2][0]) == -127
    assert int(odf2.collect()[0][2][1]) == 21
    assert int(odf2.collect()[0][2][2]) == 26
    assert int(odf2.collect()[0][3][0]) == 62
    assert int(odf2.collect()[0][3][1]) == 24
    assert int(odf2.collect()[0][3][2]) == 12
    assert int(odf2.collect()[0][4][0]) == 105
    assert int(odf2.collect()[0][4][1]) == 7
    assert int(odf2.collect()[0][4][2]) == 56

    assert int(odf3.collect()[0][0]) == 1
    assert int(odf3.collect()[0][1]) == -1
    assert int(odf3.collect()[0][2]) == -2
    assert int(odf3.collect()[0][3]) == 1
    assert int(odf3.collect()[0][4]) == 1

    assert int(odf4.collect()[0][0]) == 1
    assert odf4.collect()[0][1] == "11mgwhvt"
    assert odf4.collect()[0][2] == "y74g025n"


def test_geoformat_geohash(hash_df):
    odf1 = geo_format_geohash(
        hash_df, ["geohash1", "geohash2"], "dd", output_mode="replace"
    )
    odf2 = geo_format_geohash(
        hash_df, ["geohash1", "geohash2"], "dms", output_mode="replace"
    )
    odf3 = geo_format_geohash(
        hash_df, ["geohash1", "geohash2"], "radian", output_mode="replace"
    )
    odf4 = geo_format_geohash(
        hash_df, ["geohash1", "geohash2"], "cartesian", output_mode="replace"
    )

    assert int(odf1.collect()[0][0]) == 1
    assert int(odf1.collect()[0][1]) == -82
    assert int(odf1.collect()[0][2]) == -126
    assert int(odf1.collect()[0][3]) == 62
    assert int(odf1.collect()[0][4]) == 105

    assert int(odf2.collect()[0][0]) == 1
    assert int(odf2.collect()[0][1][0]) == -83
    assert int(odf2.collect()[0][1][1]) == 40
    assert int(odf2.collect()[0][1][2]) == 26
    assert int(odf2.collect()[0][2][0]) == -127
    assert int(odf2.collect()[0][2][1]) == 21
    assert int(odf2.collect()[0][2][2]) == 25
    assert int(odf2.collect()[0][3][0]) == 62
    assert int(odf2.collect()[0][3][1]) == 24
    assert int(odf2.collect()[0][3][2]) == 10
    assert int(odf2.collect()[0][4][0]) == 105
    assert int(odf2.collect()[0][4][1]) == 7
    assert int(odf2.collect()[0][4][2]) == 55

    assert int(odf3.collect()[0][0]) == 1
    assert int(odf3.collect()[0][1]) == -1
    assert int(odf3.collect()[0][2]) == -2
    assert int(odf3.collect()[0][3]) == 1
    assert int(odf3.collect()[0][4]) == 1

    assert int(odf4.collect()[0][0]) == 1
    assert int(odf4.collect()[0][1]) == -507757
    assert int(odf4.collect()[0][2]) == -682625
    assert int(odf4.collect()[0][3]) == -6313949
    assert int(odf4.collect()[0][4]) == -770435
    assert int(odf4.collect()[0][5]) == 2849034
    assert int(odf4.collect()[0][6]) == 5646165


def test_location_distance(df, null_df, invalid_df, dms_df, rad_df, cart_df, hash_df):
    odf1 = location_distance(
        df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    odf2 = location_distance(
        df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    odf3 = location_distance(
        df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    odf4 = location_distance(
        df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    odf5 = location_distance(
        df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    odf6 = location_distance(
        df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )
    null_odf1 = location_distance(
        null_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    null_odf2 = location_distance(
        null_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    null_odf3 = location_distance(
        null_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    null_odf4 = location_distance(
        null_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    null_odf5 = location_distance(
        null_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    null_odf6 = location_distance(
        null_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )
    invalid_odf1 = location_distance(
        invalid_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    invalid_odf2 = location_distance(
        invalid_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    invalid_odf3 = location_distance(
        invalid_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    invalid_odf4 = location_distance(
        invalid_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    invalid_odf5 = location_distance(
        invalid_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    invalid_odf6 = location_distance(
        invalid_df,
        ["lat1", "lon1"],
        ["lat2", "lon2"],
        "dd",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )
    rad_odf1 = location_distance(
        rad_df,
        ["lat_rad1", "lon_rad1"],
        ["lat_rad2", "lon_rad2"],
        "radian",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    rad_odf2 = location_distance(
        rad_df,
        ["lat_rad1", "lon_rad1"],
        ["lat_rad2", "lon_rad2"],
        "radian",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    rad_odf3 = location_distance(
        rad_df,
        ["lat_rad1", "lon_rad1"],
        ["lat_rad2", "lon_rad2"],
        "radian",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    rad_odf4 = location_distance(
        rad_df,
        ["lat_rad1", "lon_rad1"],
        ["lat_rad2", "lon_rad2"],
        "radian",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    rad_odf5 = location_distance(
        rad_df,
        ["lat_rad1", "lon_rad1"],
        ["lat_rad2", "lon_rad2"],
        "radian",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    rad_odf6 = location_distance(
        rad_df,
        ["lat_rad1", "lon_rad1"],
        ["lat_rad2", "lon_rad2"],
        "radian",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )

    dms_odf1 = location_distance(
        dms_df,
        ["lat_dms1", "lon_dms1"],
        ["lat_dms2", "lon_dms2"],
        "dms",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    dms_odf2 = location_distance(
        dms_df,
        ["lat_dms1", "lon_dms1"],
        ["lat_dms2", "lon_dms2"],
        "dms",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    dms_odf3 = location_distance(
        dms_df,
        ["lat_dms1", "lon_dms1"],
        ["lat_dms2", "lon_dms2"],
        "dms",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    dms_odf4 = location_distance(
        dms_df,
        ["lat_dms1", "lon_dms1"],
        ["lat_dms2", "lon_dms2"],
        "dms",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    dms_odf5 = location_distance(
        dms_df,
        ["lat_dms1", "lon_dms1"],
        ["lat_dms2", "lon_dms2"],
        "dms",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    dms_odf6 = location_distance(
        dms_df,
        ["lat_dms1", "lon_dms1"],
        ["lat_dms2", "lon_dms2"],
        "dms",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )
    cart_odf1 = location_distance(
        cart_df,
        ["x1", "y1", "z1"],
        ["x2", "y2", "z2"],
        "cartesian",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    cart_odf2 = location_distance(
        cart_df,
        ["x1", "y1", "z1"],
        ["x2", "y2", "z2"],
        "cartesian",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    cart_odf3 = location_distance(
        cart_df,
        ["x1", "y1", "z1"],
        ["x2", "y2", "z2"],
        "cartesian",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    cart_odf4 = location_distance(
        cart_df,
        ["x1", "y1", "z1"],
        ["x2", "y2", "z2"],
        "cartesian",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    cart_odf5 = location_distance(
        cart_df,
        ["x1", "y1", "z1"],
        ["x2", "y2", "z2"],
        "cartesian",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    cart_odf6 = location_distance(
        cart_df,
        ["x1", "y1", "z1"],
        ["x2", "y2", "z2"],
        "cartesian",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )
    hash_odf1 = location_distance(
        hash_df,
        ["geohash1"],
        ["geohash2"],
        "geohash",
        "",
        "haversine",
        "m",
        output_mode="replace",
    )
    hash_odf2 = location_distance(
        hash_df,
        ["geohash1"],
        ["geohash2"],
        "geohash",
        "",
        "haversine",
        "km",
        output_mode="replace",
    )
    hash_odf3 = location_distance(
        hash_df,
        ["geohash1"],
        ["geohash2"],
        "geohash",
        "",
        "vincenty",
        "m",
        output_mode="replace",
    )
    hash_odf4 = location_distance(
        hash_df,
        ["geohash1"],
        ["geohash2"],
        "geohash",
        "",
        "vincenty",
        "km",
        output_mode="replace",
    )
    hash_odf5 = location_distance(
        hash_df,
        ["geohash1"],
        ["geohash2"],
        "geohash",
        "",
        "euclidean",
        "m",
        output_mode="replace",
    )
    hash_odf6 = location_distance(
        hash_df,
        ["geohash1"],
        ["geohash2"],
        "geohash",
        "",
        "euclidean",
        "km",
        output_mode="replace",
    )

    assert int(odf1.collect()[0][0]) == 1
    assert int(odf1.collect()[0][1]) == 17394182
    assert int(odf2.collect()[0][0]) == 1
    assert int(odf2.collect()[0][1]) == 17394
    assert int(odf3.collect()[0][0]) == 1
    assert int(odf3.collect()[0][1]) == 17373936
    assert int(odf4.collect()[0][0]) == 1
    assert int(odf4.collect()[0][1]) == 17373
    assert int(odf5.collect()[0][0]) == 1
    assert int(odf5.collect()[0][1]) == 12473414
    assert int(odf6.collect()[0][0]) == 1
    assert int(odf6.collect()[0][1]) == 12473

    assert int(null_odf1.collect()[0][0]) == 1
    assert int(null_odf1.collect()[0][1]) == 17394182
    assert int(null_odf2.collect()[0][0]) == 1
    assert int(null_odf2.collect()[0][1]) == 17394
    assert int(null_odf3.collect()[0][0]) == 1
    assert int(null_odf3.collect()[0][1]) == 17373936
    assert int(null_odf4.collect()[0][0]) == 1
    assert int(null_odf4.collect()[0][1]) == 17373
    assert int(null_odf5.collect()[0][0]) == 1
    assert int(null_odf5.collect()[0][1]) == 12473414
    assert int(null_odf6.collect()[0][0]) == 1
    assert int(null_odf6.collect()[0][1]) == 12473

    assert int(invalid_odf1.collect()[0][0]) == 1
    assert int(invalid_odf1.collect()[0][1]) == 17394182
    assert int(invalid_odf2.collect()[0][0]) == 1
    assert int(invalid_odf2.collect()[0][1]) == 17394
    assert int(invalid_odf3.collect()[0][0]) == 1
    assert int(invalid_odf3.collect()[0][1]) == 17373936
    assert int(invalid_odf4.collect()[0][0]) == 1
    assert int(invalid_odf4.collect()[0][1]) == 17373
    assert int(invalid_odf5.collect()[0][0]) == 1
    assert int(invalid_odf5.collect()[0][1]) == 12473414
    assert int(invalid_odf6.collect()[0][0]) == 1
    assert int(invalid_odf6.collect()[0][1]) == 12473

    assert int(rad_odf1.collect()[0][0]) == 1
    assert int(rad_odf1.collect()[0][1]) == 17394182
    assert int(rad_odf2.collect()[0][0]) == 1
    assert int(rad_odf2.collect()[0][1]) == 17394
    assert int(rad_odf3.collect()[0][0]) == 1
    assert int(rad_odf3.collect()[0][1]) == 17373936
    assert int(rad_odf4.collect()[0][0]) == 1
    assert int(rad_odf4.collect()[0][1]) == 17373
    assert int(rad_odf5.collect()[0][0]) == 1
    assert int(rad_odf5.collect()[0][1]) == 12473414
    assert int(rad_odf6.collect()[0][0]) == 1
    assert int(rad_odf6.collect()[0][1]) == 12473

    assert int(dms_odf1.collect()[0][0]) == 1
    assert int(dms_odf1.collect()[0][1]) == 17394182
    assert int(dms_odf2.collect()[0][0]) == 1
    assert int(dms_odf2.collect()[0][1]) == 17394
    assert int(dms_odf3.collect()[0][0]) == 1
    assert int(dms_odf3.collect()[0][1]) == 17373936
    assert int(dms_odf4.collect()[0][0]) == 1
    assert int(dms_odf4.collect()[0][1]) == 17373
    assert int(dms_odf5.collect()[0][0]) == 1
    assert int(dms_odf5.collect()[0][1]) == 12473414
    assert int(dms_odf6.collect()[0][0]) == 1
    assert int(dms_odf6.collect()[0][1]) == 12473

    assert int(cart_odf1.collect()[0][0]) == 1
    assert int(cart_odf1.collect()[0][1]) == 17394182
    assert int(cart_odf2.collect()[0][0]) == 1
    assert int(cart_odf2.collect()[0][1]) == 17394
    assert int(cart_odf3.collect()[0][0]) == 1
    assert int(cart_odf3.collect()[0][1]) == 17373936
    assert int(cart_odf4.collect()[0][0]) == 1
    assert int(cart_odf4.collect()[0][1]) == 17373
    assert int(cart_odf5.collect()[0][0]) == 1
    assert int(cart_odf5.collect()[0][1]) == 12473414
    assert int(cart_odf6.collect()[0][0]) == 1
    assert int(cart_odf6.collect()[0][1]) == 12473

    assert int(hash_odf1.collect()[0][0]) == 1
    assert int(hash_odf1.collect()[0][1]) == 17394166
    assert int(hash_odf2.collect()[0][0]) == 1
    assert int(hash_odf2.collect()[0][1]) == 17394
    assert int(hash_odf3.collect()[0][0]) == 1
    assert int(hash_odf3.collect()[0][1]) == 17373920
    assert int(hash_odf4.collect()[0][0]) == 1
    assert int(hash_odf4.collect()[0][1]) == 17373
    assert int(hash_odf5.collect()[0][0]) == 1
    assert int(hash_odf5.collect()[0][1]) == 12473411
    assert int(hash_odf6.collect()[0][0]) == 1
    assert int(hash_odf6.collect()[0][1]) == 12473


def test_geohash_precision_control(hash_df):
    odf1 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 8, output_mode="replace"
    )
    odf2 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 7, output_mode="replace"
    )
    odf3 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 6, output_mode="replace"
    )
    odf4 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 5, output_mode="replace"
    )
    odf5 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 4, output_mode="replace"
    )
    odf6 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 3, output_mode="replace"
    )
    odf7 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 2, output_mode="replace"
    )
    odf8 = geohash_precision_control(
        hash_df, ["geohash1", "geohash2"], 1, output_mode="replace"
    )

    assert int(odf1.collect()[0][0]) == 1
    assert odf1.collect()[0][1] == "11mgwhvt"
    assert odf1.collect()[0][2] == "y74g025n"
    assert int(odf2.collect()[0][0]) == 1
    assert odf2.collect()[0][1] == "11mgwhv"
    assert odf2.collect()[0][2] == "y74g025"
    assert int(odf3.collect()[0][0]) == 1
    assert odf3.collect()[0][1] == "11mgwh"
    assert odf3.collect()[0][2] == "y74g02"
    assert int(odf4.collect()[0][0]) == 1
    assert odf4.collect()[0][1] == "11mgw"
    assert odf4.collect()[0][2] == "y74g0"
    assert int(odf5.collect()[0][0]) == 1
    assert odf5.collect()[0][1] == "11mg"
    assert odf5.collect()[0][2] == "y74g"
    assert int(odf6.collect()[0][0]) == 1
    assert odf6.collect()[0][1] == "11m"
    assert odf6.collect()[0][2] == "y74"
    assert int(odf7.collect()[0][0]) == 1
    assert odf7.collect()[0][1] == "11"
    assert odf7.collect()[0][2] == "y7"
    assert int(odf8.collect()[0][0]) == 1
    assert odf8.collect()[0][1] == "1"
    assert odf8.collect()[0][2] == "y"


def test_location_in_polygon(df, null_df, invalid_df, africa):
    odf = location_in_polygon(
        df, ["lat1", "lat2"], ["lon1", "lon2"], africa, output_mode="replace"
    )
    null_odf = location_in_polygon(
        null_df, ["lat1", "lat2"], ["lon1", "lon2"], africa, output_mode="replace"
    )
    invalid_odf = location_in_polygon(
        invalid_df, ["lat1", "lat2"], ["lon1", "lon2"], africa, output_mode="replace"
    )

    assert int(odf.collect()[0][0]) == 1
    assert int(odf.collect()[0][1]) == 0
    assert int(odf.collect()[0][2]) == 0
    assert int(odf.collect()[4][0]) == 5
    assert int(odf.collect()[4][1]) == 1
    assert int(odf.collect()[4][2]) == 0
    assert int(null_odf.collect()[0][0]) == 1
    assert int(null_odf.collect()[0][1]) == 0
    assert int(null_odf.collect()[0][2]) == 0
    assert int(null_odf.collect()[4][0]) == 5
    assert int(null_odf.collect()[4][1]) == 1
    assert int(null_odf.collect()[4][2]) == 0
    assert int(invalid_odf.collect()[0][0]) == 1
    assert int(invalid_odf.collect()[0][1]) == 0
    assert int(invalid_odf.collect()[0][2]) == 0
    assert int(invalid_odf.collect()[4][0]) == 5
    assert int(invalid_odf.collect()[4][1]) == 1
    assert int(invalid_odf.collect()[4][2]) == 0


def test_location_in_country(spark_session, df, null_df, invalid_df):
    odf1 = location_in_country(
        spark_session,
        df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "AQ",
        method_type="approx",
        output_mode="replace",
    )
    odf2 = location_in_country(
        spark_session,
        df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "AQ",
        countries,
        method_type="exact",
        output_mode="replace",
    )
    odf3 = location_in_country(
        spark_session,
        df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "RU",
        method_type="approx",
        output_mode="replace",
    )
    odf4 = location_in_country(
        spark_session,
        df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "RU",
        countries,
        method_type="exact",
        output_mode="replace",
    )

    null_odf1 = location_in_country(
        spark_session,
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "AQ",
        method_type="approx",
        output_mode="replace",
    )
    null_odf2 = location_in_country(
        spark_session,
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "AQ",
        countries,
        method_type="exact",
        output_mode="replace",
    )
    null_odf3 = location_in_country(
        spark_session,
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "RU",
        method_type="approx",
        output_mode="replace",
    )
    null_odf4 = location_in_country(
        spark_session,
        null_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "RU",
        countries,
        method_type="exact",
        output_mode="replace",
    )

    invalid_odf1 = location_in_country(
        spark_session,
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "AQ",
        method_type="approx",
        output_mode="replace",
    )
    invalid_odf2 = location_in_country(
        spark_session,
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "AQ",
        countries,
        method_type="exact",
        output_mode="replace",
    )
    invalid_odf3 = location_in_country(
        spark_session,
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "RU",
        method_type="approx",
        output_mode="replace",
    )
    invalid_odf4 = location_in_country(
        spark_session,
        invalid_df,
        ["lat1", "lat2"],
        ["lon1", "lon2"],
        "RU",
        countries,
        method_type="exact",
        output_mode="replace",
    )

    assert int(odf1.collect()[0][0]) == 1
    assert int(odf1.collect()[0][1]) == 1
    assert int(odf1.collect()[0][2]) == 0
    assert int(odf2.collect()[0][0]) == 1
    assert int(odf2.collect()[0][1]) == 1
    assert int(odf2.collect()[0][2]) == 0
    assert int(odf3.collect()[0][0]) == 1
    assert int(odf3.collect()[0][1]) == 0
    assert int(odf3.collect()[0][2]) == 1
    assert int(odf4.collect()[0][0]) == 1
    assert int(odf4.collect()[0][1]) == 0
    assert int(odf4.collect()[0][2]) == 1

    assert int(null_odf1.collect()[0][0]) == 1
    assert int(null_odf1.collect()[0][1]) == 1
    assert int(null_odf1.collect()[0][2]) == 0
    assert int(null_odf2.collect()[0][0]) == 1
    assert int(null_odf2.collect()[0][1]) == 1
    assert int(null_odf2.collect()[0][2]) == 0
    assert int(null_odf3.collect()[0][0]) == 1
    assert int(null_odf3.collect()[0][1]) == 0
    assert int(null_odf3.collect()[0][2]) == 1
    assert int(null_odf4.collect()[0][0]) == 1
    assert int(null_odf4.collect()[0][1]) == 0
    assert int(null_odf4.collect()[0][2]) == 1

    assert int(invalid_odf1.collect()[0][0]) == 1
    assert int(invalid_odf1.collect()[0][1]) == 1
    assert int(invalid_odf1.collect()[0][2]) == 0
    assert int(invalid_odf2.collect()[0][0]) == 1
    assert int(invalid_odf2.collect()[0][1]) == 1
    assert int(invalid_odf2.collect()[0][2]) == 0
    assert int(invalid_odf3.collect()[0][0]) == 1
    assert int(invalid_odf3.collect()[0][1]) == 0
    assert int(invalid_odf3.collect()[0][2]) == 1
    assert int(invalid_odf4.collect()[0][0]) == 1
    assert int(invalid_odf4.collect()[0][1]) == 0
    assert int(invalid_odf4.collect()[0][2]) == 1


def test_centroid(df2, null_df2, invalid_df2):
    df = df2
    null_df = null_df2
    invalid_df = invalid_df2

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


def test_weighted_centroid(df2, null_df2, invalid_df2):
    df = df2
    null_df = null_df2
    invalid_df = invalid_df2

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


def test_rog_calculation(df2, null_df2, invalid_df2):
    df = df2
    null_df = null_df2
    invalid_df = invalid_df2

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


def test_reverse_geocoding(df2, null_df2, invalid_df2):
    df = df2
    null_df = null_df2
    invalid_df = invalid_df2

    odf = reverse_geocoding(df, lat_col="latitude", long_col="longitude")
    null_odf = reverse_geocoding(null_df, lat_col="latitude", long_col="longitude")
    invalid_odf = reverse_geocoding(
        invalid_df, lat_col="latitude", long_col="longitude"
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

    assert int(odf.collect()[0][0]) == -82
    assert int(odf.collect()[0][1]) == -126
    assert int(null_odf.collect()[0][0]) == -82
    assert int(null_odf.collect()[0][1]) == -126
    assert int(invalid_odf.collect()[0][0]) == -82
    assert int(invalid_odf.collect()[0][1]) == -126
