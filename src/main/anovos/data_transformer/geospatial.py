"""
The geospatial module supports transformation & calculation functions for geospatial fields, such as transforming
between different formats, controlling geohash values precisions, checking if a location is inside a country, calculating
centroid and Radius of Gyration, and reverse latitude-longitude pairs into address.
Functions supported through this module are listed below:
- geo_format_latlon
- geo_format_cartesian
- geo_format_geohash
- location_distance
- geohash_precision_control
- location_in_polygon
- location_in_country
- centroid
- weighted_centroid
- rog_calculation
- reverse_geocoding
"""


from math import sin, cos, sqrt, atan2, pi, radians
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql import types as T
import reverse_geocoder as rg
import warnings
from anovos.data_ingest.data_ingest import recast_column
from anovos.data_transformer.geo_utils import (
    EARTH_RADIUS,
    from_latlon_decimal_degrees,
    to_latlon_decimal_degrees,
    haversine_distance,
    vincenty_distance,
    euclidean_distance,
    f_point_in_polygons,
    point_in_country_approx,
)


def geo_format_latlon(
    idf,
    list_of_lat,
    list_of_lon,
    input_format,
    output_format,
    result_prefix=[],
    optional_configs={"geohash_precision": 8, "radius": EARTH_RADIUS},
    output_mode="append",
):
    """
    This function is the main function to convert the input data's location columns from lat,lon format to desired format
    based on output_format. If output_mode is set to True, the original location columns will be dropped.
    For each location column, "to_latlon_decimal_degrees" will be called to convert them to decimal degrees, and then
    "from_latlon_decimal_degrees" will be called to transform decimal degrees to output_format
    If output_format is "dd" or "dms" or "radian", 2 new columns containing "_lat_", "_long_" in the column names will be created.
    If output_format is "cartesian", 3 new columns containing "_x_", "_y_", "_z_" in the column names will be created.
    If output_format is "geohash", 1 new column containing "_geohash" in the column name will be created.

    Parameters
    ----------
    idf
        Input Dataframe.
    list_of_lat
        List of columns representing latitude e.g., ["lat1","lat2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lat1|lat2".
    list_of_lon
        List of columns representing longitude e.g., ["lon1","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lon1|lon2".
        list_of_lon must have the same length as list_of_lat such that i-th element of
        list_of_lat and i-th element of list_of_lon form a lat-lon pair to format.
    input_format
        "dd", "dms", "radian".
        "dd" represents latitude and longitude in decimal degrees.
        "dms" represents latitude and longitude in degrees minutes second.
        "radian" represents latitude and longitude in radians.
    output_format
        "dd", "dms", "radian", "cartesian", "geohash".
        "cartesian" represents the Cartesian coordinates of the point in three-dimensional space.
        "geohash" represents geocoded locations.
    result_prefix
        List of prefixes for the newly generated column names.
        Alternatively, prefixes can be specified in a string format,
        where different prefixes are separated by pipe delimiter "|" e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_lat and list_of_lon.
        If it is empty, <lat>_<lon> will be used for each lat-lon pair.
        For example, list_of_lat is "lat1|lat2", list_of_lon is "L1|L2".
        Case 1: result_prefix = "L1|L2".
            If output_format is "dd", "dms" or "radian", new columns will be named as
            L1_lat_<output_format>, L1_lon_<output_format>, L2_lat_<output_format>, L2_lon_<output_format>.
            If output_format is "cartesian", new columns will be named as
            L1_x, L1_y, L1_z, L2_x, L2_y, L2_z.
            If output_format is "geohash", new columns will be named as
            L1_geohash and L2_geohash.
        Case 2: result_prefix = [].
            Prefixes "L1" and "L2" in above column names will be replaced by "lat1_lon1" and "lat2_lon2".
        (Default value = [])
    optional_configs
        The following keys can be used:
        - geohash_precision: precision of the resultant geohash. This key is only used when output_format
          is "geohash". (Default value = 8)
        - radius: radius of Earth. Necessary only when output_format is "cartesian".
          (Default value = EARTH_RADIUS)
    output_mode
        "replace", "append".
        "replace" option appends transformed column to the input dataset and removes the original ones.
        "append" option appends transformed column to the input dataset.
        (Default value = "append")

    Returns
    -------
    DataFrame
    """
    geohash_precision = int(optional_configs.get("geohash_precision", 8))
    radius = optional_configs.get("radius", EARTH_RADIUS)

    if isinstance(list_of_lat, str):
        list_of_lat = [x.strip() for x in list_of_lat.split("|")]
    if isinstance(list_of_lon, str):
        list_of_lon = [x.strip() for x in list_of_lon.split("|")]
    if isinstance(result_prefix, str):
        result_prefix = [x.strip() for x in result_prefix.split("|")]

    if any(x not in idf.columns for x in list_of_lat + list_of_lon):
        raise TypeError("Invalid input for list_of_lat or list_of_lon")

    format_list = ["dd", "dms", "radian", "cartesian", "geohash"]
    if (input_format not in format_list[:3]) or (output_format not in format_list):
        raise TypeError("Invalid input for input_format or output_format")

    if len(list_of_lat) != len(list_of_lon):
        raise TypeError("list_of_lat and list_of_lon must have the same length")
    if result_prefix and (len(result_prefix) != len(list_of_lat)):
        raise TypeError(
            "result_prefix must have the same length as list_of_lat and list_of_lon if it is not empty"
        )

    f_to_latlon_dd = F.udf(
        lambda loc: to_latlon_decimal_degrees(loc, input_format, radius),
        T.ArrayType(T.FloatType()),
    )

    from_latlon_dd_ = lambda loc: from_latlon_decimal_degrees(
        loc, output_format, radius, geohash_precision
    )
    if output_format in ["dd", "radian", "cartesian"]:
        f_from_latlon_dd = F.udf(from_latlon_dd_, T.ArrayType(T.FloatType()))
    elif output_format == "dms":
        f_from_latlon_dd = F.udf(
            from_latlon_dd_, T.ArrayType(T.ArrayType(T.FloatType()))
        )
    elif output_format == "geohash":
        f_from_latlon_dd = F.udf(from_latlon_dd_, T.StringType())

    odf = idf
    for i, (lat, lon) in enumerate(zip(list_of_lat, list_of_lon)):
        col = result_prefix[i] if result_prefix else (lat + "_" + lon)

        odf = odf.withColumn(col + "_temp", F.array(lat, lon)).withColumn(
            col + "_" + output_format, f_from_latlon_dd(f_to_latlon_dd(col + "_temp"))
        )

        if output_format in ["dd", "dms", "radian"]:
            odf = (
                odf.withColumn(
                    col + "_lat_" + output_format, F.col(col + "_" + output_format)[0]
                )
                .withColumn(
                    col + "_lon_" + output_format, F.col(col + "_" + output_format)[1]
                )
                .drop(col + "_" + output_format)
            )

        if output_format == "cartesian":
            odf = (
                odf.withColumn(col + "_x", F.col(col + "_" + output_format)[0])
                .withColumn(col + "_y", F.col(col + "_" + output_format)[1])
                .withColumn(col + "_z", F.col(col + "_" + output_format)[2])
                .drop(col + "_" + output_format)
            )

        odf = odf.drop(col + "_temp")

        if output_mode == "replace":
            odf = odf.drop(lat, lon)

    return odf


def geo_format_cartesian(
    idf,
    list_of_x,
    list_of_y,
    list_of_z,
    output_format,
    result_prefix=[],
    optional_configs={"geohash_precision": 8, "radius": EARTH_RADIUS},
    output_mode="append",
):
    """
    This function helps to convert the input data's location columns from cartesian format to desired format based on
    output_format. If output_mode is set to True, the original location columns will be dropped.
    For each location column, "to_latlon_decimal_degrees" will be called to convert them to decimal degrees, and then
    "from_latlon_decimal_degrees" will be called to transform decimal degrees to output_format
    If output_format is "dd" or "dms" or "radian", 2 new columns containing "_lat_", "_long_" in the column names will be created.
    If output_format is "geohash", 1 new column containing "_geohash" in the column name will be created.

    Parameters
    ----------
    idf
        Input Dataframe.
    list_of_x
        List of columns representing x axis values e.g., ["x1","x2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "x1|x2".
    list_of_y
        List of columns representing y axis values e.g., ["y1","y2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "y1|y2".
    list_of_z
        List of columns representing z axis values e.g., ["z1","z2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "z1|z2".
        list_of_x, list_of_y and list_of_z must have the same length such that the
        i-th element of 3 lists form an x-y-z pair to format.
    output_format
        "dd", "dms", "radian", "geohash"
        "dd" represents latitude and longitude in decimal degrees.
        "dms" represents latitude and longitude in degrees minutes second.
        "radian" represents latitude and longitude in radians.
        "geohash" represents geocoded locations.
    result_prefix
        List of prefixes for the newly generated column names.
        Alternatively, prefixes can be specified in a string format,
        where different prefixes are separated by pipe delimiter "|" e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_x, list_of_y and list_of_z.
        If it is empty, <x>_<y>_<z> will be used for each x-y-z pair.
        For example, list_of_x is "x1|x2", list_of_y is "y1|y2" and list_of_z is "z1|z2"
        Case 1: result_prefix = "L1|L2".
            If output_format is "dd", "dms" or "radian", new columns will be named as
            L1_lat_<output_format>, L1_lon_<output_format>, L2_lat_<output_format>, L2_lon_<output_format>.
            If output_format is "geohash", new columns will be named as
            L1_geohash and L2_geohash.
        Case 2: result_prefix = [].
            Prefixes "L1" and "L2" in above column names will be replaced by "x1_y1_z1" and "x2_y2_z2".
        (Default value = [])
    optional_configs
        The following keys can be used:
        - geohash_precision: precision of the resultant geohash. This key is only used when output_format
          is "geohash". (Default value = 8)
        - radius: radius of Earth. (Default value = EARTH_RADIUS)
    output_mode
        "replace", "append".
        "replace" option appends transformed column to the input dataset and removes the original ones.
        "append" option appends transformed column to the input dataset.
        (Default value = "append")

    Returns
    -------
    DataFrame
    """

    geohash_precision = int(optional_configs.get("geohash_precision", 8))
    radius = optional_configs.get("radius", EARTH_RADIUS)

    if isinstance(list_of_x, str):
        list_of_x = [x.strip() for x in list_of_x.split("|")]
    if isinstance(list_of_y, str):
        list_of_y = [x.strip() for x in list_of_y.split("|")]
    if isinstance(list_of_z, str):
        list_of_z = [x.strip() for x in list_of_z.split("|")]
    if isinstance(result_prefix, str):
        result_prefix = [x.strip() for x in result_prefix.split("|")]

    if any(x not in idf.columns for x in list_of_x + list_of_y + list_of_z):
        raise TypeError("Invalid input for list_of_x or list_of_y or list_of_z")

    format_list = ["dd", "dms", "radian", "geohash"]
    if output_format not in format_list:
        raise TypeError("Invalid input for output_format")

    if len({len(list_of_x), len(list_of_y), len(list_of_z)}) != 1:
        raise TypeError("list_of_x, list_of_y and list_of_z must have the same length")
    if result_prefix and (len(result_prefix) != len(list_of_x)):
        raise TypeError(
            "result_prefix must have the same length as list_of_x, list_of_y and list_of_y if it is not empty"
        )

    f_to_latlon_dd = F.udf(
        lambda loc: to_latlon_decimal_degrees(loc, "cartesian", radius),
        T.ArrayType(T.FloatType()),
    )

    from_latlon_dd_ = lambda loc: from_latlon_decimal_degrees(
        loc, output_format, radius, geohash_precision
    )
    if output_format in ["dd", "radian"]:
        f_from_latlon_dd = F.udf(from_latlon_dd_, T.ArrayType(T.FloatType()))
    elif output_format == "dms":
        f_from_latlon_dd = F.udf(
            from_latlon_dd_, T.ArrayType(T.ArrayType(T.FloatType()))
        )
    elif output_format == "geohash":
        f_from_latlon_dd = F.udf(from_latlon_dd_, T.StringType())

    odf = idf
    for i, (x, y, z) in enumerate(zip(list_of_x, list_of_y, list_of_z)):
        col = result_prefix[i] if result_prefix else (x + "_" + y + "_" + z)

        odf = odf.withColumn(col + "_temp", F.array(x, y, z)).withColumn(
            col + "_" + output_format, f_from_latlon_dd(f_to_latlon_dd(col + "_temp"))
        )

        if output_format in ["dd", "dms", "radian"]:
            odf = (
                odf.withColumn(
                    col + "_lat_" + output_format, F.col(col + "_" + output_format)[0]
                )
                .withColumn(
                    col + "_lon_" + output_format, F.col(col + "_" + output_format)[1]
                )
                .drop(col + "_" + output_format)
            )

        odf = odf.drop(col + "_temp")

        if output_mode == "replace":
            odf = odf.drop(x, y, z)

    return odf


def geo_format_geohash(
    idf,
    list_of_geohash,
    output_format,
    result_prefix=[],
    optional_configs={"radius": EARTH_RADIUS},
    output_mode="append",
):
    """
    This function is the main function to convert the input data's location columns from geohash  format to desired
    format based on output_format. If output_mode is set to True, the original location columns will be dropped.
    For each location column, "to_latlon_decimal_degrees" will be called to convert them to decimal degrees, and then
    "from_latlon_decimal_degrees" will be called to transform decimal degrees to output_format
    If output_format is "dd" or "dms" or "radian", 2 new columns containing "_lat_", "_long_" in the column names will be created.
    If output_format is "cartesian", 3 new columns containing "_x_", "_y_", "_z_" in the column names will be created.

    Parameters
    ----------
    idf
        Input Dataframe.
    list_of_geohash
        List of columns representing geohash e.g., ["gh1","gh2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "gh1|gh2".
    output_format
        "dd", "dms", "radian", "cartesian"
        "dd" represents latitude and longitude in decimal degrees.
        "dms" represents latitude and longitude in degrees minutes second.
        "radian" represents latitude and longitude in radians.
        "cartesian" represents the Cartesian coordinates of the point in three-dimensional space.
    result_prefix
        List of prefixes for the newly generated column names.
        Alternatively, prefixes can be specified in a string format,
        where different prefixes are separated by pipe delimiter "|" e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_geohash.
        If it is empty, <geohash column name> will be used as the prefix for transformed columns.
        For example, list_of_geohash is "gh1|gh2".
        Case 1: result_prefix = "L1|L2".
            If output_format is "dd", "dms" or "radian", new columns will be named as
            L1_lat_<output_format>, L1_lon_<output_format>, L2_lat_<output_format>, L2_lon_<output_format>.
            If output_format is "cartesian", new columns will be named as
            L1_x, L1_y, L1_z, L2_x, L2_y, L2_z.
        Case 2: result_prefix = [].
            Prefixes "L1" and "L2" in above column names will be replaced by "gh1" and "gh2".
        (Default value = [])
    optional_configs
        The following key can be used:
        - radius: radius of Earth. Necessary only when output_format is "cartesian".
          (Default value = EARTH_RADIUS)
    output_mode
        "replace", "append".
        "replace" option appends transformed column to the input dataset and removes the original ones.
        "append" option appends transformed column to the input dataset.
        (Default value = "append")

    Returns
    -------
    DataFrame
    """

    radius = optional_configs.get("radius", EARTH_RADIUS)

    if isinstance(list_of_geohash, str):
        list_of_geohash = [x.strip() for x in list_of_geohash.split("|")]
    if isinstance(result_prefix, str):
        result_prefix = [x.strip() for x in result_prefix.split("|")]

    if any(x not in idf.columns for x in list_of_geohash):
        raise TypeError("Invalid input for list_of_geohash")

    format_list = ["dd", "dms", "radian", "cartesian"]
    if output_format not in format_list:
        raise TypeError("Invalid input for output_format")

    if result_prefix and (len(result_prefix) != len(list_of_geohash)):
        raise TypeError(
            "result_prefix must have the same length as list_of_geohash if it is not empty"
        )

    f_to_latlon_dd = F.udf(
        lambda loc: to_latlon_decimal_degrees(loc, "geohash", radius),
        T.ArrayType(T.FloatType()),
    )

    from_latlon_dd_ = lambda loc: from_latlon_decimal_degrees(
        loc, output_format, radius
    )
    if output_format in ["dd", "radian", "cartesian"]:
        f_from_latlon_dd = F.udf(from_latlon_dd_, T.ArrayType(T.FloatType()))
    elif output_format == "dms":
        f_from_latlon_dd = F.udf(
            from_latlon_dd_, T.ArrayType(T.ArrayType(T.FloatType()))
        )

    odf = idf
    for i, geohash in enumerate(list_of_geohash):
        col = result_prefix[i] if result_prefix else geohash

        odf = odf.withColumn(
            col + "_" + output_format, f_from_latlon_dd(f_to_latlon_dd(geohash))
        )

        if output_format in ["dd", "dms", "radian"]:
            odf = (
                odf.withColumn(
                    col + "_lat_" + output_format, F.col(col + "_" + output_format)[0]
                )
                .withColumn(
                    col + "_lon_" + output_format, F.col(col + "_" + output_format)[1]
                )
                .drop(col + "_" + output_format)
            )

        if output_format == "cartesian":
            odf = (
                odf.withColumn(col + "_x", F.col(col + "_" + output_format)[0])
                .withColumn(col + "_y", F.col(col + "_" + output_format)[1])
                .withColumn(col + "_z", F.col(col + "_" + output_format)[2])
                .drop(col + "_" + output_format)
            )

        if output_mode == "replace":
            odf = odf.drop(geohash)

    return odf


def location_distance(
    idf,
    list_of_cols_loc1,
    list_of_cols_loc2,
    loc_format="dd",
    result_prefix="",
    distance_type="haversine",
    unit="m",
    optional_configs={"radius": EARTH_RADIUS, "vincenty_model": "WGS-84"},
    output_mode="append",
):
    """
    This function calculates the distance between 2 locations, and the distance formula is determined by distance_type.
    If distance_type = "vincenty", thed loc_format should be "dd", and list_of_cols_loc1, list_of_cols_loc2
    should be in [lat1, lon1] and [lat2, lon2] format respectively. "vincenty_distance" function will be called to
    calculate the distance.
    If distance_type = "haversine", then loc_format should be "radian", and list_of_cols_loc1, list_of_cols_loc2
    should be in [lat1, lon1] and [lat2, lon2] format respectively. "haversine_distance" function will be called to
    calculate the distance.
     If distance_type = "euclidean", then loc_format should be "cartesian", and list_of_cols_loc1, list_of_cols_loc2
    should be in  [x1, y1, z1] and [x2, y2, z2] format respectively. "euclidean_distance" function will be called to
    calculate the distance.
    If loc_format does not match with distance_type's desired format, necessary conversion of location columns will be performed
    with the help of "geo_format_latlon", "geo_format_cartesian" and "geo_format_geohash" functions.

    Parameters
    ----------
    idf
        Input Dataframe.
    list_of_cols_loc1
        List of columns to express the first location e.g., ["lat1","lon1"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lat1|lon1".
    list_of_cols_loc2
        List of columns to express the second location e.g., ["lat2","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lat2|lon2".
    loc_format
        "dd", "dms", "radian", "cartesian", "geohash". (Default value = "dd")
    result_prefix
        Prefix for the newly generated column. It must be a string or a list with one element.
        If it is empty, <list_of_cols_loc1 joined by '_'>_<list_of_cols_loc2 joined by '_'>
        will be used as the prefix.
        For example, list_of_cols_loc1 is "lat1|lon1", list_of_cols_loc2 is "lat2|lon2".
        Case 1: result_prefix = "L1_L2": the new column will be named as L1_L2_distance.
        Case 2: result_prefix = []: the new column will be named as lat1_lon1_lat2_lon2_distance.
        (Default value = '')
    distance_type
        "vincenty", "haversine", "euclidean". (Default value = "haversine")
        "vincenty" option calculates the distance between two points on the surface of a spheroid.
        "haversine" option calculates the great-circle distance between two points on a sphere.
        "euclidean" option calculates the length of the line segment between two points.
    unit
        "m", "km".
        Unit of the result. (Default value = "m")
    optional_configs
        The following keys can be used:
        - radius: radius of Earth. Necessary only when output_format is "cartesian".
          (Default value = EARTH_RADIUS)
        - vincenty_model: The ellipsoidal model to use. Supported values: "WGS-84", "GRS-80", "Airy (1830)",
          "Intl 1924", "Clarke (1880)", "GRS-67". For more information, please refer to geopy.distance.ELLIPSOIDS.
          (Default value = "WGS-84")
    output_mode
        "replace", "append".
        "replace" option replaces original columns with transformed column.
        "append" option appends the transformed column to the input dataset with name "<loc1>_<loc2>_distance".
        (Default value = "append")

    Returns
    -------
    DataFrame
    """

    radius = optional_configs.get("radius", EARTH_RADIUS)
    vincenty_model = optional_configs.get("vincenty_model", "WGS-84")

    if isinstance(list_of_cols_loc1, str):
        list_of_cols_loc1 = [x.strip() for x in list_of_cols_loc1.split("|")]
    if isinstance(list_of_cols_loc2, str):
        list_of_cols_loc2 = [x.strip() for x in list_of_cols_loc2.split("|")]

    if isinstance(result_prefix, list):
        if len(result_prefix) > 1:
            raise TypeError(
                "If result_prefix is a list, it can contain maximally 1 element"
            )
        elif len(result_prefix) == 1:
            result_prefix = result_prefix[0]

    if any(i not in idf.columns for i in list_of_cols_loc1 + list_of_cols_loc2):
        raise TypeError("Invalid input for list_of_cols_loc1 or list_of_cols_loc2")

    if distance_type not in ["vincenty", "haversine", "euclidean"]:
        raise TypeError("Invalid input for distance_type")

    if loc_format not in ["dd", "dms", "radian", "cartesian", "geohash"]:
        raise TypeError("Invalid input for loc_format")

    format_mapping = {"vincenty": "dd", "haversine": "radian", "euclidean": "cartesian"}
    format_required = format_mapping[distance_type]

    if loc_format != format_required:
        if loc_format in ["dd", "dms", "radian"]:
            idf = geo_format_latlon(
                idf,
                list_of_lat=[list_of_cols_loc1[0], list_of_cols_loc2[0]],
                list_of_lon=[list_of_cols_loc1[1], list_of_cols_loc2[1]],
                input_format=loc_format,
                output_format=format_required,
                result_prefix=["temp_loc1", "temp_loc2"],
                optional_configs={"radius": radius},
                output_mode="append",
            )

        elif loc_format == "cartesian":
            idf = geo_format_cartesian(
                idf,
                list_of_x=[list_of_cols_loc1[0], list_of_cols_loc2[0]],
                list_of_y=[list_of_cols_loc1[1], list_of_cols_loc2[1]],
                list_of_z=[list_of_cols_loc1[2], list_of_cols_loc2[2]],
                output_format=format_required,
                result_prefix=["temp_loc1", "temp_loc2"],
                optional_configs={"radius": radius},
                output_mode="append",
            )

        elif loc_format == "geohash":
            idf = geo_format_geohash(
                idf,
                list_of_geohash=[list_of_cols_loc1[0], list_of_cols_loc2[0]],
                output_format=format_required,
                result_prefix=["temp_loc1", "temp_loc2"],
                optional_configs={"radius": radius},
                output_mode="append",
            )

        if format_required == "dd":
            loc1, loc2 = ["temp_loc1_lat_dd", "temp_loc1_lon_dd"], [
                "temp_loc2_lat_dd",
                "temp_loc2_lon_dd",
            ]

        elif format_required == "radian":
            loc1, loc2 = ["temp_loc1_lat_radian", "temp_loc1_lon_radian"], [
                "temp_loc2_lat_radian",
                "temp_loc2_lon_radian",
            ]

        elif format_required == "cartesian":
            loc1, loc2 = ["temp_loc1_x", "temp_loc1_y", "temp_loc1_z"], [
                "temp_loc2_x",
                "temp_loc2_y",
                "temp_loc2_z",
            ]

        idf = (
            idf.withColumn("temp_loc1", F.array(*loc1))
            .withColumn("temp_loc2", F.array(*loc2))
            .drop(*(loc1 + loc2))
        )
    else:
        idf = idf.withColumn("temp_loc1", F.array(*list_of_cols_loc1)).withColumn(
            "temp_loc2", F.array(*list_of_cols_loc2)
        )

    if distance_type == "vincenty":
        compute_distance = lambda x1, x2: vincenty_distance(
            x1, x2, unit, vincenty_model
        )
    elif distance_type == "haversine":
        compute_distance = lambda x1, x2: haversine_distance(
            x1, x2, "radian", unit, radius
        )
    else:
        compute_distance = lambda x1, x2: euclidean_distance(x1, x2, unit)

    f_compute_distance = F.udf(compute_distance, T.FloatType())

    col_prefix = (
        result_prefix
        if result_prefix
        else "_".join(list_of_cols_loc1) + "_" + "_".join(list_of_cols_loc2)
    )
    odf = idf.withColumn(
        col_prefix + "_distance", f_compute_distance("temp_loc1", "temp_loc2")
    ).drop("temp_loc1", "temp_loc2")

    if output_mode == "replace":
        odf = odf.drop(*(list_of_cols_loc1 + list_of_cols_loc2))

    return odf


def geohash_precision_control(
    idf, list_of_geohash, output_precision=8, km_max_error=None, output_mode="append"
):
    """
    This function controls the precision of input data's geohash columns.

    Parameters
    ----------
    idf
        Input Dataframe.
    list_of_geohash
        List of columns in geohash format e.g., ["gh1","gh2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "gh1|gh2".
    output_precision
        Precision of the transformed geohash in the output dataframe. (Default value = 8)
    km_max_error
        Maximum permissible error in kilometers. If km_max_error is specified, output_precision
        will be ignored and km_max_error will be mapped to an output_precision according to the
        following dictionary: {2500: 1, 630: 2, 78: 3, 20: 4, 2.4: 5, 0.61: 6, 0.076: 7,
        0.019: 8, 0.0024: 9, 0.00060: 10, 0.000074: 11}. (Default value = None)
    output_mode
        "replace", "append".
        "replace" option replaces original columns with transformed column.
        "append" option appends the transformed column to the input dataset
        with postfix "_precision_<output_precision>". (Default value = "append")

    Returns
    -------
    DataFrame
    """

    if isinstance(list_of_geohash, str):
        list_of_geohash = [x.strip() for x in list_of_geohash.split("|")]

    if any(x not in idf.columns for x in list_of_geohash):
        raise TypeError("Invalid input for list_of_geohash")

    error_precision_mapping = {
        2500: 1,
        630: 2,
        78: 3,
        20: 4,
        2.4: 5,
        0.61: 6,
        0.076: 7,
        0.019: 8,
        0.0024: 9,
        0.00060: 10,
        0.000074: 11,
    }
    if km_max_error is not None:
        output_precision = 12
        for key, val in error_precision_mapping.items():
            if km_max_error >= key:
                output_precision = val
                break
    output_precision = int(output_precision)
    logger.info(
        "Precision of the output geohashes will be capped at "
        + str(output_precision)
        + "."
    )
    odf = idf
    for i, geohash in enumerate(list_of_geohash):
        if output_mode == "replace":
            col_name = geohash
        else:
            col_name = geohash + "_precision_" + str(output_precision)
        odf = odf.withColumn(col_name, F.substring(geohash, 1, output_precision))

    return odf


def location_in_polygon(
    idf, list_of_lat, list_of_lon, polygon, result_prefix=[], output_mode="append"
):
    """
    This function checks whether each lat-lon pair is insided a GeoJSON object. The following types of GeoJSON objects
    are supported by this function: Polygon, MultiPolygon, Feature or FeatureCollection

    Parameters
    ----------
    idf
        Input Dataframe.
    list_of_lat
        List of columns representing latitude e.g., ["lat1","lat2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lat1|lat2".
    list_of_lon
        List of columns representing longitude e.g., ["lon1","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lon1|lon2".
        list_of_lon must have the same length as list_of_lat such that i-th element of
        list_of_lat and i-th element of list_of_lon form a lat-lon pair to format.
    polygon
        The following types of GeoJSON objects are supported: Polygon, MultiPolygon, Feature or FeatureCollection
    result_prefix
        List of prefixes for the newly generated column names.
        Alternatively, prefixes can be specified in a string format,
        where different prefixes are separated by pipe delimiter "|" e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_lat and list_of_lon.
        If it is empty, <lat>_<lon> will be used for each lat-lon pair.
        For example, list_of_lat is "lat1|lat2", list_of_lon is "lon1|lon2".
        Case 1: result_prefix = "L1|L2".
            New columns will be named as L1_in_poly and L2_in_poly.
        Calse 2: result_prefix = [].
            New columns will be named as lat1_lon1_in_poly and lat2_lon2_in_poly.
        (Default value = [])
    output_mode
        "replace", "append".
        "replace" option appends transformed column to the input dataset and removes the original ones.
        "append" option appends transformed column to the input dataset.
        (Default value = "append")

    Returns
    -------
    DataFrame
    """

    if isinstance(list_of_lat, str):
        list_of_lat = [x.strip() for x in list_of_lat.split("|")]
    if isinstance(list_of_lon, str):
        list_of_lon = [x.strip() for x in list_of_lon.split("|")]
    if isinstance(result_prefix, str):
        result_prefix = [x.strip() for x in result_prefix.split("|")]

    if any(x not in idf.columns for x in list_of_lat + list_of_lon):
        raise TypeError("Invalid input for list_of_lat or list_of_lon")

    if len(list_of_lat) != len(list_of_lon):
        raise TypeError("list_of_lat and list_of_lon must have the same length")
    if result_prefix and (len(result_prefix) != len(list_of_lat)):
        raise TypeError(
            "result_prefix must have the same length as list_of_lat and list_of_lon if it is not empty"
        )

    if "coordinates" in polygon.keys():
        polygon_list = polygon["coordinates"]
        if polygon["type"] == "Polygon":
            polygon_list = [polygon_list]
    elif "geometry" in polygon.keys():
        polygon_list = [polygon["geometry"]["coordinates"]]
    elif "features" in polygon.keys():
        polygon_list = []
        for poly in polygon["features"]:
            polygon_list.append(poly["geometry"]["coordinates"])

    odf = idf
    for i, (lat, lon) in enumerate(zip(list_of_lat, list_of_lon)):
        col = result_prefix[i] if result_prefix else (lat + "_" + lon)
        odf = odf.withColumn(
            col + "_in_poly", f_point_in_polygons(polygon_list)(F.col(lon), F.col(lat))
        )

        if output_mode == "replace":
            odf = odf.drop(lat, lon)

    return odf


def location_in_country(
    spark,
    idf,
    list_of_lat,
    list_of_lon,
    country,
    country_shapefile_path="",
    method_type="approx",
    result_prefix=[],
    output_mode="append",
):
    """
    This function checks whether each lat-lon pair is insided a country. Two ways of checking are supported: "approx" (using the
    bounding box of a country) and "exact" (using the shapefile of a country).

    Parameters
    ----------
    spark
        Spark Session
    idf
        Input Dataframe.
    list_of_lat
        List of columns representing latitude e.g., ["lat1","lat2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lat1|lat2".
    list_of_lon
        List of columns representing longitude e.g., ["lon1","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter "|" e.g., "lon1|lon2".
        list_of_lon must have the same length as list_of_lat such that i-th element of
        list_of_lat and i-th element of list_of_lon form a lat-lon pair to format.
    country
        The Alpha-2 country code.
    country_shapefile_path
        The geojson file with a FeatureCollection object containing polygons for each country. One example
        file country_polygons.geojson can be downloaded from Anovos GitHub repository:
        https://github.com/anovos/anovos/tree/main/data/
    method_type
        "approx", "exact".
        "approx" uses the bounding box of a country to estimate whether a location is inside the country
        "exact" uses the shapefile of a country to calculate whether a location is inside the country
    result_prefix
        List of prefixes for the newly generated column names.
        Alternatively, prefixes can be specified in a string format,
        where different prefixes are separated by pipe delimiter "|" e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_lat and list_of_lon.
        If it is empty, <lat>_<lon> will be used for each lat-lon pair.
        For example, list_of_lat is "lat1|lat2", list_of_lon is "lon1|lon2".
        Case 1: result_prefix = "L1|L2", country="US"
            New columns will be named as L1_in_US and L2_in_US.
        Calse 2: result_prefix = [], country="US"
            New columns will be named as lat1_lon1_in_US and lat2_lon2_in_US.
        (Default value = [])
    output_mode
        "replace", "append".
        "replace" option appends transformed column to the input dataset and removes the original ones.
        "append" option appends transformed column to the input dataset.
        (Default value = "append")

    Returns
    -------
    DataFrame
    """

    if isinstance(list_of_lat, str):
        list_of_lat = [x.strip() for x in list_of_lat.split("|")]
    if isinstance(list_of_lon, str):
        list_of_lon = [x.strip() for x in list_of_lon.split("|")]
    if isinstance(result_prefix, str):
        result_prefix = [x.strip() for x in result_prefix.split("|")]

    if any(x not in idf.columns for x in list_of_lat + list_of_lon):
        raise TypeError("Invalid input for list_of_lat or list_of_lon")

    if len(list_of_lat) != len(list_of_lon):
        raise TypeError("list_of_lat and list_of_lon must have the same length")
    if result_prefix and (len(result_prefix) != len(list_of_lat)):
        raise TypeError(
            "result_prefix must have the same length as list_of_lat and list_of_lon if it is not empty"
        )
    if method_type not in ("approx", "exact"):
        raise TypeError("Invalid input for method_type.")

    f_point_in_country_approx = F.udf(point_in_country_approx, T.IntegerType())

    if method_type == "exact":

        def zip_feats(x, y):
            """zipping two features (in list form) elementwise"""
            return zip(x, y)

        f_zip_feats = F.udf(
            zip_feats,
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("first", T.StringType()),
                        T.StructField(
                            "second",
                            T.ArrayType(
                                T.ArrayType(T.ArrayType(T.ArrayType(T.DoubleType())))
                            ),
                        ),
                    ]
                )
            ),
        )

        geo_data = spark.read.json(country_shapefile_path, multiLine=True).withColumn(
            "tmp",
            f_zip_feats("features.properties.ISO_A2", "features.geometry.coordinates"),
        )
        polygon_list = (
            geo_data.select(F.explode(F.col("tmp")).alias("country_coord"))
            .withColumn("country_code", F.col("country_coord").getItem("first"))
            .withColumn("coordinates", F.col("country_coord").getItem("second"))
            .where(F.col("country_code") == country)
            .select("coordinates")
            .rdd.map(lambda x: x[0])
            .collect()[0]
        )
        print("No. of polygon: " + str(len(polygon_list)))

        min_lon, min_lat = polygon_list[0][0][0]
        max_lon, max_lat = polygon_list[0][0][0]
        for polygon in polygon_list:
            exterior = polygon[0]
            for loc in exterior:
                if loc[0] < min_lon:
                    min_lon = loc[0]
                elif loc[0] > max_lon:
                    max_lon = loc[0]

                if loc[1] < min_lat:
                    min_lat = loc[1]
                elif loc[1] > max_lat:
                    max_lat = loc[1]

    odf = idf
    for i, (lat, lon) in enumerate(zip(list_of_lat, list_of_lon)):
        col = result_prefix[i] if result_prefix else (lat + "_" + lon)

        if method_type == "exact":
            odf = odf.withColumn(
                col + "_in_" + country + "_exact",
                f_point_in_polygons(
                    polygon_list, [min_lon, min_lat], [max_lon, max_lat]
                )(F.col(lon), F.col(lat)),
            )
        else:
            odf = odf.withColumn(
                col + "_in_" + country + "_approx",
                f_point_in_country_approx(F.col(lat), F.col(lon), F.lit(country)),
            )

        if output_mode == "replace":
            odf = odf.drop(lat, lon)

    return odf


def centroid(idf, lat_col, long_col, id_col=None):
    """
    This function calculates the centroid of a given DataFrame using lat-long pairs based on its identifier column (if applicable)

    Parameters
    ----------
    idf
        Input Dataframe
    lat_col
        Column in the input DataFrame that contains latitude data
    long_col
        Column in the input DataFrame that contains longitude data
    id_col
        Column in the input DataFrame that contains identifier for the DataFrame (Default is None)
        If id_col=None, the function will calculate centriod of latitude and longitude for the whole dataset.
    Returns
    -------
    odf : DataFrame
        Output DataFrame, which contains lat_centroid and long_centroid and identifier (if applicable)
    """
    if id_col not in idf.columns and id_col:
        raise TypeError("Invalid input for id_col")
    if lat_col not in idf.columns:
        raise TypeError("Invalid input for lat_col")
    if long_col not in idf.columns:
        raise TypeError("Invalid input for long_col")

    idf = recast_column(
        idf, list_of_cols=[lat_col, long_col], list_of_dtypes=["double", "double"]
    )

    if idf != idf.dropna(subset=(lat_col, long_col)):
        warnings.warn(
            "Rows dropped due to null value in longitude and/or latitude values"
        )
        idf = idf.dropna(subset=(lat_col, long_col))

    if not idf.where(
        (F.col(lat_col) > 90)
        | (F.col(lat_col) < -90)
        | (F.col(long_col) > 180)
        | (F.col(long_col) < -180)
    ).rdd.isEmpty():
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )
        idf = idf.where(
            (F.col(lat_col) <= 90)
            & (F.col(lat_col) >= -90)
            & (F.col(long_col) <= 180)
            & (F.col(long_col) >= -180)
        )

    if idf.rdd.isEmpty():
        warnings.warn(
            "No reverse_geocoding Computation - No valid latitude/longitude row(s) to compute"
        )
        return idf

    def degree_to_radian(deg):
        return deg * pi / 180

    f_degree_to_radian = F.udf(degree_to_radian, T.FloatType())

    idf_rad = (
        idf.withColumn("lat_rad", f_degree_to_radian(lat_col))
        .withColumn("long_rad", f_degree_to_radian(long_col))
        .withColumn("x", F.cos("lat_rad") * F.cos("long_rad"))
        .withColumn("y", F.cos("lat_rad") * F.sin("long_rad"))
        .withColumn("z", F.sin("lat_rad"))
    )
    if id_col:
        idf_groupby = idf_rad.groupby(id_col).agg(
            F.sum("x").alias("x_group"),
            F.sum("y").alias("y_group"),
            F.sum("z").alias("z_group"),
        )

        odf = (
            idf_groupby.withColumn(
                "hyp",
                F.sqrt(
                    F.col("x_group") * F.col("x_group")
                    + F.col("y_group") * F.col("y_group")
                ),
            )
            .withColumn(
                lat_col + "_centroid",
                F.atan2(F.col("z_group"), F.col("hyp")) * 180 / pi,
            )
            .withColumn(
                long_col + "_centroid",
                F.atan2(F.col("y_group"), F.col("x_group")) * 180 / pi,
            )
            .select(id_col, lat_col + "_centroid", long_col + "_centroid")
        )
    else:
        idf_groupby = idf_rad.groupby().agg(
            F.sum("x").alias("x_group"),
            F.sum("y").alias("y_group"),
            F.sum("z").alias("z_group"),
        )

        odf = (
            idf_groupby.withColumn(
                "hyp",
                F.sqrt(
                    F.col("x_group") * F.col("x_group")
                    + F.col("y_group") * F.col("y_group")
                ),
            )
            .withColumn(
                lat_col + "_centroid",
                F.atan2(F.col("z_group"), F.col("hyp")) * 180 / pi,
            )
            .withColumn(
                long_col + "_centroid",
                F.atan2(F.col("y_group"), F.col("x_group")) * 180 / pi,
            )
            .select(lat_col + "_centroid", long_col + "_centroid")
        )
    return odf


def weighted_centroid(idf, id_col, lat_col, long_col):
    """
    This function calculates the weighted centroid of a given DataFrame using lat-long pairs based on its identifier column

    Parameters
    ----------
    idf
        Input Dataframe
    lat_col
        Column in the input DataFrame that contains latitude data
    long_col
        Column in the input DataFrame that contains longitude data
    id_col
        Column in the input DataFrame that contains identifier for the DataFrame
    Returns
    -------
    odf : DataFrame
        Output DataFrame, which contains weighted lat_centroid and long_centroid and identifier
    """
    if id_col not in idf.columns:
        raise TypeError("Invalid input for id_col")
    if lat_col not in idf.columns:
        raise TypeError("Invalid input for lat_col")
    if long_col not in idf.columns:
        raise TypeError("Invalid input for long_col")

    idf = recast_column(
        idf, list_of_cols=[lat_col, long_col], list_of_dtypes=["double", "double"]
    )

    if idf != idf.dropna(subset=(lat_col, long_col)):
        warnings.warn(
            "Rows dropped due to null value in longitude and/or latitude values"
        )
        idf = idf.dropna(subset=(lat_col, long_col))

    if not idf.where(
        (F.col(lat_col) > 90)
        | (F.col(lat_col) < -90)
        | (F.col(long_col) > 180)
        | (F.col(long_col) < -180)
    ).rdd.isEmpty():
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )
        idf = idf.where(
            (F.col(lat_col) <= 90)
            & (F.col(lat_col) >= -90)
            & (F.col(long_col) <= 180)
            & (F.col(long_col) >= -180)
        )

    if idf.rdd.isEmpty():
        warnings.warn(
            "No reverse_geocoding Computation - No valid latitude/longitude row(s) to compute"
        )
        return idf

    def degree_to_radian(deg):
        return deg * pi / 180

    f_degree_to_radian = F.udf(degree_to_radian, T.FloatType())

    idf_rad = (
        idf.withColumn("lat_rad", f_degree_to_radian(lat_col))
        .withColumn("long_rad", f_degree_to_radian(long_col))
        .withColumn("x", F.cos("lat_rad") * F.cos("long_rad"))
        .withColumn("y", F.cos("lat_rad") * F.sin("long_rad"))
        .withColumn("z", F.sin("lat_rad"))
    )

    idf_groupby = (
        idf_rad.groupby(id_col)
        .agg(
            F.sum("x").alias("x_group"),
            F.sum("y").alias("y_group"),
            F.sum("z").alias("z_group"),
            F.count(id_col).alias("weight_group"),
        )
        .withColumn("weighted_x", F.col("x_group") * F.col("weight_group"))
        .withColumn("weighted_y", F.col("y_group") * F.col("weight_group"))
        .withColumn("weighted_z", F.col("z_group") * F.col("weight_group"))
    )

    total_weight = (
        idf_groupby.groupBy()
        .agg(F.sum("weight_group"))
        .rdd.map(lambda x: x[0])
        .collect()[0]
    )
    total_x = (
        idf_groupby.groupBy()
        .agg(F.sum("weighted_x"))
        .rdd.map(lambda x: x[0])
        .collect()[0]
    )
    total_y = (
        idf_groupby.groupBy()
        .agg(F.sum("weighted_y"))
        .rdd.map(lambda x: x[0])
        .collect()[0]
    )
    total_z = (
        idf_groupby.groupBy()
        .agg(F.sum("weighted_z"))
        .rdd.map(lambda x: x[0])
        .collect()[0]
    )

    x = total_x / total_weight
    y = total_y / total_weight
    z = total_z / total_weight
    hyp = sqrt(x * x + y * y)
    lat_centroid, long_centroid = atan2(z, hyp) * 180 / pi, atan2(y, x) * 180 / pi

    odf = (
        idf_groupby.select(id_col)
        .withColumn(lat_col + "_centroid", F.lit(lat_centroid))
        .withColumn(long_col + "_centroid", F.lit(long_centroid))
    )

    return odf


def rog_calculation(idf, lat_col, long_col, id_col=None):
    """
    This function calculates the Radius of Gyration (in meter) of a given DataFrame, based on its identifier column (if applicable)
    Parameters
    ----------
    idf
        Input Dataframe
    lat_col
        Column in the input DataFrame that contains latitude data
    long_col
        Column in the input DataFrame that contains longitude data
    id_col
        Column in the input DataFrame that contains identifier for the DataFrame (Default is None)
    Returns
    -------
    odf : DataFrame
        Output DataFrame, which contains Radius of Gyration (in meter) and identifier (if applicable)
    """
    if id_col not in idf.columns and id_col:
        raise TypeError("Invalid input for id_col")
    if lat_col not in idf.columns:
        raise TypeError("Invalid input for lat_col")
    if long_col not in idf.columns:
        raise TypeError("Invalid input for long_col")

    idf = recast_column(
        idf, list_of_cols=[lat_col, long_col], list_of_dtypes=["double", "double"]
    )

    if idf != idf.dropna(subset=(lat_col, long_col)):
        warnings.warn(
            "Rows dropped due to null value in longitude and/or latitude values"
        )
        idf = idf.dropna(subset=(lat_col, long_col))

    if not idf.where(
        (F.col(lat_col) > 90)
        | (F.col(lat_col) < -90)
        | (F.col(long_col) > 180)
        | (F.col(long_col) < -180)
    ).rdd.isEmpty():
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )
        idf = idf.where(
            (F.col(lat_col) <= 90)
            & (F.col(lat_col) >= -90)
            & (F.col(long_col) <= 180)
            & (F.col(long_col) >= -180)
        )

    if idf.rdd.isEmpty():
        warnings.warn(
            "No reverse_geocoding Computation - No valid latitude/longitude row(s) to compute"
        )
        return idf

    def getHaversineDist(lat1, lon1, lat2, lon2):

        R = 6378126  # approximate radius of earth in m

        lat1 = radians(float(lat1))
        lon1 = radians(float(lon1))
        lat2 = radians(float(lat2))
        lon2 = radians(float(lon2))

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    udf_harver_dist = F.udf(getHaversineDist, T.FloatType())

    if id_col:
        idf_centroid = centroid(idf, lat_col, long_col, id_col)
        idf_join = idf_centroid.join(idf, id_col, "inner")
        idf_calc = idf_join.withColumn(
            "distance",
            udf_harver_dist(
                F.col(lat_col),
                F.col(long_col),
                F.col(lat_col + "_centroid"),
                F.col(long_col + "_centroid"),
            ),
        )

        odf = idf_calc.groupby(id_col).agg(
            F.mean("distance").alias("radius_of_gyration")
        )
    else:
        centroid_info = centroid(idf, lat_col, long_col, id_col).rdd.collect()
        lat_centroid = centroid_info[0][0]
        long_centroid = centroid_info[0][1]
        idf_join = idf.withColumn(
            lat_col + "_centroid", F.lit(lat_centroid)
        ).withColumn(long_col + "_centroid", F.lit(long_centroid))
        idf_calc = idf_join.withColumn(
            "distance",
            udf_harver_dist(
                F.col(lat_col),
                F.col(long_col),
                F.col(lat_col + "_centroid"),
                F.col(long_col + "_centroid"),
            ),
        )

        odf = idf_calc.groupby().agg(F.mean("distance").alias("radius_of_gyration"))
    return odf


def reverse_geocoding(idf, lat_col, long_col):
    """
    This function reverses the input latitude and longitude of a given DataFrame into address
    Parameters
    ----------
    idf
        Input Dataframe
    lat_col
        Column in the input DataFrame that contains latitude data
    long_col
        Column in the input DataFrame that contains longitude data
    Returns
    -------
    odf : DataFrame
        Output DataFrame, which contains latitude, longitude and address appropriately
    """
    if lat_col not in idf.columns:
        raise TypeError("Invalid input for lat_col")
    if long_col not in idf.columns:
        raise TypeError("Invalid input for long_col")

    idf = recast_column(
        idf, list_of_cols=[lat_col, long_col], list_of_dtypes=["double", "double"]
    )

    if idf != idf.dropna(subset=(lat_col, long_col)):
        warnings.warn(
            "Rows dropped due to null value in longitude and/or latitude values"
        )
        idf = idf.dropna(subset=(lat_col, long_col))

    if not idf.where(
        (F.col(lat_col) > 90)
        | (F.col(lat_col) < -90)
        | (F.col(long_col) > 180)
        | (F.col(long_col) < -180)
    ).rdd.isEmpty():
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )
        idf = idf.where(
            (F.col(lat_col) <= 90)
            & (F.col(lat_col) >= -90)
            & (F.col(long_col) <= 180)
            & (F.col(long_col) >= -180)
        )

    if idf.rdd.isEmpty():
        warnings.warn(
            "No reverse_geocoding Computation - No valid latitude/longitude row(s) to compute"
        )
        return idf

    def reverse_geocode(lat, long):
        coordinates = (float(lat), float(long))
        location = rg.search(coordinates, mode=1)
        if location:
            return (
                str(location[0]["name"])
                + ","
                + str(location[0]["admin1"])
                + ","
                + str(location[0]["cc"])
            )
        else:
            return "N/A"

    udf_reverse_geocode = F.udf(reverse_geocode)
    odf = (
        idf.withColumn("info", udf_reverse_geocode(F.col(lat_col), F.col(long_col)))
        .select(lat_col, long_col, "info")
        .withColumn("name_of_place", F.split(F.col("info"), ",").getItem(0))
        .withColumn("region", F.split(F.col("info"), ",").getItem(1))
        .withColumn("country_code", F.split(F.col("info"), ",").getItem(2))
        .drop("info")
    )
    return odf
