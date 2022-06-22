from .geo_utils import (
    EARTH_RADIUS,
    from_latlon_decimal_degrees,
    to_latlon_decimal_degrees,
    haversine_distance,
    vincenty_distance,
    euclidean_distance,
    f_point_in_polygons,
)
from pyspark.sql import functions as F
from pyspark.sql import types as T
from loguru import logger


def geo_format_latlon(
    idf,
    list_of_lat,
    list_of_lon,
    input_format,
    output_format,
    result_prefix=None,
    geohash_precision=8,
    radius=EARTH_RADIUS,
    output_mode="append",
):
    """
    Convert locations from lat, lon format to other formats.

    Parameters
    ----------

    idf
        Input Dataframe.
    list_of_lat
        List of columns representing latitude e.g., ["lat1","lat2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "lat1|lat2".
    list_of_lon
        List of columns representing longitude e.g., ["lon1","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "lon1|lon2".
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
        where different prefixes are separated by pipe delimiter “|” e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_lat and list_of_lon.
        If it is None, <lat>_<lon> will be used for each lat-lon pair.
        For example, list_of_lat is "lat1|lat2", list_of_lon is "L1|L2".
        Case 1: result_prefix = "L1|L2".
            If output_format is "dd", "dms" or "radian", new columns added will be
            L1_lat_<output_format>, L1_lon_<output_format>, L2_lat_<output_format>, L2_lon_<output_format>.
            If output_format is "cartesian", new columns added will be
            L1_x, L1_y, L1_z, L2_x, L2_y, L2_z.
            If output_format is "geohash", new columns added will be
            L1_geohash and L2_geohash.
        Calse 2: result_prefix = None.
            The "L1" and "L2" in above column names will be replaced by "lat1_lon1" and "lat2_lon2".
        (Default value = None)
    geohash_precision
        Precision of the resultant geohash.
        This argument is only used when output_format is "geohash". (Default value = 8)
    radius
        Radius of Earth.
        Necessary only when input_format or output_format is "cartesian". (Default value = EARTH_RADIUS).
    output_mode
        "replace", "append".
        "replace" option appends transformed column to the input dataset and removes the original ones.
        "append" option appends transformed column to the input dataset.
        (Default value = "append")

    Returns
    -------
    DataFrame
    """
    geohash_precision = int(geohash_precision)

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
    if len(result_prefix) != len(list_of_lat):
        raise TypeError(
            "result_prefix must have the same length as list_of_lat and list_of_lon"
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
        col = result_prefix[i] if result_prefix is not None else (lat + "_" + lon)

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
    result_prefix=None,
    geohash_precision=8,
    radius=EARTH_RADIUS,
    output_mode="append",
):

    geohash_precision = int(geohash_precision)

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

    if len(set([len(list_of_x), len(list_of_y), len(list_of_z)])) != 1:
        raise TypeError("list_of_x, list_of_y and list_of_z must have the same length")
    if len(result_prefix) != len(list_of_x):
        raise TypeError(
            "result_prefix must have the same length as list_of_x, list_of_y and list_of_y"
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
        col = result_prefix[i] if result_prefix is not None else (x + "_" + y + "_" + z)

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
    result_prefix=None,
    geohash_precision=8,
    radius=EARTH_RADIUS,
    output_mode="append",
):

    geohash_precision = int(geohash_precision)

    if isinstance(list_of_geohash, str):
        list_of_geohash = [x.strip() for x in list_of_geohash.split("|")]
    if isinstance(result_prefix, str):
        result_prefix = [x.strip() for x in result_prefix.split("|")]

    if any(x not in idf.columns for x in list_of_geohash):
        raise TypeError("Invalid input for list_of_geohash")

    format_list = ["dd", "dms", "radian", "cartesian"]
    if output_format not in format_list:
        raise TypeError("Invalid input for output_format")

    if len(result_prefix) != len(list_of_geohash):
        raise TypeError("result_prefix must have the same length as list_of_geohash")

    f_to_latlon_dd = F.udf(
        lambda loc: to_latlon_decimal_degrees(loc, "geohash", radius),
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

    odf = idf
    for i, geohash in enumerate(list_of_geohash):
        col = result_prefix[i] if result_prefix is not None else geohash

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
    result_prefix=None,
    distance_type="haversine",
    unit="m",
    radius=EARTH_RADIUS,
    vincenty_model="WGS-84",
    output_mode="append",
):

    """
    Parameters
    ----------

    idf
        Input Dataframe.
    list_of_cols_loc1
        List of columns to express the first location e.g., ["lat1","lon1"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "lat1|lon1".
    list_of_cols_loc2
        List of columns to express the second location e.g., ["lat2","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "lat2|lon2".
    loc_format
        "dd", "dms", "radian", "cartesian", "geohash". (Default value = "dd")
    distance_type
        "vincenty", "haversine", "euclidean". (Default value = "haversine")
        "vincenty" option calculates the distance between two points on the surface of a spheroid.
        "haversine" option calculates the great-circle distance between two points on a sphere.
        "euclidean" option calculates the length of the line segment between two points.
    unit
        "m", "km".
        Unit of the result. (Default value = "m")
    vincenty_model
        "WGS-84", "GRS-80", "Airy (1830)", "Intl 1924", "Clarke (1880)", "GRS-67".
        The ellipsoidal model to use. For more information, please refer to geopy.distance.ELLIPSOIDS.
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

    if isinstance(list_of_cols_loc1, str):
        list_of_cols_loc1 = [x.strip() for x in list_of_cols_loc1.split("|")]
    if isinstance(list_of_cols_loc2, str):
        list_of_cols_loc2 = [x.strip() for x in list_of_cols_loc2.split("|")]

    if any(i not in idf.columns for i in list_of_cols_loc1 + list_of_cols_loc2):
        raise TypeError("Invalid input for list_of_cols_loc1 or list_of_cols_loc2")

    if distance_type not in ["vincenty", "haversine", "euclidean"]:
        raise TypeError("Invalid input for distance_type")

    if loc_format not in ["dd", "dms", "radian", "cartesian", "geohash"]:
        raise TypeError("Invalid input for loc_format")

    format_mapping = {"vincenty": "dd", "haversine": "radian", "euclidean": "cartesian"}
    format_required = format_mapping[distance_type]

    # dd/radian/dms: [lat1, lon1, lat2, lon2]
    # cartesian: [x1, y1, z1, x2, y2, z2]
    # geohasg: [gh1, gh2]

    if loc_format != format_required:
        # format_required
        # dd: + cols [temp_loc1_lat_dd, temp_loc1_lon_dd, temp_loc2_lat_dd, temp_loc2_lon_dd]
        # radian: + cols [temp_loc1_lat_radian, temp_loc1_lon_radian, temp_loc2_lat_radian, temp_loc2_lon_radian]
        # cartesian: + cols [temp_loc1_x, temp_loc1_y, temp_loc1_z, temp_loc2_x, temp_loc2_y, temp_loc2_z]
        if loc_format in ["dd", "dms", "radian"]:
            # list_of_cols_loc1 = [lat1, lon1]
            # list_of_cols_loc2 = [lat2, lon2]
            idf = geo_format_latlon(
                idf,
                list_of_lat=[list_of_cols_loc1[0], list_of_cols_loc2[0]],
                list_of_lon=[list_of_cols_loc1[1], list_of_cols_loc2[1]],
                input_format=loc_format,
                output_format=format_required,
                result_prefix=["temp_loc1", "temp_loc2"],
                radius=radius,
                output_mode="append",
            )

        elif loc_format == "cartesian":
            # list_of_cols_loc1 = [x1, y1, z1]
            # list_of_cols_loc2 = [x2, y2, z2]
            idf = geo_format_cartesian(
                idf,
                list_of_x=[list_of_cols_loc1[0], list_of_cols_loc2[0]],
                list_of_y=[list_of_cols_loc1[1], list_of_cols_loc2[1]],
                list_of_z=[list_of_cols_loc1[2], list_of_cols_loc2[2]],
                output_format=format_required,
                result_prefix=["temp_loc1", "temp_loc2"],
                radius=radius,
                output_mode="append",
            )

        elif loc_format == "geohash":
            # list_of_cols_loc1 = [gh1]
            # list_of_cols_loc2 = [gh2]
            idf = geo_format_geohash(
                idf,
                list_of_geohash=[list_of_cols_loc1[0], list_of_cols_loc2[0]],
                output_format=format_required,
                result_prefix=["temp_loc1", "temp_loc2"],
                radius=radius,
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
        compute_distance = lambda x1, x2: euclidean_distance(x1, x2)

    f_compute_distance = F.udf(compute_distance, T.FloatType())

    col_prefix = (
        result_prefix
        if result_prefix is not None
        else "_".join(list_of_cols_loc1) + "_" + "_".join(list_of_cols_loc2)
    )
    odf = idf.withColumn(
        col_prefix + "_" + "distance", f_compute_distance("temp_loc1", "temp_loc2")
    ).drop("temp_loc1", "temp_loc2")

    if output_mode == "replace":
        odf = odf.drop(*(list_of_cols_loc1 + list_of_cols_loc2))

    return odf


def geohash_precision_control(
    idf, list_of_geohash, output_precision=8, km_max_error=None, output_mode="append"
):

    """
    Parameters
    ----------

    idf
        Input Dataframe.
    list_of_geohash
        List of columns in geohash format e.g., ["gh1","gh2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "gh1|gh2".
    output_precision
        Precision of the transformed geohash in the output dataframe. (Default value = 8)
    km_max_error
        Maximum permissible error in kilometers. If km_max_error is sprcified, output_precision
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
    idf, list_of_lat, list_of_lon, polygon, result_prefix=None, output_mode="replace"
):
    """
    To check whether each lat-lon pair is insided of a GeoJSON object

    Parameters
    ----------

    idf
        Input Dataframe.
    list_of_lat
        List of columns representing latitude e.g., ["lat1","lat2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "lat1|lat2".
    list_of_lon
        List of columns representing longitude e.g., ["lon1","lon2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "lon1|lon2".
        list_of_lon must have the same length as list_of_lat such that i-th element of
        list_of_lat and i-th element of list_of_lon form a lat-lon pair to format.
    polygon
        The following types of GeoJSON objects are supported: Polygon, MultiPolygon, Feature or FeatureCollection
    output_format
        "dd", "dms", "radian", "cartesian", "geohash".
        "cartesian" represents the Cartesian coordinates of the point in three-dimensional space.
        "geohash" represents geocoded locations.
    result_prefix
        List of prefixes for the newly generated column names.
        Alternatively, prefixes can be specified in a string format,
        where different prefixes are separated by pipe delimiter “|” e.g., "pf1|pf2".
        result_prefix must have the same length as list_of_lat and list_of_lon.
        If it is None, <lat>_<lon> will be used for each lat-lon pair.
        For example, list_of_lat is "lat1|lat2", list_of_lon is "L1|L2".
        Case 1: result_prefix = "L1|L2".
            New columns will be named as L1_in_poly and L2_in_poly.
        Calse 2: result_prefix = None.
             New columns will be named as lat1_lon1_in_poly and lat2_lon2_in_poly.
        (Default value = None)
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
    if (result_prefix is not None) and (len(result_prefix) != len(list_of_lat)):
        raise TypeError(
            "result_prefix must have the same length as list_of_lat and list_of_lon"
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
        col = result_prefix[i] if result_prefix is not None else (lat + "_" + lon)
        odf = odf.withColumn(
            col + "_in_poly", f_point_in_polygons(polygon_list)(F.col(lon), F.col(lat))
        )

        if output_mode == "replace":
            odf = odf.drop(lat, lon)

    return odf
