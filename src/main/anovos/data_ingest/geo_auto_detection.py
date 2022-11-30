# coding=utf-8

"""This module help to automatically identify the latitude, longitude as well as geohash columns present in the analysis data through some intelligent checks provisioned as a part of the module.

As a part of generation of the auto detection output, there are various functions created such as -

- reg_lat_lon
- conv_str_plus
- precision_lev
- geo_to_latlong
- latlong_to_geo
- ll_gh_cols

Respective functions have sections containing the detailed definition of the parameters used for computing.

"""

import pygeohash as gh
from pyspark.sql import functions as F
from pyspark.sql import types as T


def reg_lat_lon(option):

    """
    This function helps to produce the relevant regular expression to be used for further processing based on the input field category - latitude / longitude

    Parameters
    ----------

    option
        Can be either latitude or longitude basis which the desired regular expression will be produced

    Returns
    -------
    Regular Expression
    """

    if option == "latitude":
        return "^(\+|-|)?(?:90(?:(?:\.0{1,10})?)|(?:[0-9]|[1-8][0-9])(?:(?:\.[0-9]{1,})?))$"
    elif option == "longitude":
        return "^(\+|-)?(?:180(?:(?:\.0{1,10})?)|(?:[0-9]|[1-9][0-9]|1[0-7][0-9])(?:(?:\.[0-9]{1,10})?))$"


def conv_str_plus(col):

    """
    This function helps to produce an extra "+" to the positive values while negative values are kept as is

    Parameters
    ----------

    col
        Analysis column

    Returns
    -------
    String
    """

    if col is None:
        return None
    elif col < 0:
        return col
    else:
        return "+" + str(col)


f_conv_str_plus = F.udf(conv_str_plus, T.StringType())


def precision_lev(col):

    """
    This function helps to capture the precision level after decimal point.

    Parameters
    ----------

    col
        Analysis column

    Returns
    -------
    String
    """

    if col is None:
        return 0
    else:
        x = float(str(format(float(col), ".8f")).split(".")[1])
        if x > 0:
            return len(str(format(float(col), ".8f")).split(".")[1])
        else:
            return 0


f_precision_lev = F.udf(precision_lev, T.StringType())


def geo_to_latlong(x, option):

    """
    This function helps to convert geohash to latitude / longitude with the help of pygeohash library.

    Parameters
    ----------

    x
        Analysis column
    option
        Can be either 0 or 1 basis which the latitude / longitude will be produced

    Returns
    -------
    Float
    """

    if x is not None:

        if option == 0:
            try:
                return [float(a) for a in gh.decode(x)][option]
            except:
                pass
        elif option == 1:
            try:
                return [float(a) for a in gh.decode(x)][option]
            except:
                pass

        else:
            return None

    else:

        return None


f_geo_to_latlong = F.udf(geo_to_latlong, T.FloatType())


def latlong_to_geo(lat, long, precision=9):

    """
    This function helps to convert latitude-longitude to geohash with the help of pygeohash library.

    Parameters
    ----------

    lat
        latitude column

    long
        longitude column

    precision
        precision at which the geohash is converted to

    Returns
    -------
    Regular String
    """

    if (lat is not None) and (long is not None):

        return gh.encode(lat, long, precision)

    else:

        return None


f_latlong_to_geo = F.udf(latlong_to_geo, T.StringType())


def ll_gh_cols(df, max_records):

    """
    This function is the main function to auto-detect latitude, longitude and geohash columns from a given dataset df.
    To detect latitude and longitude columns, it will check whether "latitude" or "longitude" appears in the columns.
    If not, it will calculate the precision level, maximum, standard deviation and mean value of each float or double-type
    column, and convert to string type by calling "conv_str_plus".
        If the converted string matches regular expression of latitude and the absolute value of maximum is <= 90,
        then it will be regarded as latitude column.
        If the converted string matches regular expression of longitude and the absolute value of maximum is > 90, then
        it will be regarded as longitude column.
    To detect geohash column, it will calculate the maximum string-length of every string column, and convert it to
    lat-long pairs by calling "geo_to_lat_long". If the conversion is successful and the maximum string-length is
    between 4 and 12 (exclusive), this string column will be regarded as geohash column.

    Parameters
    ----------

    df
        Analysis dataframe

    max_records

        Maximum geospatial points analyzed

    Returns
    -------
    List

    """

    lat_cols, long_cols, gh_cols = [], [], []
    for i in df.dtypes:
        if i[1] in ("float", "double", "float32", "float64"):

            c = 0

            prec_val = (
                df.withColumn("__", f_precision_lev(F.col(i[0])))
                .agg(F.max("__"))
                .rdd.flatMap(lambda x: x)
                .collect()[0]
            )

            max_val = (
                df.withColumn("__", F.col(i[0]))
                .agg(F.max("__"))
                .rdd.flatMap(lambda x: x)
                .collect()[0]
            )
            stddev_val = (
                df.agg(F.stddev(F.col(i[0]))).rdd.flatMap(lambda x: x).collect()[0]
            )

            mean_val = df.agg(F.mean(F.col(i[0]))).rdd.flatMap(lambda x: x).collect()[0]

            if prec_val is None:
                p = 0
            elif prec_val == 0:
                p = 0
            else:
                p = prec_val
            if "latitude" in i[0].lower():
                lat_cols.append(i[0])
            elif "longitude" in i[0].lower():
                long_cols.append(i[0])
            elif (
                (int(p) > 0)
                & (max_val <= 180)
                & (stddev_val >= 1)
                & (float(stddev_val) / float(mean_val) < 1)
            ):
                for j in [reg_lat_lon("latitude"), reg_lat_lon("longitude")]:
                    if c == 0:
                        x = (
                            df.select(F.col(i[0]))
                            .dropna()
                            .withColumn(
                                "_", F.regexp_extract(f_conv_str_plus(i[0]), j, 0)
                            )
                        )

                        max_val = abs(
                            float(
                                x.agg(F.max(i[0])).rdd.flatMap(lambda x: x).collect()[0]
                            )
                        )
                        if (x.groupBy("_").count().count() > 2) & (max_val <= 90):
                            lat_cols.append(i[0])
                            c = c + 1

                        elif (x.groupBy("_").count().count() > 2) & (max_val > 90):
                            long_cols.append(i[0])
                            c = c + 1

        elif i[1] in ("string", "object"):
            x = (
                df.select(F.col(i[0]))
                .dropna()
                .limit(max_records)
                .withColumn("len_gh", F.length(F.col(i[0])))
            )
            x_ = x.agg(F.max("len_gh")).rdd.flatMap(lambda x: x).collect()[0]
            try:
                if (x_ > 4) & (x_ < 12):
                    if (
                        x.withColumn("_", f_geo_to_latlong(i[0], F.lit(0)))
                        .groupBy("_")
                        .count()
                        .count()
                        > 2
                    ):
                        gh_cols.append(i[0])
                    else:
                        pass
            except:
                pass

    if len(lat_cols) != len(long_cols):
        lat_cols, long_cols = [], []

    return lat_cols, long_cols, gh_cols
