import warnings

from pyspark.sql import functions as F
from pyspark.sql import types as T
from math import sin, cos, sqrt, atan2, radians, pi
from anovos.data_ingest.data_ingest import recast_column, delete_column


def centroid(idf, lat_col, long_col, id_col=None):
    """
    Calculate centroid of a given DataFrame
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

    invalid = False
    for x in idf.collect():
        if x[lat_col] is None or x[long_col] is None:
            invalid = True
            break

    if invalid:
        warnings.warn("Rows dropped due to invalid longitude and/or latitude values")

    idf = idf.withColumn("lat_null", idf[lat_col].contains("")).withColumn(
        "long_null", idf[long_col].contains("")
    )

    idf = delete_column(
        idf.where((idf["lat_null"]) & (idf["long_null"])), ["lat_null", "long_null"]
    )

    out_of_range = False
    for x in idf.collect():
        if (
            x[lat_col] > 90
            or x[lat_col] < -90
            or x[long_col] > 180
            or x[long_col] < -180
        ):
            out_of_range = True
            break

    if out_of_range:
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )

    idf = idf.where(
        ((idf[lat_col] <= 90) & (idf[lat_col] >= -90))
        & ((idf[long_col] <= 180) & (idf[long_col] >= -180))
    )

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
                "lat_centroid", F.atan2(F.col("z_group"), F.col("hyp")) * 180 / pi
            )
            .withColumn(
                "long_centroid", F.atan2(F.col("y_group"), F.col("x_group")) * 180 / pi
            )
            .select(id_col, "lat_centroid", "long_centroid")
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
                "lat_centroid", F.atan2(F.col("z_group"), F.col("hyp")) * 180 / pi
            )
            .withColumn(
                "long_centroid", F.atan2(F.col("y_group"), F.col("x_group")) * 180 / pi
            )
            .select("lat_centroid", "long_centroid")
        )
    return odf


def weighted_centroid(idf, id_col, lat_col, long_col):
    """
    Calculate weighted centroid of a given DataFrame, based on its identifier column
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

    invalid = False
    for x in idf.collect():
        if x[lat_col] is None or x[long_col] is None:
            invalid = True
            break

    if invalid:
        warnings.warn("Rows dropped due to invalid longitude and/or latitude values")

    idf = idf.withColumn("lat_null", idf[lat_col].contains("")).withColumn(
        "long_null", idf[long_col].contains("")
    )

    idf = delete_column(
        idf.where((idf["lat_null"]) & (idf["long_null"])), ["lat_null", "long_null"]
    )

    out_of_range = False
    for x in idf.collect():
        if (
            x[lat_col] > 90
            or x[lat_col] < -90
            or x[long_col] > 180
            or x[long_col] < -180
        ):
            out_of_range = True
            break

    if out_of_range:
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )

    idf = idf.where(
        ((idf[lat_col] <= 90) & (idf[lat_col] >= -90))
        & ((idf[long_col] <= 180) & (idf[long_col] >= -180))
    )

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
        .withColumn("lat_centroid", F.lit(lat_centroid))
        .withColumn("long_centroid", F.lit(long_centroid))
    )

    return odf


def rog_calculation(idf, lat_col, long_col, id_col=None):
    """
    Calculate Radius of Gyration (in meter) of a given DataFrame, based on its identifier column (if applicable)
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

    invalid = False
    for x in idf.collect():
        if x[lat_col] is None or x[long_col] is None:
            invalid = True
            break

    if invalid:
        warnings.warn("Rows dropped due to invalid longitude and/or latitude values")

    idf = idf.withColumn("lat_null", idf[lat_col].contains("")).withColumn(
        "long_null", idf[long_col].contains("")
    )

    idf = delete_column(
        idf.where((idf["lat_null"]) & (idf["long_null"])), ["lat_null", "long_null"]
    )

    out_of_range = False
    for x in idf.collect():
        if (
            x[lat_col] > 90
            or x[lat_col] < -90
            or x[long_col] > 180
            or x[long_col] < -180
        ):
            out_of_range = True
            break

    if out_of_range:
        warnings.warn(
            "Rows dropped due to longitude and/or latitude values being out of the valid range"
        )

    idf = idf.where(
        ((idf[lat_col] <= 90) & (idf[lat_col] >= -90))
        & ((idf[long_col] <= 180) & (idf[long_col] >= -180))
    )

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
                F.col("lat_centroid"),
                F.col("long_centroid"),
            ),
        )

        odf = idf_calc.groupby(id_col).agg(
            F.mean("distance").alias("radius_of_gyration")
        )
    else:
        centroid_info = centroid(idf, lat_col, long_col, id_col).rdd.collect()
        lat_centroid = centroid_info[0][0]
        long_centroid = centroid_info[0][1]
        idf_join = idf.withColumn("lat_centroid", F.lit(lat_centroid)).withColumn(
            "long_centroid", F.lit(long_centroid)
        )
        idf_calc = idf_join.withColumn(
            "distance",
            udf_harver_dist(
                F.col(lat_col),
                F.col(long_col),
                F.col("lat_centroid"),
                F.col("long_centroid"),
            ),
        )

        odf = idf_calc.groupby().agg(F.mean("distance").alias("radius_of_gyration"))
    return odf
