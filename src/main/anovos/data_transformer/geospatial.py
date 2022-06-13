from pyspark.sql import functions as F
from pyspark.sql import types as T
from math import pi, sqrt, atan2


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
