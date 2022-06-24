import pygeohash as gh
from pyspark.sql import functions as F
from pyspark.sql import types as T


def reg_lat_lon(option):
    if option == "latitude":
        return "^(\+|-|)?(?:90(?:(?:\.0{1,10})?)|(?:[0-9]|[1-8][0-9])(?:(?:\.[0-9]{1,})?))$"
    elif option == "longitude":
        return "^(\+|-)?(?:180(?:(?:\.0{1,10})?)|(?:[0-9]|[1-9][0-9]|1[0-7][0-9])(?:(?:\.[0-9]{1,10})?))$"


def conv_str_plus(col):
    if col is None:
        return None
    elif col < 0:
        return col
    else:
        return "+" + str(col)


f_conv_str_plus = F.udf(conv_str_plus, T.StringType())


def precision_lev(col):
    if col is None:
        return None
    else:
        return int(str(float(col)).split(".")[1])


f_precision_lev = F.udf(precision_lev, T.IntegerType())


def geo_to_latlong(x, option):

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


def ll_gh_cols(df, max_records=100000):

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

            if prec_val is None:
                p = 0
            elif prec_val == 0:
                p = 0
            else:
                p = prec_val

            if p > 0:

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
                if x_ < 12:
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

    return lat_cols, long_cols, gh_cols
