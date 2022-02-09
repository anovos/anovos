import calendar
import warnings

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from datetime import datetime as dt


def argument_checker(func_name, args):
    list_of_cols = args["list_of_cols"]
    all_columns = args["all_columns"]

    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if any(x not in all_columns for x in list_of_cols):
        raise TypeError("Invalid input for Column(s)")
    if len(list_of_cols) == 0:
        warnings.warn("No timestamp conversion - No column(s) to convert")
        return []
    if func_name not in ["aggregator"]:
        if args["output_mode"] not in ("replace", "append"):
            raise TypeError("Invalid input for output_mode")

    if func_name in ["timestamp_to_unix", "unix_to_timestamp"]:
        if args["precision"] not in ("ms", "s"):
            raise TypeError("Invalid input for precision")
        if args["tz"] not in ("local", "gmt", "utc"):
            raise TypeError("Invalid input for timezone")

    if func_name in ["string_to_timestamp"]:
        if args["output_type"] not in ("ts", "dt"):
            raise TypeError("Invalid input for output_type")

    if func_name in ["timeUnits_extraction"]:
        if any(x not in args["all_units"] for x in args["units"]):
            raise TypeError("Invalid input for Unit(s)")

    if func_name in ["adding_timeUnits"]:
        if args["unit"] not in (
            args["all_units"] + [(e + "s") for e in args["all_units"]]
        ):
            raise TypeError("Invalid input for Unit")

    if func_name in ["timestamp_comparison"]:
        if args["comparison_type"] not in args["all_types"]:
            raise TypeError("Invalid input for comparison_type")

    if func_name in ["is_selectedHour"]:
        hours = list(range(0, 24))
        if args["start_hour"] not in hours:
            raise TypeError("Invalid input for start_hour")
        if args["end_hour"] not in hours:
            raise TypeError("Invalid input for end_hour")

    if func_name in ["window_aggregator"]:
        if any(x not in args["all_aggs"] for x in args["list_of_aggs"]):
            raise TypeError("Invalid input for Aggregate Function(s)")
        if args["window_type"] not in ("expanding", "rolling"):
            raise TypeError("Invalid input for Window Type")
        if (args["window_type"] == "rolling") & (
            not str(args["window_size"]).isnumeric()
        ):
            raise TypeError("Invalid input for Window Size")

    if func_name in ["aggregator"]:
        if any(x not in args["all_aggs"] for x in args["list_of_aggs"]):
            raise TypeError("Invalid input for Aggregate Function(s)")
        if args["time_col"] not in all_columns:
            raise TypeError("Invalid input for time_col")

    if func_name in ["lagged_ts"]:
        if not str(args["lag"]).isnumeric():
            raise TypeError("Invalid input for Lag")
        if args["output_type"] not in ("ts", "ts_diff"):
            raise TypeError("Invalid input for output_type")

    return list_of_cols


def timestamp_to_unix(
    spark, idf, list_of_cols, precision="s", tz="local", output_mode="replace"
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param precision: "ms", "s".
                      "ms" option returns the number of milliseconds from the unix epoch (1970-01-01 00:00:00 UTC) .
                      "s" option returns the number of seconds from the unix epoch.
    :param tz: "local", "gmt", "utc".
               Timezone of the input column(s)
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column. “append” option appends derived
                        column to the input dataset with a postfix "_unix" e.g. column X is appended as X_unix.
    :return: Output Dataframe with derived column
    """
    tz = tz.lower()
    list_of_cols = argument_checker(
        "timestamp_to_unix",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
            "precision": precision,
            "tz": tz,
        },
    )
    if not list_of_cols:
        return idf

    localtz = (
        spark.sql("SET spark.sql.session.timeZone")
        .select("value")
        .rdd.flatMap(lambda x: x)
        .collect()[0]
    )

    factor = {"ms": 1000, "s": 1}

    odf = idf
    for i in list_of_cols:
        if (tz in ("gmt", "utc")) & (localtz.lower() not in ("gmt", "utc")):
            odf = odf.withColumn(i + "_local", F.from_utc_timestamp(i, localtz))
        else:
            odf = odf.withColumn(i + "_local", F.col(i))

        modify_col = {"replace": i, "append": i + "_unix"}
        odf = odf.withColumn(
            modify_col[output_mode],
            (F.col(i + "_local").cast("double") * factor[precision]).cast("long"),
        ).drop(i + "_local")
    return odf


def unix_to_timestamp(
    spark, idf, list_of_cols, precision="s", tz="local", output_mode="replace"
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param precision: "ms", "s".
                      "ms" treats the input columns as the number of milliseconds from the unix epoch (1970-01-01 00:00:00 UTC) .
                      "s" treats the input columns as the number of seconds from the unix epoch.
    :param tz: "local", "gmt", "utc".
               timezone of the output column(s)
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column. “append” option appends derived
                        column to the input dataset with a postfix "_ts" e.g. column X is appended as X_ts.
    :return: Output Dataframe with derived column
    """
    tz = tz.lower()
    list_of_cols = argument_checker(
        "unix_to_timestamp",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
            "precision": precision,
            "tz": tz,
        },
    )
    if not list_of_cols:
        return idf

    localtz = (
        spark.sql("SET spark.sql.session.timeZone")
        .select("value")
        .rdd.flatMap(lambda x: x)
        .collect()[0]
    )

    factor = {"ms": 1000, "s": 1}

    odf = idf
    for i in list_of_cols:
        modify_col = {"replace": i, "append": i + "_ts"}
        odf = odf.withColumn(
            modify_col[output_mode], F.to_timestamp(F.col(i) / factor[precision])
        )
        if (tz in ("gmt", "utc")) & (localtz.lower() not in ("gmt", "utc")):
            odf = odf.withColumn(
                modify_col[output_mode],
                F.to_utc_timestamp(modify_col[output_mode], localtz),
            )

    return odf


def timezone_conversion(
    spark, idf, list_of_cols, given_tz, output_tz, output_mode="replace"
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param given_tz: Timezone of the input column(s). If "local", the timezone of the spark session will be used.
    :param output_tz: Timezone of the output column(s). If "local", the timezone of the spark session will be used.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column. “append” option appends derived
                        column to the input dataset with a postfix "_tzconverted" e.g. column X is appended as X_tzconverted.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "timezone_conversion",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    localtz = (
        spark.sql("SET spark.sql.session.timeZone")
        .select("value")
        .rdd.flatMap(lambda x: x)
        .collect()[0]
    )
    if given_tz == "local":
        given_tz = localtz
    if output_tz == "local":
        output_tz = localtz

    odf = idf
    for i in list_of_cols:
        modify_col = {"replace": i, "append": i + "_tzconverted"}
        odf = odf.withColumn(
            modify_col[output_mode],
            F.from_utc_timestamp(F.to_utc_timestamp(i, given_tz), output_tz),
        )

    return odf


def string_to_timestamp(
    idf,
    list_of_cols,
    input_format="%Y-%m-%d %H:%M:%S",
    output_type="ts",
    output_mode="replace",
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param date_format: Format of the input column(s) in string
    :param output_type: "ts", "dt"
                        "ts" option returns result in T.TimestampType()
                        "dt" option returns result in T.DateType()
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column. “append” option appends derived
                        column to the input dataset with a postfix "_ts" e.g. column X is appended as X_ts.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "string_to_timestamp",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
            "output_type": output_type,
        },
    )
    if not list_of_cols:
        return idf

    def conversion(col, form):
        output = dt.strptime(str(col), form)
        return output

    data_type = {"ts": T.TimestampType(), "dt": T.DateType()}
    f_conversion = F.udf(conversion, data_type[output_type])

    odf = idf
    for i in list_of_cols:
        modify_col = {"replace": i, "append": i + "_ts"}
        odf = odf.withColumn(
            modify_col[output_mode], f_conversion(F.col(i), F.lit(input_format))
        )

    return odf


def timestamp_to_string(
    idf, list_of_cols, output_format="%Y-%m-%d %H:%M:%S", output_mode="replace"
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         Columns must be of Datetime type or String type in "%Y-%m-%d %H:%M:%S" format.
    :param date_format: Format of the output column(s)
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column. “append” option appends derived
                        column to the input dataset with a postfix "_str" e.g. column X is appended as X_str.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "timestamp_to_string",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    def conversion(col, form):
        output = col.strftime(form)
        return output

    f_conversion = F.udf(conversion, T.StringType())

    odf = idf
    for i in list_of_cols:
        modify_col = {"replace": i, "append": i + "_str"}
        odf = odf.withColumn(
            modify_col[output_mode], f_conversion(F.col(i), F.lit(output_format))
        )

    return odf


def dateformat_conversion(
    idf,
    list_of_cols,
    input_format="%Y-%m-%d %H:%M:%S",
    output_format="%Y-%m-%d %H:%M:%S",
    output_mode="replace",
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param input_format: Format of the input column(s) in string
    :param output_format: Format of the output column(s) in string
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column. “append” option appends derived
                        column to the input dataset with a postfix "_ts" e.g. column X is appended as X_ts.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "dateformat_conversion",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf_tmp = string_to_timestamp(
        idf,
        list_of_cols,
        input_format=input_format,
        output_type="ts",
        output_mode=output_mode,
    )
    appended_cols = {
        "append": [col + "_ts" for col in list_of_cols],
        "replace": list_of_cols,
    }
    odf = timestamp_to_string(
        odf_tmp,
        appended_cols[output_mode],
        output_format=output_format,
        output_mode="replace",
    )

    return odf


def timeUnits_extraction(idf, list_of_cols, units, output_mode="append"):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param units: List of unit(s) to extract. Alternatively, unit(s) can be specified in a string format,
                  where different units are separated by pipe delimiter “|” e.g., "hour|minute".
                  Supported units to extract: 'hour', 'minute', 'second', 'dayofmonth', 'dayofweek',
                                              'dayofyear', 'weekofyear', 'month', 'quarter', 'year'.
                  "all" can be passed to compute all supported metrics.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived columns with a postfix "_<unit>",
                        e.g. column X is replaced with X_second for units="second".
                        “append” option appends derived column to the input dataset with a postfix "_<unit>",
                        e.g. column X is appended as X_second for units="second".
    :return: Output Dataframe with derived column
    """
    all_units = [
        "hour",
        "minute",
        "second",
        "dayofmonth",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "month",
        "quarter",
        "year",
    ]
    if units == "all":
        units = all_units
    if isinstance(units, str):
        units = [x.strip() for x in units.split("|")]

    list_of_cols = argument_checker(
        "timeUnits_extraction",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
            "units": units,
            "all_units": all_units,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        for e in units:
            func = getattr(F, e)
            odf = odf.withColumn(i + "_" + e, func(i))

        if output_mode == "replace":
            odf = odf.drop(i)

    return odf


def time_diff(idf, ts1, ts2, unit, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param ts1, ts2: The two columns to calculate the difference between.
    :param unit: 'second', 'minute', 'hour', 'day', 'week', 'month', 'year'.
                 Unit of the output values.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column <ts1>_<ts2>_<unit>diff,
                        e.g. Given ts1=X, ts2=Y , X and Y are replaced with X_Y_daydiff for unit="day".
                        “append” option appends derived column to the input dataset with name = <ts1>_<ts2>_<unit>diff,
                        e.g. Given ts1=X, ts2=Y, X_Y_daydiff is appended for unit="day".
    :return: Output Dataframe with derived column
    """
    argument_checker(
        "time_diff",
        {
            "list_of_cols": [ts1, ts2],
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )

    factor_mapping = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
        "week": 604800,
        "month": 2628000,
        "year": 31536000,
    }
    if unit in factor_mapping.keys():
        factor = factor_mapping[unit]
    elif unit in [(e + "s") for e in factor_mapping.keys()]:
        unit = unit[:-1]
        factor = factor_mapping[unit]
    else:
        raise TypeError("Invalid input of unit")

    odf = idf.withColumn(
        ts1 + "_" + ts2 + "_" + unit + "diff",
        F.abs((F.col(ts1).cast("double") - F.col(ts2).cast("double"))) / factor,
    )

    if output_mode == "replace":
        odf = odf.drop(ts1, ts2)

    return odf


def time_elapsed(idf, list_of_cols, unit, output_mode="append"):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param unit: 'second', 'minute', 'hour', 'day', 'week', 'month', 'year'.
                 Unit of the output values.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived columns with a postfix "_<unit>diff",
                        e.g. column X is replaced with X_daydiff for unit="day".
                        “append” option appends derived column to the input dataset with a postfix "_<unit>diff",
                        e.g. column X is appended as X_daydiff for unit="day".
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "time_elapsed",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    factor_mapping = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
        "week": 604800,
        "month": 2628000,
        "year": 31536000,
    }
    if unit in factor_mapping.keys():
        factor = factor_mapping[unit]
    elif unit in [(e + "s") for e in factor_mapping.keys()]:
        unit = unit[:-1]
        factor = factor_mapping[unit]
    else:
        raise TypeError("Invalid input of unit")

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_" + unit + "diff",
            F.abs(
                (F.lit(F.current_timestamp()).cast("double") - F.col(i).cast("double"))
            )
            / factor,
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def adding_timeUnits(idf, list_of_cols, unit, unit_value, output_mode="append"):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param unit: 'hour','minute','second','day','week','month','year'.
                 Unit of the added value.
    :param unit_value: The value to be added to input column(s).
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived columns with a postfix "_adjusted",
                        e.g. column X is replaced with X_adjusted.
                        “append” option appends derived column to the input dataset with a postfix "_adjusted",
                        e.g. column X is appended as X_adjusted.
    :return: Output Dataframe with derived column
    """
    all_units = ["hour", "minute", "second", "day", "week", "month", "year"]
    list_of_cols = argument_checker(
        "adding_timeUnits",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
            "unit": unit,
            "all_units": all_units,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_adjusted",
            F.col(i) + F.expr("Interval " + str(unit_value) + " " + unit),
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def timestamp_comparison(
    idf,
    list_of_cols,
    comparison_type,
    comparison_value,
    comparison_format="%Y-%m-%d %H:%M:%S",
    output_mode="append",
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param comparison_type: "greater_than", "less_than", "greaterThan_equalTo", "lessThan_equalTo"
                            The comparison type of the transformation.
    :param comparison_value: The timestamp / date value to compare with in string.
    :param comparison_format: The format of comparison_value in string.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived columns with a postfix "_compared",
                        e.g. column X is replaced with X_compared.
                        “append” option appends derived column to the input dataset with a postfix "_compared",
                        e.g. column X is appended as X_compared.
    :return: Output Dataframe with derived column
    """
    all_types = ["greater_than", "less_than", "greaterThan_equalTo", "lessThan_equalTo"]
    list_of_cols = argument_checker(
        "timestamp_comparison",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
            "comparison_type": comparison_type,
            "all_types": all_types,
        },
    )
    if not list_of_cols:
        return idf

    base_ts = dt.strptime(comparison_value, comparison_format)

    odf = idf
    for i in list_of_cols:
        if comparison_type == "greater_than":
            odf = odf.withColumn(
                i + "_compared", F.when(F.col(i) > F.lit(base_ts), 1).otherwise(0)
            )
        elif comparison_type == "less_than":
            odf = odf.withColumn(
                i + "_compared", F.when(F.col(i) < F.lit(base_ts), 1).otherwise(0)
            )
        elif comparison_type == "greaterThan_equalTo":
            odf = odf.withColumn(
                i + "_compared", F.when(F.col(i) >= F.lit(base_ts), 1).otherwise(0)
            )
        else:
            odf = odf.withColumn(
                i + "_compared", F.when(F.col(i) <= F.lit(base_ts), 1).otherwise(0)
            )

        if output_mode == "replace":
            odf = odf.drop(i)

    return odf


def start_of_month(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_monthStart".
                        “append” option appends derived column to the input dataset with a postfix "_monthStart",
                        e.g. column X is appended as X_monthStart.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "start_of_month",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(i + "_monthStart", F.trunc(i, "month"))

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_monthStart(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_ismonthStart".
                        “append” option appends derived column to the input dataset with a postfix "_ismonthStart",
                        e.g. column X is appended as X_ismonthStart.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_monthStart",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = start_of_month(idf, list_of_cols, output_mode="append")

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_ismonthStart",
            F.when(F.to_date(F.col(i)) == F.col(i + "_monthStart"), 1).otherwise(0),
        ).drop(i + "_monthStart")

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def end_of_month(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_monthEnd".
                        “append” option appends derived column to the input dataset with a postfix "_monthEnd",
                        e.g. column X is appended as X_monthEnd.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "end_of_month",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(i + "_monthEnd", F.last_day(i))

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_monthEnd(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_ismonthEnd".
                        “append” option appends derived column to the input dataset with a postfix "_ismonthEnd",
                         e.g. column X is appended as X_ismonthEnd.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_monthEnd",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = end_of_month(idf, list_of_cols, output_mode="append")

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_ismonthEnd",
            F.when(F.to_date(F.col(i)) == F.col(i + "_monthEnd"), 1).otherwise(0),
        ).drop(i + "_monthEnd")

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def start_of_year(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_yearStart".
                        “append” option appends derived column to the input dataset with a postfix "_yearStart",
                        e.g. column X is appended as X_yearStart.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "start_of_year",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(i + "_yearStart", F.trunc(i, "year"))

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_yearStart(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isyearStart".
                        “append” option appends derived column to the input dataset with a postfix "_isyearStart",
                        e.g. column X is appended as X_isyearStart.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_yearStart",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = start_of_year(idf, list_of_cols, output_mode="append")

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isyearStart",
            F.when(F.to_date(F.col(i)) == F.col(i + "_yearStart"), 1).otherwise(0),
        ).drop(i + "_yearStart")

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def end_of_year(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_yearEnd".
                        “append” option appends derived column to the input dataset with a postfix "_yearEnd",
                        e.g. column X is appended as X_yearEnd.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "end_of_year",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_yearEnd",
            F.concat_ws("-", F.year(i), F.lit(12), F.lit(31)).cast("date"),
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_yearEnd(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isyearEnd".
                        “append” option appends derived column to the input dataset with a postfix "_isyearEnd",
                        e.g. column X is appended as X_isyearEnd.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_yearEnd",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = end_of_year(idf, list_of_cols, output_mode="append")

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isyearEnd",
            F.when(F.to_date(F.col(i)) == F.col(i + "_yearEnd"), 1).otherwise(0),
        ).drop(i + "_yearEnd")

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def start_of_quarter(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_quarterStart.
                        “append” option appends derived column to the input dataset with a postfix "_quarterStart",
                        e.g. column X is appended as X_quarterStart.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "start_of_quarter",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(i + "_quarterStart", F.to_date(F.date_trunc("quarter", i)))

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_quarterStart(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isquarterStart".
                        “append” option appends derived column to the input dataset with a postfix "_isquarterStart",
                        e.g. column X is appended as X_isquarterStart.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_quarterStart",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = start_of_quarter(idf, list_of_cols, output_mode="append")

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isquarterStart",
            F.when(F.to_date(F.col(i)) == F.col(i + "_quarterStart"), 1).otherwise(0),
        ).drop(i + "_quarterStart")

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def end_of_quarter(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_quarterEnd".
                        “append” option appends derived column to the input dataset with a postfix "_quarterEnd",
                        e.g. column X is appended as X_quarterEnd.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "end_of_quarter",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_quarterEnd",
            F.to_date(F.date_trunc("quarter", i))
            + F.expr("Interval 3 months")
            + F.expr("Interval -1 day"),
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_quarterEnd(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isquarterEnd".
                        “append” option appends derived column to the input dataset with a postfix "_isquarterEnd",
                        e.g. column X is appended as X_isquarterEnd.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_quarterEnd",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = end_of_quarter(idf, list_of_cols, output_mode="append")

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isquarterEnd",
            F.when(F.to_date(F.col(i)) == F.col(i + "_quarterEnd"), 1).otherwise(0),
        ).drop(i + "_quarterEnd")

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_yearFirstHalf(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isFirstHalf".
                        “append” option appends derived column to the input dataset with a postfix "_isFirstHalf",
                        e.g. column X is appended as X_isFirstHalf.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_yearFirstHalf",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isFirstHalf",
            F.when(F.month(F.col(i)).isin(*range(1, 7)), 1).otherwise(0),
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_selectedHour(idf, list_of_cols, start_hour, end_hour, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isselectedHour".
                        “append” option appends derived column to the input dataset with a postfix "_isselectedHour",
                        e.g. column X is appended as X_isselectedHour.
    :param start_hour: the starting hour of the hour range (inclusive)
    :param end_hour: : the ending hour of the hour range (inclusive)
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_selectedHour",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    if start_hour < end_hour:
        list_of_hrs = range(start_hour, end_hour + 1)
    elif start_hour > end_hour:
        list_of_hrs = list(range(start_hour, 24)) + list(range(0, end_hour + 1))
    else:
        list_of_hrs = [start_hour]

    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isselectedHour",
            F.when(F.hour(F.col(i)).isin(*list_of_hrs), 1).otherwise(0),
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_leapYear(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isleapYear".
                        “append” option appends derived column to the input dataset with a postfix "_isleapYear",
                        e.g. column X is appended as X_isleapYear.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_leapYear",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    def check(year):
        if calendar.isleap(year):
            return 1
        else:
            return 0

    f_check = F.udf(check, T.IntegerType())

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(i + "_isleapYear", f_check(F.year(i)))

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def is_weekend(idf, list_of_cols, output_mode="append"):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column with a postfix "_isweekend".
                        “append” option appends derived column to the input dataset with a postfix "_isweekend",
                        e.g. column X is appended as X_isweekend.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "is_weekend",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        odf = odf.withColumn(
            i + "_isweekend", F.when(F.dayofweek(F.col(i)).isin([1, 7]), 1).otherwise(0)
        )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def aggregator(
    idf, list_of_cols, list_of_aggs, time_col, granularity_format="%Y-%m-%d"
):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to aggregate e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param list_of_aggs: List of aggregate metrics to compute e.g., ["f1","f2"].
                         Alternatively, metrics can be specified in a string format,
                         where different metrics are separated by pipe delimiter “|” e.g., "f1|f2".
                         Supported metrics: 'count', 'min', 'max', 'sum', 'mean', 'median', 'stddev',
                         'countDistinct', 'sumDistinct', 'collect_list', 'collect_set'.
    :param time_col: (Timestamp) Column to group by.
    :param granularity_format: Format to be allied to time_col before groupBy. The default value is
                               '%Y-%m-%d', which means grouping by the date component of time_col.
                               Alternatively, '' can be used if no formatting is necessary.

    :return: Dataframe with time_col + aggregated columns
    """
    all_aggs = [
        "count",
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "stddev",
        "countDistinct",
        "sumDistinct",
        "collect_list",
        "collect_set",
    ]
    if isinstance(list_of_aggs, str):
        list_of_aggs = [x.strip() for x in list_of_aggs.split("|")]
    list_of_cols = argument_checker(
        "aggregator",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "list_of_aggs": list_of_aggs,
            "all_aggs": all_aggs,
            "time_col": time_col,
        },
    )
    if not list_of_cols:
        return idf

    if granularity_format != "":
        idf = timestamp_to_string(
            idf, time_col, output_format=granularity_format, output_mode="replace"
        )

    def agg_funcs(col, agg):
        mapping = {
            "count": F.count(col).alias(col + "_count"),
            "min": F.min(col).alias(col + "_min"),
            "max": F.max(col).alias(col + "_max"),
            "sum": F.sum(col).alias(col + "_sum"),
            "mean": F.mean(col).alias(col + "_mean"),
            "median": F.expr("percentile_approx(" + col + ", 0.5)").alias(
                col + "_median"
            ),
            "stddev": F.stddev(col).alias(col + "_stddev"),
            "countDistinct": F.countDistinct(col).alias(col + "_countDistinct"),
            "sumDistinct": F.sumDistinct(col).alias(col + "_sumDistinct"),
            "collect_list": F.collect_list(col).alias(col + "_collect_list"),
            "collect_set": F.collect_set(col).alias(col + "_collect_set"),
        }
        return mapping[agg]

    derived_cols = []
    for i in list_of_cols:
        for j in list_of_aggs:
            derived_cols.append(agg_funcs(i, j))
    odf = idf.groupBy(time_col).agg(*derived_cols)

    return odf


def window_aggregator(
    idf,
    list_of_cols,
    list_of_aggs,
    order_col,
    window_type="expanding",
    window_size="unbounded",
    partition_col="",
    output_mode="append",
):
    """
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to aggregate e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param list_of_aggs: List of aggregate metrics to compute e.g., ["f1","f2"].
                         Alternatively, metrics can be specified in a string format,
                         where different metrics are separated by pipe delimiter “|” e.g., "f1|f2".
                         Supported metrics: 'count','min','max','sum','mean','median'
    :param order_col: (Timestamp) Column to order window
    :param window_type: "expanding", "rolling"
                        "expanding" option have a fixed lower bound (first row in the partition)
                        "rolling" option have a fixed window size defined by window_size param
    :param window_size: window size for rolling window type. Integer value with value >= 1.
    :param partition_col: Rows partitioned by this column before creating window.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column(s) with metric name as postfix.
                        “append” option appends derived column(s) to the input dataset with metric name as postfix,
                        e.g. "_count", "_mean".
    :return: Output Dataframe with derived column(s)
    """

    if isinstance(list_of_aggs, str):
        list_of_aggs = [x.strip() for x in list_of_aggs.split("|")]
    all_aggs = ["count", "min", "max", "sum", "mean", "median"]
    list_of_cols = argument_checker(
        "window_aggregator",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "list_of_aggs": list_of_aggs,
            "all_aggs": all_aggs,
            "output_mode": output_mode,
            "window_type": window_type,
            "window_size": window_size,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    window_upper = (
        Window.unboundedPreceding if window_type == "expanding" else -int(window_size)
    )
    if partition_col:
        window = (
            Window.partitionBy(partition_col)
            .orderBy(order_col)
            .rowsBetween(window_upper, 0)
        )
    else:
        window = Window.partitionBy().orderBy(order_col).rowsBetween(window_upper, 0)

    def agg_funcs(col):
        mapping = {
            "count": F.count(col).over(window).alias(col + "_count"),
            "min": F.min(col).over(window).alias(col + "_min"),
            "max": F.max(col).over(window).alias(col + "_max"),
            "sum": F.sum(col).over(window).alias(col + "_sum"),
            "mean": F.mean(col).over(window).alias(col + "_mean"),
            "median": F.expr("percentile_approx(" + col + ", 0.5)")
            .over(window)
            .alias(col + "_median"),
        }
        derived_cols = []
        for agg in list_of_aggs:
            derived_cols.append(mapping[agg])
        return derived_cols

    for i in list_of_cols:
        derived_cols = agg_funcs(i)
        odf = odf.select(odf.columns + derived_cols)

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf


def lagged_ts(
    idf,
    list_of_cols,
    lag,
    output_type="ts",
    tsdiff_unit="days",
    partition_col="",
    output_mode="append",
):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to transform e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param lag: Integer - number of row(s) to extend.
    :param output_type: "ts", "ts_diff".
                        "ts" option generats a lag column for each input column having the value that is
                        <lag> rows before the current row, and None if there is less than <lag> rows
                        before the current row.
                        "ts_diff" option generates the lag column in the same way as the "ts" option.
                        On top of that, it appends a column which represents the time_diff between the
                        original and the lag column.
    :param tsdiff_unit: 'second', 'minute', 'hour', 'day', 'week', 'month', 'year'.
                        Unit of the time_diff if output_type="ts_diff".
    :param partition_col: Rows partitioned by this column before creating window.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with derived column: <col>_lag<lag> for "ts" output_type,
                        <col>_lag<lag> and  <col>_<col>_lag<lag>_<tsdiff_unit>diff for "ts_diff" output_type.
                        “append” option appends derived column to the input dataset, e.g. given output_type="ts_diff",
                        lag=5, tsdiff_unit="days", column X is appended as X_lag5 and X_X_lag5_daydiff.
    :return: Output Dataframe with derived column
    """
    list_of_cols = argument_checker(
        "lagged_ts",
        {
            "list_of_cols": list_of_cols,
            "all_columns": idf.columns,
            "lag": lag,
            "output_type": output_type,
            "output_mode": output_mode,
        },
    )
    if not list_of_cols:
        return idf

    odf = idf
    for i in list_of_cols:
        if partition_col:
            window = Window.partitionBy(partition_col).orderBy(i)
        else:
            window = Window.partitionBy().orderBy(i)
        lag = int(lag)
        odf = odf.withColumn(i + "_lag" + str(lag), F.lag(F.col(i), lag).over(window))

        if output_type == "ts_diff":
            odf = time_diff(
                odf, i, i + "_lag" + str(lag), unit=tsdiff_unit, output_mode="append"
            )

        if output_mode == "replace":
            odf = odf.drop(i)
    return odf
