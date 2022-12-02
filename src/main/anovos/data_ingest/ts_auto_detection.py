# coding=utf-8

"""This module help produce the output containing a transformation through auto timestamp / date detection by reading the ingested dataframe from source.

As a part of generation of the auto detection output, there are various functions created such as -

- regex_date_time_parser
- ts_loop_cols_pre
- ts_preprocess

Respective functions have sections containing the detailed definition of the parameters used for computing.

"""

import calendar
import csv
import datetime
import io
import os
import re
import subprocess
import warnings
from pathlib import Path

import dateutil.parser
import mlflow
import numpy as np
import pandas as pd
import pyspark
from loguru import logger
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from anovos.data_analyzer.stats_generator import measures_of_percentiles
from anovos.data_transformer.datetime import (
    lagged_ts,
    timeUnits_extraction,
    unix_to_timestamp,
)
from anovos.shared.utils import (
    attributeType_segregation,
    ends_with,
    output_to_local,
    path_ak8s_modify,
)

###regex based ts parser function


def regex_date_time_parser(
    spark,
    idf,
    id_col,
    col,
    tz,
    val_unique_cat,
    trans_cat,
    save_output=None,
    output_mode="replace",
):

    """

    This function helps to produce the transformed output in timestamp (if auto-detected) based on the input data.


    Parameters
    ----------

    spark
        Spark session
    idf
        Input dataframe
    id_col
        ID Column
    col
        Column passed for Auto detection of Timestamp / date type
    tz
        Timezone offset (Option to chose between options like Local, GMT, UTC). Default option is set as "Local".
    val_unique_cat
        Maximum character length of the field.
    trans_cat
        Custom data type basis which further processing will be conditioned.
    save_output
        Output path where the transformed ddata can be saved
    output_mode
        Option to choose between Append or Replace. If the option Append is selected, the column names are Appended by "_ts" else it's replaced by the original column name

    Returns
    -------
    DataFrame
    """

    REGEX_PARTS = {
        "Y": r"(?:19[4-9]\d|20[0-3]\d)",  # 1940 to 2039
        "y": r"(?:\d\d)",  # 00 to 99
        "m": r"(?:1[012]|0?[1-9])",  # 0?1 to 12
        "mz": r"(?:1[012]|0[1-9])",  # 01 to 12
        "B": r"(?:"
        r"D?JAN(?:UAR[IY])?|"
        r"[FP]EB(?:RUAR[IY])?|"
        r"MAC|MAR(?:CH|ET)?|MRT|"
        r"APR(?:IL)?|"
        r"M[EA]I|MAY|"
        r"JUNE?|D?JUNI?|"
        r"JUL(?:Y|AI)?|D?JULI?|"
        r"OG(?:OS)?|AUG(?:UST)?|AGT?(?:USTUS)?|"
        r"SEP(?:T(?:EMBER)?)?|"
        r"O[KC]T(?:OBER)?|"
        r"NO[VP](?:EMBER)?|"
        r"D[EI][SC](?:EMBER)?"
        r")",
        "d": r"(?:3[01]|[12]\d|0?[1-9])",  # 0?1 to 31
        "d_range": r"(?:3[01]|[12]\d|0?[1-9])(?: ?[-] ?(?:3[01]|[12]\d|0?[1-9]))?",  # 14-15
        "dz": r"(?:3[01]|[12]\d|0[1-9])",  # 01 to 31
        "j": r"(?:36[0-6]|3[0-5]\d|[12]\d\d|0?[1-9]\d|0?0?[1-9])",  # 0?0?1 to 366
        "H": r"(?:2[0-4]|[01]?\d)",  # 0?0 to 24
        "HZ": r"(?:2[0-4]|[01]\d)",  # 0?0 to 24
        "I": r"(?:1[012]|0?[1-9])",  # 0?1 to 12
        "M": r"(?:[1-5]\d|0\d)",  # 00 to 59
        "S": r"(?:6[01]|[0-5]\d)",  # 00 to 61 (leap second)
        "p": r"(?:MIDNI(?:GHT|TE)|AFTERNOON|MORNING|NOON|[MN]N|H(?:(?:OU)?RS?)?|[AP]\.? ?M\.?)",
        "p2": r"(?:MIDNI(?:GHT|TE)|NOON|[AP]\.? ?M\.?)",
        "Z": r"(?:A(?:C(?:DT|ST|T|WST)|DT|E(?:DT|ST)|FT|K(?:DT|ST)|M(?:ST|T)|RT|ST|WST"
        r"|Z(?:O(?:ST|T)|T))|B(?:DT|I(?:OT|T)|OT|R(?:ST|T)|ST|TT)|C(?:AT|CT|DT|E("
        r"?:ST|T)|H(?:A(?:DT|ST)|O(?:ST|T)|ST|UT)|I(?:ST|T)|KT|L(?:ST|T)|O(?:ST|T"
        r")|ST|T|VT|WST|XT)|D(?:AVT|DUT|FT)|E(?:A(?:S(?:ST|T)|T)|CT|DT|E(?:ST|T)|"
        r"G(?:ST|T)|IT|ST)|F(?:ET|JT|K(?:ST|T)|NT)|G(?:A(?:LT|MT)|ET|FT|I(?:LT|T)"
        r"|MT|ST|YT)|H(?:AEC|DT|KT|MT|OV(?:ST|T)|ST)|I(?:CT|D(?:LW|T)|OT|R(?:DT|K"
        r"T|ST)|ST)|JST|K(?:ALT|GT|OST|RAT|ST)|L(?:HST|INT)|M(?:A(?:GT|RT|WT)|DT|"
        r"E(?:ST|T)|HT|I(?:ST|T)|MT|S(?:K|T)|UT|VT|YT)|N(?:CT|DT|FT|PT|ST|T|UT|Z("
        r"?:DT|ST))|O(?:MST|RAT)|P(?:DT|ET(?:T)?|GT|H(?:OT|T)|KT|M(?:DT|ST)|ONT|S"
        r"T|Y(?:ST|T))|R(?:ET|OTT)|S(?:A(?:KT|MT|ST)|BT|CT|DT|GT|LST|R(?:ET|T)|ST"
        r"|YOT)|T(?:AHT|FT|HA|JT|KT|LT|MT|OT|RT|VT)|U(?:LA(?:ST|T)|TC|Y(?:ST|T)|Z"
        r"T)|V(?:ET|LAT|O(?:LT|ST)|UT)|W(?:A(?:KT|ST|T)|E(?:ST|T)|IT|ST)|Y(?:AKT|"
        r"EKT))",  # FROM: en.wikipedia.org/wiki/List_of_time_zone_abbreviations
        "z": r"(?:[+-](?:0\d|1[0-4]):?(?:00|15|30|45))",  # [+-] 00:00 to 14:45
        "A": r"(?:"
        r"MON(?:DAY)?|(?:IS|SE)N(?:[IE]N)?|"
        r"TUE(?:S(?:DAY)?)?|SEL(?:ASA)?|"
        r"WED(?:NESDAY)?|RABU?|"
        r"THU(?:RS(?:DAY)?)?|KH?A(?:M(?:IS)?)?|"
        r"FRI(?:DAY)?|JUM(?:[AM]A?T)?|"
        r"SAT(?:URDAY)?|SAB(?:TU)?|"
        r"SUN(?:DAY)?|AHA?D|MIN(?:GGU)?"
        r")",
        "th": r"(?:ST|ND|RD|TH|º)",
    }

    REGEX_PATTERNS_PARSERS = {
        # 14/8/1991
        "dd_mm_YYYY_1": r"(?:{d}/{m}/{Y})",
        "dd_d2": r"(?:{d}\\{m}\\{Y})",
        "dd_mm_YYYY_3": r"(?:{d}[-]{m}[-]{Y})",
        "dd_mm_YYYY_4": r"(?:{d}\.{m}\.{Y})",
        # 'dd_mm_YYYY_5':          r"(?:{d}{m}{Y})",  # too many phone numbers
        "dd_mm_YYYY_6": r"(?:{d} ?{m} ?{Y})",
        "dd_mm_YYYY_7": r"(?:{dz}{mz}{Y})",
        # 14/8/91
        "dd_mm_yy_1": r"(?:{d}/{m}/{y})",
        "dd_mm_yy_2": r"(?:{d}\\{m}\\{y})",
        "dd_mm_yy_3": r"(?:{d}[-]{m}[-]{y})",
        "dd_mm_yy_4": r"(?:{d}\.{m}\.{y})",
        # 'dd_mm_yy_5':            r"(?:{dz}{mz}{y})",  # too many phone numbers
        # 14 Aug, 1991
        "dd_mmm_YYYY_1": r"(?:{d}{th}? ?/ ?{B} ?/ ?{Y})",
        "dd_mmm_YYYY_2": r"(?:{d}{th}? ?\\ ?{B} ?\\ ?{Y})",
        "dd_mmm_YYYY_3": r"(?:{d}{th}? ?[-] ?{B} ?[ -] ?{Y})",
        "dd_mmm_YYYY_4": r"(?:{d}{th}? ?[ -]? ?{B} ?,? ?{Y})",
        "dd_mmm_YYYY_5": r"(?:{d}{th}? ?\. ?{B} ?\. ?{Y})",
        # 14 Aug '91
        "dd_mmm_yy_1": r"(?:{d}{th}? ?/ ?{B} ?/ ?'?{y})",
        "dd_mmm_yy_2": r"(?:{d}{th}? ?\\ ?{B} ?\\ ?'?{y})",
        "dd_mmm_yy_3": r"(?:{d}{th}? ?[-] ?{B} ?[-] ?'?{y})",
        "dd_mmm_yy_4": r"(?:{d}{th}? ?[ -]? ?{B} ?,? ?'?{y})",
        "dd_mmm_yy_5": r"(?:{d}{th}? ?\. ?{B} ?\. ?'?{y})",
        # 14th Aug
        "dd_mmm": r"(?:{d}{th}? ?[/\\. -] ?{B})",
        # 08/14/1991  # WARNING! dateutil set to day first
        "mm_dd_YYYY_1": r"(?:{m}/{d}/{Y})",
        "mm_dd_YYYY_2": r"(?:{m}\\{d}\\{Y})",
        "mm_dd_YYYY_3": r"(?:{m}[-]{d}[-]{Y})",
        "mm_dd_YYYY_4": r"(?:{m} {d} {Y})",
        "mm_dd_YYYY_5": r"(?:{m}\.{d}\.{Y})",
        "mm_dd_YYYY_6": r"(?:{mz}{dz}{Y})",
        # 8/14/91  # WARNING! dateutil set to day first
        "mm_dd_yy_1": r"(?:{m}/{d}/{y})",
        "mm_dd_yy_2": r"(?:{m}\\{d}\\{y})",
        "mm_dd_yy_3": r"(?:{m}[-]{d}[-]{y})",
        "mm_dd_yy_4": r"(?:{m}\.{d}\.{y})",
        # 'mm_dd_yy_5':            r"(?:{mz}{dz}{y})",  # too many phone numbers
        # Aug 14th, 1991
        "mmm_dd_YYYY_1": r"(?:{B} ?/ ?{d}{th}? ?/ ?{Y})",
        "mmm_dd_YYYY_2": r"(?:{B} ?\\ ?{d}{th}? ?\\ ?{Y})",
        "mmm_dd_YYYY_3": r"(?:{B} ?[-] ?{d}{th}? ?[ -] ?{Y})",
        "mmm_dd_YYYY_4": r"(?:{B} ?[ -]? ?{d}{th}? ?, ?{Y})",
        "mmm_dd_YYYY_5": r"(?:{B} ?\. ?{d}{th}? ?\. ?{Y})",
        # Aug-14 '91
        "mmm_dd_yy_1": r"(?:{B} ?/ ?{d}{th}? ?/ ?'?{y})",
        "mmm_dd_yy_2": r"(?:{B} ?\\ ?{d}{th}? ?\\ ?'?{y})",
        "mmm_dd_yy_3": r"(?:{B} ?[-] ?{d}{th}? ?[-] ?'?{y})",
        "mmm_dd_yy_4": r"(?:{B} ?[. -]? ?{d}{th}?, '?{y})",
        "mmm_dd_yy_5": r"(?:{B} ?\. ?{d}{th}? ?\. ?'?{y})",
        # Aug-14  # WARNING! dateutil assumes current year
        "mmm_dd": r"(?:{B} ?[/\\. -] ?{d}{th}?)",
        # # Aug-91
        # 'mmm_yy':                r"(?:{B} ?[/\\. -] ?'{y})",  # too many false positives
        # August 1991
        "mmm_YYYY": r"(?:{B} ?[/\\. -] ?{Y})",  # many non-useful dates
        # 1991-8-14
        "YYYY_mm_dd_1": r"(?:{Y}/{m}/{d})",
        "YYYY_mm_dd_2": r"(?:{Y}\\{m}\\{d})",
        "YYYY_mm_dd_3": r"(?:{Y}[-]{m}[-]{d})",
        "YYYY_mm_dd_4": r"(?:{Y} {m} {d})",
        "YYYY_mm_dd_5": r"(?:{Y}\.{m}\.{d})",
        "YYYY_mm_dd_6": r"(?:{Y}{mz}{dz})",
        # 910814 (ISO 8601)
        # 'yy_mm_dd_1':            r"(?:{y} {m} {d})",  # too many random numbers
        "yy_mm_dd_2": r"(?:{y}/{m}/{d})",
        "yy_mm_dd_3": r"(?:{y}\\{m}\\{d})",
        "yy_mm_dd_4": r"(?:{y}[-]{m}[-]{d})",
        "yy_mm_dd_5": r"(?:{y}\.{m}\.{d})",
        # 'yy_mm_dd_6':            r"(?:{y}{mz}{dz})",  # too many phone numbers
        # 1991-Aug-14
        "YYYY_mmm_dd_1": r"(?:{Y} ?/ ?{B} ?/ ?{d})",
        "YYYY_mmm_dd_2": r"(?:{Y} ?\\ ?{B} ?\\ ?{d})",
        "YYYY_mmm_dd_3": r"(?:{Y} ?[-] ?{B} ?[-] ?{d})",
        "YYYY_mmm_dd_4": r"(?:{Y} ?{B} ?[ -]? ?{d}{th}?)",
        # 91-Aug-14
        "yy_mmm_dd_1": r"(?:'?{y} ?/ ?{B} ?/ ?{d})",
        "yy_mmm_dd_2": r"(?:'?{y} ?\\ ?{B} ?\\ ?{d})",
        "yy_mmm_dd_3": r"(?:'?{y} ?[-] ?{B} ?[-] ?{d})",
        "yy_mmm_dd_4": r"(?:'?{y} ?{B} ?[ -]? ?{d}{th}?)",
        # # 1991.226 (Aug 14 = day 226 in 1991)  # dateutil fails
        # 'YYYY_ddd_1':            r"(?:{Y}\.{j})",  # too many random numbers
        # 'YYYY_ddd_2':            r"(?:{Y}[-]{j})",  # too many random numbers
        # time
        "HH_MM_SS": r"(?:{H}:{M}:{S}(?: ?{p})?(?: ?(?:Z|{Z}|{z}))?)",
        "HH_MZ_pp_1": r"(?:{H}:{M}(?: ?{p})?(?: ?(?:Z|{Z}|{z}))?)",
        "HH_MZ_pp_1b": r"(?:{H}[:. ]{M}(?: ?{p})(?: ?(?:Z|{Z}|{z}))?)",
        "HH_MZ_pp_2": r"(?:(?<!\.){HZ}[. ]?{M}(?: ?{p})(?: ?(?:Z|{Z}|{z}))?)",
        "HH_pp": r"(?:(?<!\.){H} ?{p2}(?: ?(?:Z|{Z}|{z}))?)",
        # # 910814094500 (9:45am)
        # 'yy_mm_dd_HH_MM_SS':     r"(?:{y}{mz}{dz}{H}{M}{S})",  # too many phone numbers
        # 1991-08-14T09:45:00Z
        "YYYY_mm_dd_HH_MM": r"(?:{Y}[-]{m}[-]{d}[T ]{H}:{M}(?: ?(?:Z|{Z}|{z}))?)",
        "YYYY_mm_dd_HH_MM_SS_1": r"(?:{Y}[-]{m}[-]{d}[T ]{H}:{M}:{S}(?: ?(?:Z|{Z}|{z}))?)",
        "YYYY_mm_dd_HH_MM_SS_2": r"(?:{Y}{mz}{d}T?{H}{M}{S}(?: ?(?:Z|{Z}|{z}))?)",
        "YYYY_dd_mm_HH_MM_SS_3": r"(?:{Y}[-]{d}[-]{m}[T ]{H}:{M}:{S}(?: ?(?:Z|{Z}|{z}))?)",
        "mm_dd_YYYY_HH_MM_SS_1": r"(?:{m}[-]{d}[-]{Y}[T ]{H}:{M}:{S}(?: ?(?:Z|{Z}|{z}))?)",
        "dd_mm_YYYY_HH_MM_SS_1": r"(?:{d}[-]{m}[-]{Y}[T ]{H}:{M}:{S}(?: ?(?:Z|{Z}|{z}))?)",
        # # standalone
        # 'day':                   r"{A}",  # too many false positives
        # 'month':                 r"{B}",  # too many false positives
        # 'year':                  r"{Y}",  # too many random numbers
        # 'timezone':              r"(?:Z|{Z}|{z})",  # too many malay words
    }

    #  unicode fixes
    REGEX_FORMATTED = {
        label: "\\b"
        + pattern.format(**REGEX_PARTS)  # fill in the chunks
        .replace("-]", "\u2009\u2010\u2011\u2012\u2013\u2014-]")  # unicode dashes
        .replace("'?", "['\u2018\u2019]?")  # unicode quotes
        + "\\b"
        for label, pattern in REGEX_PATTERNS_PARSERS.items()
    }

    #     match emails and urls to avoid returning chunks of them
    REGEX_FORMATTED[
        "eml"
    ] = r"""[a-zA-Z0-9][^\s`!@%$^={}\[\]/\\"',()<>:;]+(?:@|%40|\s+at\s+|\s*<\s*at\s*>\s*)[a-zA-Z0-9][-_a-zA-Z0-9~.]+\.[a-zA-Z]{2,15}"""
    REGEX_FORMATTED[
        "url"
    ] = r"\b(?:(?:https?|ftp|file)://|www\d?\.|ftp\.)[-A-Z0-9+&@#/%=~_|$?!:,.]*[A-Z0-9+&@#/%=~_|$]"
    REGEX_FORMATTED["dot"] = r"(?:\d+\.){3,}\d+"

    # compile all the regex patterns
    REGEX_COMPILED = {
        label: re.compile(pattern, flags=re.I | re.U)
        for label, pattern in REGEX_FORMATTED.items()
    }

    if trans_cat == "dt":

        return idf

    elif (trans_cat in ["long_c", "bigint_c", "int_c"]) & (
        int(val_unique_cat) in [10, 13]
    ):

        if int(val_unique_cat) == 10:
            precision = "s"
        elif int(val_unique_cat) == 13:
            precision = "ms"
        else:
            precision = "ms"

        output_df = unix_to_timestamp(
            spark, idf, col, precision=precision, tz=tz, output_mode=output_mode
        ).orderBy(id_col, col)

        if save_output is not None:
            output_df.write.parquet(save_output, mode="overwrite")

        else:
            return output_df

    elif trans_cat == "string":

        list_dates = list(set(idf.select(col).rdd.flatMap(lambda x: x).collect()))

        def regex_text(text, longest=True, context_max_len=999, dayfirst=False):
            # join multiple spaces, convert tabs, strip leading/trailing whitespace

            if isinstance(text, str):
                pass
            else:
                raise ValueError("Incompatible Column Type!!")

            text = " ".join(text.split())
            matches = []

            for regex_label, regex_obj in REGEX_COMPILED.items():
                for m in regex_obj.finditer(text):

                    context_start = max(0, (m.start() + m.end() - context_max_len) // 2)
                    context_end = min(len(text), context_start + context_max_len)

                    context_str = text[context_start:context_end]

                    if context_start != 0:
                        context_str = "\u2026" + context_str[1:]
                    if context_end != len(text):
                        context_str = (
                            context_str[:-1] + "\u2026"
                        )  # this is the `...` character

                    parsed_date = None
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter(
                                "ignore",
                                category=dateutil.parser.UnknownTimezoneWarning,
                            )

                            if "HH" in regex_label:
                                if "dd" in regex_label or "YYYY" in regex_label:
                                    matched_text = re.sub(r"[\\]", "/", m.group())
                                    parsed_date = dateutil.parser.parse(
                                        matched_text, dayfirst=dayfirst
                                    )
                                else:
                                    matched_text = re.sub(
                                        r"H(?:(?:OU)?RS?)?", "", m.group(), flags=re.I
                                    )
                                    matched_text = re.sub(
                                        r"MN", r"AM", matched_text, flags=re.I
                                    )
                                    matched_text = re.sub(
                                        r"NN", r"PM", matched_text, flags=re.I
                                    )
                                    matched_text = re.sub(
                                        r"(\d)[. ](\d)", r"\1:\2", matched_text
                                    )
                                    matched_text = f"1970-01-01 {matched_text}"
                                    parsed_date = dateutil.parser.parse(
                                        matched_text, dayfirst=dayfirst
                                    )
                            elif "dd" in regex_label or "YYYY" in regex_label:
                                matched_text = re.sub(r"[\\]", "/", m.group())
                                parsed_date = dateutil.parser.parse(
                                    matched_text, dayfirst=dayfirst
                                )
                    except ValueError:
                        pass

                    matches.append(
                        {
                            "REGEX_LABEL": regex_label,
                            "MATCH": m.group(),
                            "START": m.start(),
                            "END": m.end(),
                            "MATCH_LEN": m.end() - m.start(),
                            "NORM_TEXT_LEN": len(text),
                            "CONTEXT": context_str,
                            "PARSED": parsed_date,
                        }
                    )

            # narrow to longest match
            for match in matches:
                if not longest or all(
                    (other["START"] >= match["START"] and other["END"] <= match["END"])
                    or other["START"] > match["END"]
                    or other["END"] < match["START"]
                    for other in matches
                ):

                    # don't return emails or urls
                    if match["REGEX_LABEL"] not in {"eml", "url", "dot"}:
                        yield match

        bl = []
        file_lines = list_dates
        for line_num, line in enumerate(file_lines):
            bl_int = []
            for match_info in regex_text(line):
                try:
                    ye, mo, da, ho, mi, se = (
                        match_info["PARSED"].year,
                        match_info["PARSED"].month,
                        match_info["PARSED"].day,
                        match_info["PARSED"].hour,
                        match_info["PARSED"].minute,
                        match_info["PARSED"].second,
                    )
                    if len(bl_int) == 0:
                        bl_int = [ye, mo, da, ho, mi, se]

                    else:
                        if ye == 1970 and mo == 1 and da == 1:
                            pass
                        if ho + mi + se == 0:
                            pass
                        if ye > 1970:
                            bl_int[0] = ye
                        if mo > 0 and ye != 1970:
                            bl_int[1] = mo
                        if da > 0 and ye != 1970:
                            bl_int[2] = da
                        if ho > 0:
                            bl_int[3] = ho
                        if mi > 0:
                            bl_int[4] = mi
                        if se > 0:
                            bl_int[5] = se
                        else:
                            pass
                except:
                    pass
            bl.append(
                [
                    match_info["CONTEXT"],
                    datetime.datetime(
                        bl_int[0],
                        bl_int[1],
                        bl_int[2],
                        bl_int[3],
                        bl_int[4],
                        bl_int[5],
                    ),
                ]
            )

        if len(bl) >= 1:
            columns = [col, col + "_ts"]
            # output_df = spark.createDataFrame(spark.parallelize(bl),columns)
            output_df = spark.createDataFrame(pd.DataFrame(bl, columns=columns))
        else:
            return idf

    elif trans_cat in ["string_c", "int_c"]:

        if int(val_unique_cat) == 4:

            output_df = idf.select(col).withColumn(
                col + "_ts", F.col(col).cast("string").cast("date")
            )

        elif int(val_unique_cat) == 6:

            output_df = (
                idf.select(col)
                .withColumn(col, F.concat(col, F.lit("01")))
                .withColumn(col + "_ts", F.to_date(col, "yyyyMMdd"))
            )

        elif int(val_unique_cat) == 8:

            f = (
                idf.select(
                    F.max(F.substring(col, 1, 4)),
                    F.max(F.substring(col, 5, 2)),
                    F.max(F.substring(col, 7, 2)),
                )
                .rdd.flatMap(lambda x: x)
                .collect()
            )

            if int(f[1]) > 12:
                frmt = "yyyyddMM"
            elif int(f[2]) > 12:
                frmt = "yyyyMMdd"
            elif (
                (int(f[0]) > 1970 & int(f[0]) < 2049)
                & (int(f[1]) > 0 & int(f[1]) <= 12)
                & (int(f[2]) > 0 & int(f[2]) <= 31)
            ):
                frmt = "yyyyMMdd"
            elif (
                (int(f[0]) > 1970 & int(f[0]) < 2049)
                & (int(f[1]) > 0 & int(f[1]) <= 31)
                & (int(f[2]) > 0 & int(f[2]) <= 12)
            ):
                frmt = "yyyyddMM"
            else:
                return idf

            output_df = idf.select(F.col(col).cast("string")).withColumn(
                col + "_ts", F.to_date(col, frmt)
            )

        else:
            return idf

    else:

        return idf

    # if ((output_df.where(F.col(col + "_ts").isNull()).count()) / output_df.count()) > 0.9:

    #     return idf

    # else:
    #     pass

    if output_mode == "replace":

        output_df = (
            idf.join(output_df, col, "left_outer")
            .drop(col)
            .withColumnRenamed(col + "_ts", col)
            .orderBy(id_col, col)
        )

    elif output_mode == "append":

        output_df = idf.join(output_df, col, "left_outer").orderBy(id_col, col + "_ts")

    else:

        return "Incorrect Output Mode Selected"

    if save_output:

        output_df.write.parquet(save_output, mode="overwrite")

    else:
        return output_df


def ts_loop_cols_pre(idf, id_col):

    """

    This function helps to analyze the potential columns which can be passed for the time series check. The columns are passed on to the auto-detection block.

    Parameters
    ----------

    idf
        Input dataframe
    id_col
        ID Column

    Returns
    -------
    Three lists
    """

    lc1, lc2, lc3 = [], [], []
    for i in idf.dtypes:
        try:
            col_len = (
                idf.select(F.max(F.length(i[0]))).rdd.flatMap(lambda x: x).collect()[0]
            )
        except:
            col_len = 0
        if idf.select(i[0]).dropna().distinct().count() == 0:
            lc1.append(i[0])
            lc2.append("NA")
            lc3.append(col_len)
        elif (
            (i[0] != id_col)
            & (idf.select(F.length(i[0])).distinct().count() == 1)
            & (col_len in [4, 6, 8, 10, 13])
        ):
            if i[1] == "string":
                lc1.append(i[0])
                lc2.append("string_c")
                lc3.append(col_len)
            elif i[1] == "long":
                lc1.append(i[0])
                lc2.append("long_c")
                lc3.append(col_len)
            elif i[1] == "bigint":
                lc1.append(i[0])
                lc2.append("bigint_c")
                lc3.append(col_len)
            elif i[1] == "int":
                lc1.append(i[0])
                lc2.append("int_c")
                lc3.append(col_len)
        elif (i[0] != id_col) & (i[1] in ["string", "object"]):
            lc1.append(i[0])
            lc2.append("string")
            lc3.append(col_len)
        elif (i[0] != id_col) & (i[1] in ["timestamp", "date"]):
            lc1.append(i[0])
            lc2.append("dt")
            lc3.append(col_len)
        else:
            lc1.append(i[0])
            lc2.append("NA")
            lc3.append(col_len)

    return lc1, lc2, lc3


def ts_preprocess(
    spark,
    idf,
    id_col,
    output_path,
    tz_offset="local",
    run_type="local",
    mlflow_config=None,
    auth_key="NA",
):

    """

    This function helps to read the input spark dataframe as source and do all the necessary processing. All the intermediate data created through this step foro the Time Series Analyzer.

    Parameters
    ----------

    spark
        Spark session
    idf
        Input dataframe
    id_col
        ID Column
    output_path
        Output path where the data would be saved
    tz_offset
        Timezone offset (Option to chose between options like Local, GMT, UTC, etc.). Default option is set as "Local".
    run_type
        Option to choose between run type "local" or "emr" or "databricks" or "ak8s" basis the user flexibility. Default option is set as "local".
    mlflow_config
        MLflow configuration. If None, all MLflow features are disabled.
    auth_key
        Option to pass an authorization key to write to filesystems. Currently applicable only for "ak8s" run_type.


    Returns
    -------
    DataFrame,Output[CSV]
    """

    if run_type == "local":
        local_path = output_path
    elif run_type == "databricks":
        local_path = output_to_local(output_path)
    elif run_type in ("emr", "ak8s"):
        local_path = "report_stats"
    else:
        raise ValueError("Invalid run_type")

    Path(local_path).mkdir(parents=True, exist_ok=True)

    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    ts_loop_col_dtls = ts_loop_cols_pre(idf, id_col)

    l1, l2 = [], []
    for index, i in enumerate(ts_loop_col_dtls[1]):
        if i != "NA":
            l1.append(index)
        if i == "dt":
            l2.append(index)

    ts_loop_cols = [ts_loop_col_dtls[0][i] for i in l1]

    pre_exist_ts_cols = [ts_loop_col_dtls[0][i] for i in l2]

    for i in ts_loop_cols:
        try:
            idx = ts_loop_col_dtls[0].index(i)
            val_unique_cat = ts_loop_col_dtls[2][idx]
            trans_cat = ts_loop_col_dtls[1][idx]
            idf = regex_date_time_parser(
                spark,
                idf,
                id_col,
                i,
                tz_offset,
                val_unique_cat,
                trans_cat,
                save_output=None,
                output_mode="replace",
            )
            idf.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

        except:
            pass

    odf = idf.distinct()

    ts_loop_cols_post = [x[0] for x in idf.dtypes if x[1] in ["timestamp", "date"]]

    num_cols = [x for x in num_cols if x not in [id_col] + ts_loop_cols_post]
    cat_cols = [x for x in cat_cols if x not in [id_col] + ts_loop_cols_post]

    c1 = ts_loop_cols
    c2 = list(set(ts_loop_cols_post) - set(pre_exist_ts_cols))
    c3 = pre_exist_ts_cols
    c4 = ts_loop_cols_post

    f = pd.DataFrame(
        [
            [",".join(idf.columns)],
            [",".join(c1)],
            [",".join(c2)],
            [",".join(c3)],
            [",".join(c4)],
            [",".join(num_cols)],
            [",".join(cat_cols)],
        ],
        columns=["cols"],
    )

    f.to_csv(ends_with(local_path) + "ts_cols_stats.csv", index=False)

    if mlflow_config is not None:
        mlflow.log_artifact(ends_with(local_path) + "ts_cols_stats.csv")

    if run_type == "emr":
        bash_cmd = (
            "aws s3 cp --recursive "
            + ends_with(local_path)
            + " "
            + ends_with(output_path)
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])

    if run_type == "ak8s":
        output_path_mod = path_ak8s_modify(output_path)
        bash_cmd = (
            'azcopy cp "'
            + ends_with(local_path)
            + '" "'
            + ends_with(output_path_mod)
            + str(auth_key)
            + '" --recursive=true '
        )
        output = subprocess.check_output(["bash", "-c", bash_cmd])

    return odf
