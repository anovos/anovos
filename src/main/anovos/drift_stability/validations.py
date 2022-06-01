from functools import wraps, partial
import warnings

import pandas as pd

from anovos.data_transformer.transformers import attribute_binning
from anovos.shared.utils import attributeType_segregation
from inspect import getcallargs
from pyspark.sql import functions as F
import pyspark
import numpy as np


def refactor_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_kwargs = getcallargs(func, *args, **kwargs)

        for boolarg in (
            "pre_existing_source",
            "pre_computed_stats",
            "print_impact",
        ):
            if boolarg in all_kwargs.keys():
                boolarg_val = str(all_kwargs.get(boolarg))
                if boolarg_val.lower() == "true":
                    boolarg_val = True
                elif boolarg_val.lower() == "false":
                    boolarg_val = False
                else:
                    raise TypeError(
                        f"Non-Boolean input for {boolarg} in the function {func.__name__}."
                    )
                all_kwargs[boolarg] = boolarg_val

        if func.__name__ == "drift_statistics":
            bin_method = all_kwargs.get("bin_method")
            if bin_method not in ("equal_frequency", "equal_range"):
                raise TypeError(f"Invalid input for bin_method")

            all_kwargs["drop_cols"] = []

        elif func.__name__ == "stability_index_computation":
            idfs = all_kwargs.get("idfs")
            list_of_cols = all_kwargs.get("list_of_cols")
            drop_cols = all_kwargs.get("drop_cols")
            binary_cols = all_kwargs.get("binary_cols")
            exclude_from_binary_cols = all_kwargs.get("exclude_from_binary_cols")
            existing_metric_path = all_kwargs.get("existing_metric_path")

            if isinstance(list_of_cols, str):
                list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]
            if isinstance(drop_cols, str):
                drop_cols = [x.strip() for x in drop_cols.split("|")]
            if isinstance(binary_cols, str):
                binary_cols = [x.strip() for x in binary_cols.split("|")]
            if isinstance(exclude_from_binary_cols, str):
                exclude_from_binary_cols = [
                    x.strip() for x in exclude_from_binary_cols.split("|")
                ]
            all_kwargs["binary_cols"] = binary_cols
            all_kwargs["exclude_from_binary_cols"] = exclude_from_binary_cols

            if all_kwargs.get("pre_computed_stats") is False:
                if len(idfs) == 0:
                    if existing_metric_path == "":
                        raise TypeError(
                            f"Invalid input dataframe in the function {func.__name__}. idfs must be provided if pre_computed_stats is False and existing_metric_path is empty."
                        )
                    if list_of_cols != ["all"]:
                        list_of_cols = [e for e in list_of_cols if e not in drop_cols]
                        if len(list_of_cols) == 0:
                            raise TypeError(
                                f"Invalid input for column(s) in the function {func.__name__}."
                            )
                        all_kwargs["drop_cols"] = []
                    else:
                        all_kwargs["drop_cols"] = drop_cols

                else:
                    num_cols = attributeType_segregation(idfs[0])[0]
                    all_valid_cols = num_cols

                    if list_of_cols == ["all"]:
                        list_of_cols = all_valid_cols

                    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

                    if len(list_of_cols) == 0:
                        raise TypeError(
                            f"Invalid input for column(s) in the function {func.__name__}."
                        )

                    for idf in idfs:
                        if any(x not in idf.columns for x in list_of_cols):
                            raise TypeError(
                                f"Invalid input for column(s) in the function {func.__name__}. One or more columns are not present in all input dataframes."
                            )
                    all_kwargs["drop_cols"] = []
                all_kwargs["list_of_cols"] = list_of_cols

            else:
                stats = all_kwargs.get("stats")
                if isinstance(stats, dict):
                    if any(
                        [
                            x not in list(stats.keys())
                            for x in ["mean", "stddev", "kurtosis"]
                        ]
                    ):
                        raise TypeError(
                            f"Invalid input for stats in the function {func.__name__}. Keys of stats must contain 'mean', 'stddev', 'kurtosis'."
                        )
                value_len = -1
                for value in list(stats.values()):
                    if value_len > 0:
                        if len(value) != value_len:
                            raise TypeError(
                                f"Invalid input for stats in the function {func.__name__}. Length of all values of stats should be the same."
                            )
                    else:
                        value_len = len(value)

                    for config in value:
                        if any(
                            [
                                x not in list(config.keys())
                                for x in ["file_path", "file_type"]
                            ]
                        ):
                            raise TypeError(
                                f"Invalid input for values of stats in the function {func.__name__}. The config for each pre-computed statistics result \
                                              should be a dictionary with two compulsory keys 'file_path' and 'file_type' and one optional key 'file_configs', \
                                              which will be used as the arguments for read_dataset (data_ingest module) function."
                            )
                if len(idfs) > 0:
                    warnings.warn(
                        "When pre_computed_stats is True, idfs will be ignored and stats will be used instead."
                    )
                all_kwargs["list_of_cols"] = list_of_cols
                all_kwargs["drop_cols"] = drop_cols

            metric_weightages = all_kwargs.get("metric_weightages")
            if (
                round(
                    metric_weightages.get("mean", 0)
                    + metric_weightages.get("stddev", 0)
                    + metric_weightages.get("kurtosis", 0),
                    3,
                )
                != 1
            ):
                raise TypeError(
                    "Invalid input for metric weightages. Either metric name is incorrect or sum of metric weightages is not 1.0."
                )
            threshold = all_kwargs.get("threshold")
            if (threshold < 0) or (threshold > 4):
                raise TypeError(
                    "Invalid input for metric threshold. It must be a number between 0 and 4."
                )
        elif func.__name__ == "feature_stability_estimation":
            metric_weightages = all_kwargs.get("metric_weightages")
            if (
                round(
                    metric_weightages.get("mean", 0)
                    + metric_weightages.get("stddev", 0)
                    + metric_weightages.get("kurtosis", 0),
                    3,
                )
                != 1
            ):
                raise TypeError(
                    "Invalid input for metric weightages. Either metric name is incorrect or sum of metric weightages is not 1.0."
                )

        if "run_type" in all_kwargs.keys():
            if all_kwargs.get("run_type") not in ("local", "emr", "databricks"):
                raise TypeError(
                    f"Invalid input for run_type in the function {func.__name__}. run_type should be local, emr or databricks - Received '{all_kwargs.get('run_type')}'."
                )
        if func.__name__ == "stability_index_computation":
            return func(all_kwargs.pop("spark"), *all_kwargs.pop("idfs"), **all_kwargs)
        else:
            return func(**all_kwargs)

    return wrapper


def generate_source(
    spark, idf_source, list_of_cols, bin_method, bin_size, source_path, model_directory
):
    source_bin = attribute_binning(
        spark,
        idf_source,
        list_of_cols=list_of_cols,
        method_type=bin_method,
        bin_size=bin_size,
        pre_existing_model=False,
        model_path=source_path + "/" + model_directory,
    )
    source_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    for column in list_of_cols:
        x = (
            source_bin.groupBy(column)
            .agg((F.count(column) / idf_source.count()).alias("p"))
            .fillna(-1)
        )
        x.coalesce(1).write.csv(
            source_path + "/" + model_directory + "/frequency_counts/" + column,
            header=True,
            mode="overwrite",
        )


def generate_bin_frequencies(
    spark, source_path, model_directory, target_bin, idf_target, column
):
    try:
        x = spark.read.csv(
            source_path + "/" + model_directory + "/frequency_counts/" + column,
            header=True,
            inferSchema=True,
        )
    except OSError as err:
        print("OS error: {0}".format(err))

    y = (
        target_bin.groupBy(column)
        .agg((F.count(column) / idf_target.count()).alias("q"))
        .fillna(-1)
    )

    xy = (
        x.join(y, column, "full_outer")
        .fillna(0.0001, subset=["p", "q"])
        .replace(0, 0.0001)
        .orderBy(column)
    )
    p = np.array(xy.select("p").rdd.flatMap(lambda x: x).collect())
    q = np.array(xy.select("q").rdd.flatMap(lambda x: x).collect())

    return p, q
