from functools import wraps, partial
import warnings

from anovos.data_transformer.transformers import attribute_binning
from anovos.shared.utils import attributeType_segregation
from inspect import getcallargs
from pyspark.sql import functions as F
import pyspark
import numpy as np
from anovos.data_ingest.data_ingest import (
    concatenate_dataset,
    join_dataset,
    read_dataset,
)


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

        elif func.__name__ == "stability_index_computation":
            idfs = all_kwargs.get("idfs")
            list_of_cols = all_kwargs.get("list_of_cols")
            drop_cols = all_kwargs.get("drop_cols")
            existing_metric_path = all_kwargs.get("existing_metric_path")

            if isinstance(list_of_cols, str):
                list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]

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
