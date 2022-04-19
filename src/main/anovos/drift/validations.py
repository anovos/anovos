from functools import wraps, partial
from loguru import logger
from anovos.shared.utils import attributeType_segregation
from inspect import getcallargs


def refactor_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_kwargs = getcallargs(func, *args, **kwargs)

        for boolarg in (
            "pre_existing_source",
            "pre_computed_raw_stats",
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

        if func.__name__ == "statistics":
            idf_target = all_kwargs.get("idf_target")
            idf_source = all_kwargs.get("idf_source")
            num_cols = attributeType_segregation(idf_target)[0]
            list_of_cols = all_kwargs.get("list_of_cols")
            drop_cols = all_kwargs.get("drop_cols", [])
            all_valid_cols = num_cols

            if list_of_cols == "all":
                list_of_cols = all_valid_cols
            if isinstance(list_of_cols, str):
                list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]
            if isinstance(drop_cols, str):
                drop_cols = [x.strip() for x in drop_cols.split("|")]

            list_of_cols = [e for e in list_of_cols if e not in drop_cols]

            if len(list_of_cols) == 0:
                raise TypeError(
                    f"Invalid input for column(s) in the function {func.__name__}."
                )
            if any(x not in all_valid_cols for x in list_of_cols):
                raise TypeError(
                    f"Invalid input for column(s) in the function {func.__name__}. Invalid Column(s): {set(list_of_cols) - set(all_valid_cols)} not found in target dataframe."
                )
            if any(x not in idf_source.columns for x in list_of_cols):
                raise TypeError(
                    f"Invalid input for column(s) in the function {func.__name__}. Invalid Column(s): {set(list_of_cols) - set(idf_source.columns)} not found in source dataframe."
                )

            method_type = all_kwargs.get("method_type")
            if isinstance(method_type, str):
                if method_type == "all":
                    method_type = ["PSI", "JSD", "HD", "KS"]
                else:
                    method_type = [x.strip() for x in method_type.split("|")]
            if any(x not in ("PSI", "JSD", "HD", "KS") for x in method_type):
                raise TypeError(f"Invalid input for method_type")

            bin_method = all_kwargs.get("bin_method")
            if bin_method not in ("equal_frequency", "equal_range"):
                raise TypeError(f"Invalid input for bin_method")

            all_kwargs["list_of_cols"] = list_of_cols
            all_kwargs["method_type"] = method_type
            all_kwargs["drop_cols"] = []

        elif func.__name__ == "stability_index_computation":
            idfs = all_kwargs.get("idfs")
            list_of_cols = all_kwargs.get("list_of_cols")
            drop_cols = all_kwargs.get("drop_cols", [])

            if isinstance(list_of_cols, str):
                list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]
            if isinstance(drop_cols, str):
                drop_cols = [x.strip() for x in drop_cols.split("|")]

            if all_kwargs.get("pre_computed_raw_stats") is False:
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

                all_kwargs["list_of_cols"] = list_of_cols
                all_kwargs["drop_cols"] = []
            else:
                if len(idfs) > 0:
                    logger.warning(
                        "When pre_computed_raw_stats is True, argument idfs will be ignored and raw_stats_path_list will be used instead."
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
        return func(**all_kwargs)

    return wrapper
