from functools import wraps
from inspect import getcallargs
from pyspark.sql import functions as F
from pyspark.sql import types as T
from anovos.shared.utils import attributeType_segregation, discrete_attributes


def refactor_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_kwargs = getcallargs(func, *args, **kwargs)

        if "idf" in all_kwargs.keys():
            idf = all_kwargs.get("idf")

            if "list_of_cols" in all_kwargs.keys():
                list_of_cols = all_kwargs.get("list_of_cols")
                drop_cols = all_kwargs.get("drop_cols", [])

                all_valid_cols = idf.columns

                if list_of_cols == "all":
                    list_of_cols = all_valid_cols
                if isinstance(list_of_cols, str):
                    list_of_cols = [
                        x.strip() for x in list_of_cols.split("|") if x.strip()
                    ]
                if isinstance(drop_cols, str):
                    drop_cols = [x.strip() for x in drop_cols.split("|")]
                list_of_cols = [e for e in list_of_cols if e not in drop_cols]
                if any(x not in all_valid_cols for x in list_of_cols):
                    raise TypeError(
                        f"Invalid input for column(s) in the function {func.__name__}. Invalid Column(s): {set(list_of_cols) - set(all_valid_cols)}."
                    )
                all_kwargs["list_of_cols"] = list_of_cols
                all_kwargs["drop_cols"] = []

            if "id_col" in all_kwargs.keys():
                id_col = all_kwargs.get("id_col")
                if id_col:
                    if id_col not in idf.columns:
                        raise TypeError(
                            f"Invalid input for ID Column in the function {func.__name__}. {id_col} not found in the dataset."
                        )

            if "label_col" in all_kwargs.keys():
                label_col = all_kwargs.get("label_col")
                event_label = all_kwargs.get("event_label")
                if label_col:
                    if label_col not in idf.columns:
                        raise TypeError(
                            f"Invalid input for Label Column in the function {func.__name__}. {label_col} not found in the dataset."
                        )
                    if idf.where(F.col(label_col) == event_label).count() == 0:
                        raise TypeError(
                            f"Invalid input for Event Label Value in the function {func.__name__}. {event_label} not found in the {label_col} column."
                        )

        if "run_type" in all_kwargs.keys():
            if all_kwargs.get("run_type") not in ("local", "emr", "databricks"):
                raise TypeError(
                    f"Invalid input for run_type in the function {func.__name__}. run_type should be local, emr or databricks - Received '{all_kwargs.get('run_type')}'."
                )

        for arg in [
            "corr_threshold",
            "iv_threshold",
            "drift_threshold_model",
            "coverage",
        ]:
            if arg in all_kwargs.keys():
                arg_val = float(all_kwargs.get(arg))
                if (arg_val < 0) | (arg_val > 1):
                    raise TypeError(
                        f"Invalid input for {arg} Value in the function {func.__name__}. {arg} should be between 0 & 1 - Received '{all_kwargs.get(arg)}'."
                    )
                all_kwargs[arg] = arg_val

        for boolarg in ("drift_detector", "outlier_charts"):
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

        if func.__name__ == "charts_to_objects":
            if all_kwargs.get("bin_method") not in ("equal_frequency", "equal_range"):
                raise TypeError(
                    f"Invalid input for bin_method in the function {func.__name__}. bin_method should be equal_frequency or equal_range - Received '{all_kwargs.get('bin_method')}'."
                )

            bin_size = int(all_kwargs.get("bin_size"))
            if bin_size < 2:
                raise TypeError(
                    f"Invalid input for bin_size in the function {func.__name__}. bin_size should be atleast 2 - Received '{bin_size}'."
                )
            else:
                all_kwargs["bin_size"] = bin_size

        return func(**all_kwargs)

    return wrapper
