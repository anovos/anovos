from functools import partial, wraps

from loguru import logger

from anovos.shared.utils import attributeType_segregation


def check_list_of_columns(
    func=None,
    columns="list_of_cols",
    target_idx: int = 1,
    target: str = "idf_target",
    drop="drop_cols",
):
    if func is None:
        return partial(check_list_of_columns, columns=columns, target=target, drop=drop)

    @wraps(func)
    def validate(*args, **kwargs):
        logger.debug("check the list of columns")

        idf_target = kwargs.get(target, "") or args[target_idx]

        cols_raw = kwargs.get(columns, "all")
        if isinstance(cols_raw, str):
            if cols_raw == "all":
                num_cols, cat_cols, other_cols = attributeType_segregation(idf_target)
                cols = num_cols + cat_cols
            else:
                cols = [x.strip() for x in cols_raw.split("|")]
        elif isinstance(cols_raw, list):
            cols = cols_raw
        else:
            raise TypeError(
                f"'{columns}' must be either a string or a list of strings."
                f" Received {type(cols_raw)}."
            )

        drops_raw = kwargs.get(drop, [])
        if isinstance(drops_raw, str):
            drops = [x.strip() for x in drops_raw.split("|")]
        elif isinstance(drops_raw, list):
            drops = drops_raw
        else:
            raise TypeError(
                f"'{drop}' must be either a string or a list of strings. "
                f"Received {type(drops_raw)}."
            )

        final_cols = list(set(e for e in cols if e not in drops))

        if not final_cols:
            raise ValueError(
                f"Empty set of columns is given. Columns to select: {cols}, columns to drop: {drops}."
            )

        if any(x not in idf_target.columns for x in final_cols):
            raise ValueError(
                f"Not all columns are in the input dataframe. "
                f"Missing columns: {set(final_cols) - set(idf_target.columns)}"
            )

        kwargs[columns] = final_cols
        kwargs[drop] = []

        return func(*args, **kwargs)

    return validate


def check_distance_method(func=None, param="method_type"):
    if func is None:
        return partial(check_distance_method, param=param)

    @wraps(func)
    def validate(*args, **kwargs):
        dist_distance_methods = kwargs.get(param, "PSI")

        if isinstance(dist_distance_methods, str):
            if dist_distance_methods == "all":
                dist_distance_methods = ["PSI", "JSD", "HD", "KS"]
            else:
                dist_distance_methods = [
                    x.strip() for x in dist_distance_methods.split("|")
                ]

        if any(x not in ("PSI", "JSD", "HD", "KS") for x in dist_distance_methods):
            raise TypeError(f"Invalid input for {param}")

        kwargs[param] = dist_distance_methods

        return func(*args, **kwargs)

    return validate
