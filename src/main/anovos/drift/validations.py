from functools import wraps, partial

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

        cols, drops = [], []

        idf_target = kwargs.get(target, "") or args[target_idx]

        if kwargs[columns] == "all" and idf_target is not None:
            num_cols, cat_cols, other_cols = attributeType_segregation(idf_target)
            cols = num_cols + cat_cols

        if isinstance(kwargs[columns], str):
            cols = [x.strip() for x in kwargs[columns].split("|")]

        if isinstance(kwargs[drop], str):
            drops = [x.strip() for x in kwargs[drop].split("|")]

        cols = list(set([e for e in cols if e not in drops]))

        if (len(cols) == 0) | any(x not in idf_target.columns for x in cols):
            raise TypeError("Invalid input for Column(s)")

        kwargs[columns] = cols
        kwargs[drop] = drops

        return func(*args, **kwargs)

    return validate


def check_distance_method(func=None, param="method_type"):
    if func is None:
        return partial(check_distance_method, argument_name=param)

    @wraps(func)
    def validate(*args, **kwargs):
        dist_distance_methods = kwargs[param]

        if isinstance(dist_distance_methods, str):
            if dist_distance_methods == "all":
                dist_distance_methods = ["PSI", "JSD", "HD", "KS"]
            else:
                dist_distance_methods = [
                    x.strip() for x in dist_distance_methods.split("|")
                ]

        if any(x not in ("PSI", "JSD", "HD", "KS") for x in dist_distance_methods):
            raise TypeError(f"Invalid input for {param}")

        return func(*args, **kwargs)

    return validate
