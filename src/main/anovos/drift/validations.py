from functools import wraps, partial

from loguru import logger

from anovos.shared.utils import attributeType_segregation


def check_list_of_columns(
    func=None,
    columns="list_of_cols",
    target="idf_target",
    drop="drop_cols",
):
    if func is None:
        return partial(check_list_of_columns, columns=columns, target=target, drop=drop)

    @wraps(func)
    def validate(*args, **kwargs):
        logger.debug("check the list of columns")

        if args[columns] == "all" and args[target] is not None:
            num_cols, cat_cols, other_cols = attributeType_segregation(args[target])
            cols = num_cols + cat_cols

        if isinstance(args[columns], str):
            cols = [x.strip() for x in args[columns].split("|")]

        if isinstance(args[drop], str):
            drops = [x.strip() for x in args[drop].split("|")]

        cols = list(set([e for e in cols if e not in drops]))

        if (len(cols) == 0) | any(x not in args[target].columns for x in cols):
            raise TypeError("Invalid input for Column(s)")

        args[columns] = cols
        args[drop] = drops

        return func(*args, **kwargs)

    return validate


def check_distance_method(func=None, param="method_type"):

    if func is None:
        return partial(check_distance_method, argument_name=param)

    @wraps(func)
    def validate(*args, **kwargs):
        dist_distance_methods = args[param]

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
