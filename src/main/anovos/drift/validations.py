from functools import wraps

from loguru import logger

from anovos.shared.utils import attributeType_segregation


def check_list_of_columns(
    columns="list_of_cols",
    target="idf_target",
    drop="drop_cols",
):
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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

        return wrapper

    return _decorator


def check_distance_method(argument_name="methods_type"):
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            dist_distance_methods = args[argument_name]

            if isinstance(dist_distance_methods, str):
                if dist_distance_methods == "all":
                    dist_distance_methods = ["PSI", "JSD", "HD", "KS"]
                else:
                    dist_distance_methods = [
                        x.strip() for x in dist_distance_methods.split("|")
                    ]

            if any(x not in ("PSI", "JSD", "HD", "KS") for x in dist_distance_methods):
                raise TypeError(f"Invalid input for {argument_name}")

            return func(*args, **kwargs)

        return wrapper

    return _decorator
