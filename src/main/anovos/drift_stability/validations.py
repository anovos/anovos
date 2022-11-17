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


def compute_score(value, method_type, cv_thresholds=[0.03, 0.1, 0.2, 0.5]):
    """
    This function maps CV or SD to a score between 0 and 4.
    """
    if value is None:
        return None

    if method_type == "cv":
        cv = abs(value)
        stability_index = [4, 3, 2, 1, 0]
        for i, thresh in enumerate(cv_thresholds):
            if cv < thresh:
                return float(stability_index[i])
        return float(stability_index[-1])

    elif method_type == "sd":
        sd = value
        if sd <= 0.005:
            return 4.0
        elif sd <= 0.01:
            return round(-100 * sd + 4.5, 1)
        elif sd <= 0.05:
            return round(-50 * sd + 4, 1)
        elif sd <= 0.1:
            return round(-30 * sd + 3, 1)
        else:
            return 0.0

    else:
        raise TypeError("method_type must be either 'cv' or 'sd'.")


def compute_si(metric_weightages):
    def compute_si_(attr_type, mean_stddev, mean_cv, stddev_cv, kurtosis_cv):
        if attr_type == "Binary":
            mean_si = compute_score(mean_stddev, "sd")
            stability_index = mean_si
            stddev_si, kurtosis_si = None, None
        else:
            mean_si = compute_score(mean_cv, "cv")
            stddev_si = compute_score(stddev_cv, "cv")
            kurtosis_si = compute_score(kurtosis_cv, "cv")
            if mean_si is None or stddev_si is None or kurtosis_si is None:
                stability_index = None
            else:
                stability_index = round(
                    mean_si * metric_weightages.get("mean", 0)
                    + stddev_si * metric_weightages.get("stddev", 0)
                    + kurtosis_si * metric_weightages.get("kurtosis", 0),
                    4,
                )
        return [mean_si, stddev_si, kurtosis_si, stability_index]

    return compute_si_


def check_metric_weightages(metric_weightages):
    if (
        round(
            metric_weightages.get("mean", 0)
            + metric_weightages.get("stddev", 0)
            + metric_weightages.get("kurtosis", 0),
            3,
        )
        != 1
    ):
        raise ValueError(
            "Invalid input for metric weightages. Either metric name is incorrect or sum of metric weightages is not 1.0."
        )


def check_threshold(threshold):
    if (threshold < 0) or (threshold > 4):
        raise ValueError(
            "Invalid input for metric threshold. It must be a number between 0 and 4."
        )
