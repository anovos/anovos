import math
import numbers


def is_close(x, y, rel_tol=1e-09, abs_tol=0.0):
    """Verifies close numbers.

    Parameters
    ----------
    x, y: values to be evaluated
    rel_tol: double, relative tolerance, Default: 1e-09
    abs_tol: double, absolute tolerance, Default: 0

    Returns
    -------
    bool
    """
    return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)


def check_equality(x, y, exact=True):
    """Verifies two numbers for equality.
    In case both values are either float('NaN') or None, this will pass the equality check

    Parameters
    ----------
    x, y: two values to be evaluated
    exact: bool
        If set to True: evaluates exact equality
        If set to False: if x, y are instance of numbers.Numbers
            `is_close(x, y)` will be called instead of `x == y`

    Returns
    -------
    bool
    """

    is_equal = False
    if x is None and y is None:
        is_equal = True
    elif isinstance(x, numbers.Number) and isinstance(y, numbers.Number):
        if math.isnan(x) and math.isnan(y):
            is_equal = True
        elif not exact and is_close(x, y):
            is_equal = True
        elif x == y:
            is_equal = True
    else:
        is_equal = x == y
    return is_equal


def assert_spark_frame_equal(df_x, df_y, exact):
    """Asserts the equality of two Spark DataFrames.
    Should only be used for small data frames.

    Parameters
    ----------
    df_x, df_y: Spark DataFrames
    exact: bool
        If set to True: evaluates exact equality
        If set to False: `is_close()` will be used for equality check

    Returns
    -------
    bool
    """

    # Might not be enough to sort single column, if there are duplicate values in it.
    sort_key = sorted(df_x.columns)
    df_x = df_x.sort(*sort_key)
    df_y = df_y.sort(*sort_key)

    if df_x != df_y:
        n_x = df_x.count()
        n_y = df_y.count()
        if n_x != n_y:
            raise AssertionError(f"n_rows - X: {n_x} != Y: {n_y}")

        msgs = []
        y_rows = df_y.rdd.collect()
        fields = y_rows[0].__fields__
        for i_row, x_row in enumerate(df_x.rdd.collect()):
            y_row = y_rows[i_row]
            msg = ""
            for current_field in fields:
                x_val = x_row[current_field]
                y_val = y_row[current_field]
                if not check_equality(x_val, y_val, exact):
                    if not msg:
                        msg = f"row counter: {i_row}"
                    msg = f"{msg}\n\tfield `{current_field}` `{x_val}` != `{y_val}`"
            if msg:
                msgs.append(msg)
        if msgs:
            msgs.insert(0, "DataFrames are not equal:")
            msg = "\n".join(msgs)
            raise AssertionError(msg)
