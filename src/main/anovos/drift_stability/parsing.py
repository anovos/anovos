from anovos.shared.utils import attributeType_segregation
import warnings


def parse_columns(list_of_cols, idfs, drop_cols=None):
    drop_cols = drop_cols or []

    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    if any(x not in list_of_cols for x in drop_cols):
        invalid_cols = [x for x in drop_cols if x not in list_of_cols]
        warnings.warn(
            f"Invalid input for column(s): {invalid_cols} not found in list of columns.",
            UserWarning,
        )

    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if len(list_of_cols) == 0:
        raise ValueError(
            f"Invalid input for column(s) in the function drift_statistics."
        )
    if len(idfs) > 0:
        for idf in idfs:
            if any(x not in idf.columns for x in list_of_cols):
                invalid_cols = [x for x in list_of_cols if x not in idf.columns]
                raise ValueError(
                    f"Invalid input for list_of_cols in the function drift_statistics. Invalid Column(s): {invalid_cols} not found in source dataframe."
                )

    return list_of_cols


def parse_numerical_columns(list_of_cols, idfs, drop_cols=None):
    idfs = idfs or []
    if not isinstance(idfs, list):
        idfs = [idfs]
    if len(idfs) > 0:
        num_cols = attributeType_segregation(idfs[0])[0]
        if list_of_cols == "all":
            list_of_cols = num_cols
    list_of_cols = parse_columns(list_of_cols, idfs, drop_cols)

    return list_of_cols


def parse_method_type(method_type):

    if isinstance(method_type, str):
        if method_type == "all":
            method_type = ["PSI", "JSD", "HD", "KS"]
        else:
            method_type = [x.strip() for x in method_type.split("|")]
    if any(x not in ("PSI", "JSD", "HD", "KS") for x in method_type):
        invalid_method = [x for x in method_type if x not in ("PSI", "JSD", "HD", "KS")]
        raise ValueError(f"Invalid input for method_type: {invalid_method}")

    return method_type
