from anovos.shared.utils import attributeType_segregation


def parse_columns(list_of_cols, idf, drop_cols=None):
    drop_cols = drop_cols or []

    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]

    if any(x not in list_of_cols for x in drop_cols):
        invalid_cols = [x for x in drop_cols if x not in list_of_cols]
        raise ValueError(
            f"Invalid input for drop_cols. Invalid Column(s): {invalid_cols} not found in source dataframe."
        )

    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if len(list_of_cols) == 0:
        raise ValueError(
            f"Invalid input for column(s) in the function drift_statistics."
        )
    if any(x not in idf.columns for x in list_of_cols):
        invalid_cols = [x for x in list_of_cols if x not in idf.columns]
        raise ValueError(
            f"Invalid input for list_of_cols in the function drift_statistics. Invalid Column(s): {invalid_cols} not found in source dataframe."
        )

    return list_of_cols


def parse_numerical_columns(list_of_cols, idf, drop_cols=None):
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == "all":
        list_of_cols = num_cols

    list_of_cols = parse_columns(list_of_cols, idf, drop_cols)

    return list_of_cols


def parse_method_type(method_type):

    if isinstance(method_type, str):
        if method_type == "all":
            method_type = ["PSI", "JSD", "HD", "KS"]
        else:
            method_type = [x.strip() for x in method_type.split("|")]
    if any(x not in ("PSI", "JSD", "HD", "KS") for x in method_type):
        raise TypeError(f"Invalid input for method_type")

    return method_type
