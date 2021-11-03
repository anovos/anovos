from itertools import chain

from pyspark.sql import functions as F


def flatten_dataframe(idf, fixed_cols):
    """
    :param idf: Input Dataframe
    :param fixed_cols: All columns except in this list will be melted/unpivoted
    :return: Flatten/Melted dataframe
    """
    valid_cols = [e for e in idf.columns if e not in fixed_cols]
    key_and_val = F.create_map(list(chain.from_iterable([[F.lit(c), F.col(c)] for c in valid_cols])))
    odf = idf.select(*fixed_cols, F.explode(key_and_val))
    return odf


def transpose_dataframe(idf, fixed_col):
    """
    :param idf: Input Dataframe
    :param fixed_col: Values in this column will be converted into columns as header.
                Ideally all values should be unique
    :return: Transposed dataframe
    """
    idf_flatten = flatten_dataframe(idf, fixed_cols=[fixed_col])
    odf = idf_flatten.groupBy('key').pivot(fixed_col).agg(F.first('value'))
    return odf


def attributeType_segregation(idf):
    """
    :param idf: Input Dataframe
    :return: (list1, list2, list3)
             3 lists - each corresponding to numerical, categorical, and others columns
    """
    cat_cols = []
    num_cols = []
    other_cols = []

    for i in idf.dtypes:
        if i[1] == 'string':
            cat_cols.append(i[0])
        elif (i[1] in ('double', 'int', 'bigint', 'float', 'long')) | (i[1].startswith('decimal')):
            num_cols.append(i[0])
        else:
            other_cols.append(i[0])
    return num_cols, cat_cols, other_cols


def get_dtype(idf, col):
    """
    :param idf: Input Dataframe
    :param col: Column Name for datatype detection
    :return: data type
    """
    return [dtype for name, dtype in idf.dtypes if name == col][0]


def ends_with(string, end_str="/"):
    """
    :param string: "s3:mw-bucket"
    :param end_str: "/"
    :return: "s3:mw-bucket/"
    """
    string = str(string)
    if string.endswith(end_str):
        return string
    return string + end_str


def pairwise_reduce(op, x):
    while len(x) > 1:
        v = [op(i, j) for i, j in zip(x[::2], x[1::2])]
        if len(x) > 1 and len(x) % 2 == 1:
            v[-1] = op(v[-1], x[-1])
        x = v
    return x[0]
