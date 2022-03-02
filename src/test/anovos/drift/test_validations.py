import pytest

from anovos.drift.validations import check_list_of_columns


@check_list_of_columns
def fut(
    spark=None, idf_target=None, idf_source=None, *, list_of_cols="all", drop_cols=None
):
    pass


def test_that_empty_set_of_columns_raises_value_error():
    with pytest.raises(ValueError):
        fut(None, None, None, list_of_cols=[], drop_cols=[])


def test_that_dropping_all_columns_raises_value_error():
    with pytest.raises(ValueError):
        fut(None, None, None, list_of_cols=["a", "b"], drop_cols=["a", "b"])
