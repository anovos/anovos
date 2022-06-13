import pytest
from pandas.util.testing import makeDataFrame
from anovos.drift_stability.parsing import (
    parse_columns,
    parse_numerical_columns,
    parse_method_type,
)


@pytest.fixture
def default_df(spark_session):
    return spark_session.createDataFrame(makeDataFrame())


def test_that_all_returns_all_columns(default_df):

    list_of_cols = parse_numerical_columns(list_of_cols="all", idf=default_df)

    assert list_of_cols == ["A", "B", "C", "D"]


def test_that_string_with_pipes_is_parsed(default_df):

    list_of_cols = parse_columns(list_of_cols="A|B|C", idf=default_df)

    assert list_of_cols == ["A", "B", "C"]


def test_that_unknown_column_raises_error(default_df):

    with pytest.raises(ValueError):
        parse_columns(list_of_cols=["W"], idf=default_df)


def test_that_dropping_unselected_column_raises_warning(default_df):

    with pytest.warns(UserWarning):
        parse_columns(list_of_cols=["A", "B"], idf=default_df, drop_cols=["C", "D"])


def test_that_all_methods_returns_all():

    method_type = parse_method_type("all")

    assert method_type == ["PSI", "JSD", "HD", "KS"]


def test_that_string_methods_as_pipe_is_parsed():

    method_type = parse_method_type(method_type="JSD|HD|KS")

    assert method_type == ["JSD", "HD", "KS"]


def test_that_unknown_method_raises_warning():

    with pytest.raises(ValueError):
        parse_method_type(method_type="JSD|HAD")
