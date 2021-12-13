import pytest

from anovos.data_analyzer.quality_checker import _parse_columns, duplicate_detection


def test_that_special_case_all_is_parsed():
    columns = _parse_columns("all")

    assert len(columns) == 1
    assert columns == ["all"]


def test_that_string_is_parsed():
    columns = _parse_columns("column_A| column_B")

    assert len(columns) == 2
    assert columns == ["column_A", "column_B"]


def test_that_list_is_parsed():
    input_cols = ["column_with_a_name", "values", "OUTPUT"]
    columns = _parse_columns(input_cols)

    assert len(columns) == 3
    assert columns == input_cols


def test_that_dropping_all_columns_causes_error(spark_session):
    df = spark_session.createDataFrame(
        [
            (1, 2),
            (3, 4)
        ],
        ['col1', 'col2']
    )

    with pytest.raises(ValueError):
        output_df, report_df = duplicate_detection(spark_session, df, drop_columns=["col1", "col2"])


