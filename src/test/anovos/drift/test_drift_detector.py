import pandas
import numpy
from anovos.drift_stability.drift_detector import drift_statistics
from numpy.testing import assert_almost_equal
from anovos.drift_stability.validations import (
    generate_list_of_cols,
    generate_method_type,
)
from pandas.util.testing import makeDataFrame
from pytest import raises


def test_that_drift_statistics_can_be_calculated(spark_session):

    rand_numbers = numpy.array(
        [0.34, -1.76, 0.32, -0.39, -0.67, 0.61, 1.03, 0.93, -0.84, -0.31]
    )
    idf_target = spark_session.createDataFrame(
        pandas.DataFrame({"A": rand_numbers, "B": rand_numbers})
    )
    idf_source = spark_session.createDataFrame(
        pandas.DataFrame({"A": rand_numbers, "B": rand_numbers + 1})
    )

    df_statistics = drift_statistics(
        spark_session, idf_target, idf_source, method_type="all"
    ).toPandas()

    df_statistics_equal_freq = drift_statistics(
        spark_session,
        idf_target,
        idf_source,
        method_type="all",
        bin_method="equal_frequency",
        print_impact=True,
    ).toPandas()

    df_statistics.index = df_statistics["attribute"]
    df_statistics_equal_freq.index = df_statistics_equal_freq["attribute"]

    assert df_statistics.loc["A", "PSI":"KS"].tolist() == [0, 0, 0, 0]
    assert_almost_equal(
        df_statistics.loc["B", "PSI":"KS"], [7.6776, 0.3704, 0.7091, 0.4999], 4,
    )
    assert df_statistics.loc[["A", "B"], "flagged"].tolist() == [0, 1]
    assert df_statistics_equal_freq.loc["A", "PSI":"KS"].tolist() == [0, 0, 0, 0]
    assert_almost_equal(
        df_statistics_equal_freq.loc["B", "PSI":"KS"], [3.0899, 0.1769, 0.4775, 0.4], 4
    )
    assert df_statistics_equal_freq.loc[["A", "B"], "flagged"].tolist() == [0, 1]


def test_that_non_boolean_input_for_expected_boolean_raises_error(spark_session):

    rand_numbers = numpy.array(
        [0.34, -1.76, 0.32, -0.39, -0.67, 0.61, 1.03, 0.93, -0.84, -0.31]
    )
    idf_target = spark_session.createDataFrame(
        pandas.DataFrame({"A": rand_numbers, "B": rand_numbers})
    )
    idf_source = spark_session.createDataFrame(
        pandas.DataFrame({"A": rand_numbers, "B": rand_numbers + 1})
    )

    with raises(TypeError):
        drift_statistics(
            spark_session, idf_target, idf_source, pre_existing_source="str"
        )
    with raises(TypeError):
        drift_statistics(
            spark_session, idf_target, idf_source, pre_computed_stats="str"
        )
    with raises(TypeError):
        drift_statistics(spark_session, idf_target, idf_source, print_impact="str")


def test_that_list_of_cols_can_be_generated(spark_session):

    df = spark_session.createDataFrame(makeDataFrame())
    list_of_cols = "A|B|C"

    list_of_cols = generate_list_of_cols(
        list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=[]
    )

    assert list_of_cols == ["A", "B", "C"]

    list_of_cols = "all"

    list_of_cols = generate_list_of_cols(
        list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=[]
    )

    assert list_of_cols == ["A", "B", "C", "D"]


def test_that_generate_list_of_cols_drops_cols(spark_session):

    df = spark_session.createDataFrame(makeDataFrame())
    list_of_cols = "all"
    drop_cols = ["A", "B"]

    list_of_cols = generate_list_of_cols(
        list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=drop_cols
    )

    assert list_of_cols == ["C", "D"]

    list_of_cols = "all"
    drop_cols = "A|B"

    list_of_cols = generate_list_of_cols(
        list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=drop_cols
    )

    assert list_of_cols == ["C", "D"]


def test_that_empty_list_of_cols_raises_error(spark_session):

    df = spark_session.createDataFrame(makeDataFrame())
    list_of_cols = []
    with raises(ValueError):
        generate_list_of_cols(
            list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=[]
        )


def test_that_wrong_column_name_raises_error(spark_session):

    df = spark_session.createDataFrame(makeDataFrame())
    list_of_cols = ["W"]

    with raises(ValueError):
        generate_list_of_cols(
            list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=[]
        )


def test_that_non_numeric_column_raises_error(spark_session):

    df = pandas.DataFrame({"A": [1, 2], "B": ["a", "b"]})
    df = spark_session.createDataFrame(df)
    list_of_cols = ["B"]

    with raises(ValueError):
        generate_list_of_cols(
            list_of_cols=list_of_cols, idf_target=df, idf_source=df, drop_cols=[]
        )


def test_generate_method_type():

    method_type = "all"
    method_type = generate_method_type(method_type)
    assert method_type == ["PSI", "JSD", "HD", "KS"]

    method_type = "PSI|JSD"
    method_type = generate_method_type(method_type)
    assert method_type == ["PSI", "JSD"]

    method_type = "PSI"
    method_type = generate_method_type(method_type)
    assert method_type == ["PSI"]


def test_that_wrong_bin_method_raises_error(spark_session):

    df = spark_session.createDataFrame(makeDataFrame())

    with raises(TypeError):
        generate_list_of_cols(
            list_of_cols="all", idf_target=df, idf_source=df, bin_method="42"
        )
