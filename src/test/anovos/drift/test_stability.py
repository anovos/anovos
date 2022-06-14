from pandas import util
import pytest
import pandas
import numpy
from numpy.testing import assert_almost_equal
from anovos.data_ingest.data_ingest import read_dataset
from anovos.drift_stability.stability import (
    stability_index_computation,
    feature_stability_estimation,
)


@pytest.fixture
def idfs_numerical(spark_session):
    list1 = numpy.array([4.34, 4.76, 4.32, 3.39, 3.67, 4.61, 4.03, 4.93, 3.84, 3.31])
    list2 = numpy.array([6.34, 4.76, 6.32, 3.39, 5.67, 4.61, 6.03, 4.93, 5.84, 3.31])
    list3 = numpy.array([8.34, 4.76, 8.32, 3.39, 7.67, 4.61, 8.03, 4.93, 3.84, 3.31])

    idfs = []
    for l in [list1, list2, list3]:
        idfs.append(spark_session.createDataFrame(pandas.DataFrame({"A": l})))
    return idfs


@pytest.fixture
def idf_new(spark_session):
    list3 = numpy.array([8.34, 4.76, 8.32, 3.39, 7.67, 4.61, 8.03, 4.93, 3.84, 3.31])
    return spark_session.createDataFrame(pandas.DataFrame({"A": list3 + 1}))


@pytest.fixture
def idfs_binary(spark_session):
    list1 = numpy.array([0] * 10 + [1] * 10)
    list2 = numpy.array([0] * 12 + [1] * 8)
    list3 = numpy.array([0] * 14 + [1] * 6)

    idf = []
    for l in [list1, list2, list3]:
        idf.append(spark_session.createDataFrame(pandas.DataFrame({"A": l})))
    return idf


@pytest.fixture
def cols_to_check_numerical():
    return [
        "mean_cv",
        "stddev_cv",
        "kurtosis_cv",
        "mean_si",
        "stddev_si",
        "kurtosis_si",
        "stability_index",
        "flagged",
    ]


@pytest.fixture
def cols_to_check_binary():
    return ["mean_stddev", "mean_si", "stability_index", "flagged"]


@pytest.fixture
def cols_to_check_si_estimation():
    return [
        "mean_cv",
        "stddev_cv",
        "mean_si",
        "stddev_si",
        "stability_index_lower_bound",
        "stability_index_upper_bound",
        "flagged_lower",
        "flagged_upper",
    ]


@pytest.fixture
def attribute_stats(spark_session, idfs_numerical):
    metric_path = "unit_testing/stats/stability/df1_4"
    stability_index_computation(
        spark_session, *idfs_numerical, appended_metric_path=metric_path
    ).toPandas()
    attribute_stats = read_dataset(
        spark_session, metric_path, "csv", {"header": True, "inferSchema": True}
    )
    return attribute_stats


def test_that_stability_index_can_be_calculated(
    spark_session, idfs_numerical, cols_to_check_numerical
):

    df_stability = stability_index_computation(
        spark_session,
        *idfs_numerical,
        appended_metric_path="unit_testing/stats/stability/df1_3"
    ).toPandas()

    df_stability.index = df_stability["attribute"]
    assert_almost_equal(
        df_stability.loc["A", cols_to_check_numerical],
        [0.132, 0.507, 0.162, 2.0, 0.0, 2.0, 1.4, 0.0],
        3,
    )


def test_that_existing_metric_can_be_used(
    spark_session, idf_new, cols_to_check_numerical
):

    df_stability = stability_index_computation(
        spark_session,
        idf_new,
        existing_metric_path="unit_testing/stats/stability/df1_3",
    ).toPandas()
    df_stability.index = df_stability["attribute"]
    assert_almost_equal(
        df_stability.loc["A", cols_to_check_numerical],
        [0.174, 0.451, 0.177, 2.0, 1.0, 2.0, 1.7, 0.0],
        3,
    )


def test_that_binary_column_can_be_calculated(
    spark_session, idfs_binary, cols_to_check_binary
):
    df_stability = stability_index_computation(
        spark_session, *idfs_binary, binary_cols="A"
    ).toPandas()
    df_stability.index = df_stability["attribute"]
    assert_almost_equal(
        df_stability.loc["A", cols_to_check_binary], [0.082, 0.6, 0.6, 1.0], 3
    )


def test_that_feature_stability_can_be_estimated(
    spark_session, attribute_stats, cols_to_check_si_estimation
):
    df_stability = feature_stability_estimation(
        spark_session, attribute_stats, {"A": "A**2"}
    ).toPandas()
    df_stability.index = df_stability["feature_formula"]
    assert_almost_equal(
        df_stability.loc["A**2", cols_to_check_si_estimation],
        [0.298, 0.603, 1.0, 0.0, 0.5, 1.3, 1.0, 0.0],
        3,
    )


def test_that_different_weightages_can_be_used(
    spark_session, attribute_stats, cols_to_check_si_estimation
):
    df_stability = feature_stability_estimation(
        spark_session,
        attribute_stats,
        {"A": "A**2"},
        metric_weightages={"mean": 0.7, "stddev": 0.3},
    ).toPandas()
    df_stability.index = df_stability["feature_formula"]
    assert_almost_equal(
        df_stability.loc["A**2", cols_to_check_si_estimation],
        [0.298, 0.603, 1.0, 0.0, 0.7, 0.7, 1.0, 1.0],
        3,
    )
