import pytest
import pandas
import numpy
from numpy.testing import assert_almost_equal
from anovos.drift.detector import (
    stability_index_computation,
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


def test_that_stability_index_can_be_calculated(
    idfs_numerical, cols_to_check_numerical
):

    df_stability = stability_index_computation(idfs_numerical).toPandas()

    df_stability.index = df_stability["attribute"]
    assert_almost_equal(
        df_stability.loc["A", cols_to_check_numerical],
        [0.162, 0.62, 0.198, 2.0, 0.0, 2.0, 1.4, 0.0],
        3,
    )
