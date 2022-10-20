import pytest
import pandas
import numpy
from numpy.testing import assert_almost_equal
from anovos.drift.detector import (
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
def attribute_stats(spark_session, idfs_numerical):
    metric_path = "unit_testing/stats/stability/df1_4"
    stability_index_computation(
        spark_session, *idfs_numerical, appended_metric_path=metric_path
    ).toPandas()
    attribute_stats = read_dataset(
        spark_session, metric_path, "csv", {"header": True, "inferSchema": True}
    )
    return attribute_stats
