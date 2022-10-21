import pandas
import numpy
from anovos.drift_stability.drift_detector import statistics
from numpy.testing import assert_almost_equal


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

    df_statistics = statistics(
        spark_session, idf_target, idf_source, method_type="all"
    ).toPandas()

    df_statistics_equal_freq = statistics(
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
    assert df_statistics.loc[["A", "B"], "flagged"].tolist() == [0, 1]
    assert_almost_equal(
        df_statistics.loc["B", "PSI":"KS"],
        [7.6776, 0.7091, 0.3704, 0.4999],
        4,
    )
    assert df_statistics_equal_freq.loc["A", "PSI":"KS"].tolist() == [0, 0, 0, 0]
    assert_almost_equal(
        df_statistics_equal_freq.loc["B", "PSI":"KS"], [3.0899, 0.4775, 0.1769, 0.4], 4
    )
    assert df_statistics_equal_freq.loc[["A", "B"], "flagged"].tolist() == [0, 1]
