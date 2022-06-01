from pandas import util
import pandas
import numpy
from anovos.data_ingest.data_ingest import read_dataset
from anovos.drift_stability.stability import (
    stability_index_computation,
    feature_stability_estimation,
)


def test_stability_index_computation(spark_session):
    list1 = numpy.array([4.34, 4.76, 4.32, 3.39, 3.67, 4.61, 4.03, 4.93, 3.84, 3.31])
    list2 = numpy.array([6.34, 4.76, 6.32, 3.39, 5.67, 4.61, 6.03, 4.93, 5.84, 3.31])
    list3 = numpy.array([8.34, 4.76, 8.32, 3.39, 7.67, 4.61, 8.03, 4.93, 3.84, 3.31])
    
    cols_to_check = [
        "mean_cv",
        "stddev_cv",
        "kurtosis_cv",
        "mean_si",
        "stddev_si",
        "kurtosis_si",
        "stability_index",
        "flagged",
    ]

    idfs = []
    for l in [list1, list2, list3]:
        idfs.append(spark_session.createDataFrame(pandas.DataFrame({"A": l})))

    df_stability = stability_index_computation(
        spark_session, *idfs, appended_metric_path="unit_testing/stats/stability/df1_3"
    ).toPandas()
    df_stability.index = df_stability["attribute"]
    df_stability_result = [
        round(float(i), 3) for i in df_stability.loc["A", cols_to_check].tolist()
    ]
    assert df_stability_result == [0.132, 0.507, 0.162, 2.0, 0.0, 2.0, 1.4, 0.0]

    idf_new = spark_session.createDataFrame(pandas.DataFrame({"A": list3 + 1}))
    df_stability = stability_index_computation(
        spark_session,
        idf_new,
        existing_metric_path="unit_testing/stats/stability/df1_3",
    ).toPandas()
    df_stability.index = df_stability["attribute"]
    df_stability_result = [
        round(float(i), 3) for i in df_stability.loc["A", cols_to_check].tolist()
    ]
    assert df_stability_result == [0.174, 0.451, 0.177, 2.0, 1.0, 2.0, 1.7, 0.0]

    list1 = numpy.array([0] * 10 + [1] * 10)
    list2 = numpy.array([0] * 12 + [1] * 8)
    list3 = numpy.array([0] * 14 + [1] * 6)
    cols_to_check = ["mean_stddev", "mean_si", "stability_index", "flagged"]

    idf = []
    for l in [list1, list2, list3]:
        idf.append(spark_session.createDataFrame(pandas.DataFrame({"A": l})))

    df_stability = stability_index_computation(
        spark_session, *idf, binary_cols="A"
    ).toPandas()
    df_stability.index = df_stability["attribute"]

    df_stability_result = [
        round(float(i), 3) for i in df_stability.loc["A", cols_to_check].tolist()
    ]
    assert df_stability_result == [0.082, 0.6, 0.6, 1.0]


def test_feature_stability_estimation(spark_session):
    list1 = numpy.array([4.34, 4.76, 4.32, 3.39, 3.67, 4.61, 4.03, 4.93, 3.84, 3.31])
    list2 = numpy.array([6.34, 4.76, 6.32, 3.39, 5.67, 4.61, 6.03, 4.93, 5.84, 3.31])
    list3 = numpy.array([8.34, 4.76, 8.32, 3.39, 7.67, 4.61, 8.03, 4.93, 3.84, 3.31])
    list4 = numpy.array([9.34, 4.76, 9.32, 3.39, 8.67, 4.61, 9.03, 4.93, 3.84, 3.31])

    idfs = []
    for l in [list1, list2, list3, list4]:
        idfs.append(spark_session.createDataFrame(pandas.DataFrame({"A": l})))

    metric_path = "unit_testing/stats/stability/df1_4"
    stability_index_computation(
        spark_session, *idfs, appended_metric_path=metric_path
    ).toPandas()
    attribute_stats = read_dataset(
        spark_session, metric_path, "csv", {"header": True, "inferSchema": True}
    )

    df_stability = feature_stability_estimation(
        spark_session, attribute_stats, {"A": "A**2"}
    ).toPandas()
    df_stability.index = df_stability["feature_formula"]
    cols_to_check = [
        "mean_cv",
        "stddev_cv",
        "mean_si",
        "stddev_si",
        "stability_index_lower_bound",
        "stability_index_upper_bound",
        "flagged_lower",
        "flagged_upper",
    ]
    df_stability_result = [
        round(float(i), 3) for i in df_stability.loc["A**2", cols_to_check].tolist()
    ]
    assert df_stability_result == [0.322, 0.59, 1.0, 0.0, 0.5, 1.3, 1.0, 0.0]

    df_stability = feature_stability_estimation(
        spark_session,
        attribute_stats,
        {"A": "A**2"},
        metric_weightages={"mean": 0.7, "stddev": 0.3},
    ).toPandas()
    df_stability.index = df_stability["feature_formula"]
    df_stability_result = [
        round(float(i), 3) for i in df_stability.loc["A**2", cols_to_check].tolist()
    ]
    assert df_stability_result == [0.322, 0.59, 1.0, 0.0, 0.7, 0.7, 1.0, 1.0]
