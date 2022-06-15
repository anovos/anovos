import os

import pytest
import platform
from pyspark.sql import functions as F
from pytest import approx

from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_transformer.transformers import (
    IQR_standardization,
    PCA_latentFeatures,
    attribute_binning,
    auto_imputation,
    autoencoder_latentFeatures,
    boxcox_transformation,
    cat_to_num_supervised,
    cat_to_num_unsupervised,
    feature_transformation,
    imputation_matrixFactorization,
    imputation_MMM,
    imputation_sklearn,
    monotonic_binning,
    normalization,
    outlier_categories,
    z_standardization,
)

sample_parquet = "./data/test_dataset/part-00001-3eb0f7bb-05c2-46ec-8913-23ba231d2734-c000.snappy.parquet"


# scaling
def test_z_standardization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = z_standardization(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    odf_stddev_dict = (
        odf.describe().where(F.col("summary") == "stddev").toPandas().to_dict("list")
    )
    assert round(float(odf_stddev_dict["age"][0])) == 1.0
    assert round(float(odf_stddev_dict["fnlwgt"][0])) == 1.0
    assert round(float(odf_stddev_dict["hours-per-week"][0])) == 1.0

    try:
        odf = z_standardization(
            spark_session,
            df,
            list_of_cols=["education-num"],
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = z_standardization(spark_session, df, list_of_cols=[])
    odf_stddev_dict = (
        odf.describe().where(F.col("summary") == "stddev").toPandas().to_dict("list")
    )
    assert round(float(odf_stddev_dict["age"][0])) != 1.0
    assert round(float(odf_stddev_dict["fnlwgt"][0])) != 1.0
    assert round(float(odf_stddev_dict["hours-per-week"][0])) != 1.0

    odf = z_standardization(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        output_mode="append",
    )
    assert len(odf.columns) == 20


def test_IQR_standardization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = IQR_standardization(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    odf_median_dict = (
        odf.summary().where(F.col("summary") == "50%").toPandas().to_dict("list")
    )
    assert round(float(odf_median_dict["age"][0])) == 0.0
    assert round(float(odf_median_dict["fnlwgt"][0])) == 0.0
    assert round(float(odf_median_dict["hours-per-week"][0])) == 0.0

    try:
        odf = IQR_standardization(
            spark_session,
            df,
            list_of_cols=["education-num"],
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = IQR_standardization(spark_session, df, list_of_cols=[])
    odf_median_dict = (
        odf.summary().where(F.col("summary") == "50%").toPandas().to_dict("list")
    )
    assert round(float(odf_median_dict["age"][0])) != 0.0
    assert round(float(odf_median_dict["fnlwgt"][0])) != 0.0
    assert round(float(odf_median_dict["hours-per-week"][0])) != 0.0

    odf = IQR_standardization(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        output_mode="append",
    )
    assert len(odf.columns) == 20


def test_normalization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = normalization(
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["age"][0])) == 0.0
    assert round(float(odf_min_dict["fnlwgt"][0])) == 0.0
    assert round(float(odf_min_dict["hours-per-week"][0])) == 0.0
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_max_dict["age"][0])) == 1.0
    assert round(float(odf_max_dict["fnlwgt"][0])) == 1.0
    assert round(float(odf_max_dict["hours-per-week"][0])) == 1.0

    try:
        odf = normalization(
            df,
            list_of_cols=["age", "fnlwgt", "hours-per-week"],
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = normalization(df, list_of_cols=[])
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["age"][0])) != 0.0
    assert round(float(odf_min_dict["fnlwgt"][0])) != 0.0
    assert round(float(odf_min_dict["hours-per-week"][0])) != 0.0
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_max_dict["age"][0])) != 1.0
    assert round(float(odf_max_dict["fnlwgt"][0])) != 1.0
    assert round(float(odf_max_dict["hours-per-week"][0])) != 1.0

    odf = normalization(
        df, list_of_cols=["age", "fnlwgt", "hours-per-week"], output_mode="append"
    )
    assert len(odf.columns) == 20


# imputation
def test_imputation_sklearn(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_sklearn(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        method_type="KNN",
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    assert odf.where(F.col("age").isNull()).count() == 0
    assert odf.where(F.col("fnlwgt").isNull()).count() == 0
    assert odf.where(F.col("hours-per-week").isNull()).count() == 0
    assert odf.where(F.col("logfnl").isNull()).count() == 10214
    assert odf.where(F.col("education").isNull()).count() == 258
    assert odf.where(F.col("race").isNull()).count() == 162
    assert odf.where(F.col("relationship").isNull()).count() == 4

    try:
        odf = imputation_sklearn(
            spark_session,
            df,
            list_of_cols=["education-num"],
            method_type="KNN",
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = imputation_sklearn(spark_session, df, list_of_cols=[], method_type="KNN")
    assert odf.where(F.col("age").isNull()).count() == 30
    assert odf.where(F.col("fnlwgt").isNull()).count() == 8
    assert odf.where(F.col("hours-per-week").isNull()).count() == 59

    odf = imputation_sklearn(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        method_type="regression",
        output_mode="append",
    )
    assert len(odf.columns) == 20


def test_imputation_matrixFactorization(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet").limit(100)
    odf = imputation_matrixFactorization(
        spark_session,
        df,
        list_of_cols=["education-num", "hours-per-week"],
        id_col="ifa",
    )
    assert len(odf.columns) == 17
    assert odf.where(F.col("hours-per-week").isNull()).count() == 0
    assert odf.where(F.col("education-num").isNull()).count() == 0

    assert (
        odf.where(F.col("education").isNull()).count()
        == df.where(F.col("education").isNull()).count()
    )
    assert (
        odf.where(F.col("race").isNull()).count()
        == df.where(F.col("race").isNull()).count()
    )
    assert (
        odf.where(F.col("relationship").isNull()).count()
        == df.where(F.col("relationship").isNull()).count()
    )


def test_imputation_matrixFactorization_with_empty_list_of_cols(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_matrixFactorization(
        spark_session, df, list_of_cols=[], id_col="ifa"
    )
    assert (
        odf.where(F.col("hours-per-week").isNull()).count()
        == df.where(F.col("hours-per-week").isNull()).count()
    )
    assert (
        odf.where(F.col("education-num").isNull()).count()
        == df.where(F.col("education-num").isNull()).count()
    )


def test_imputation_matrixFactorization_with_appended_output(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_matrixFactorization(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        id_col="ifa",
        output_mode="append",
    )

    assert len(odf.columns) == 20
    assert (
        odf.where(F.col("age").isNull()).count()
        == df.where(F.col("age").isNull()).count()
    )
    assert (
        odf.where(F.col("fnlwgt").isNull()).count()
        == df.where(F.col("fnlwgt").isNull()).count()
    )
    assert (
        odf.where(F.col("hours-per-week").isNull()).count()
        == df.where(F.col("hours-per-week").isNull()).count()
    )
    assert odf.where(F.col("age_imputed").isNull()).count() == 0
    assert odf.where(F.col("fnlwgt_imputed").isNull()).count() == 0
    assert odf.where(F.col("hours-per-week_imputed").isNull()).count() == 0


def test_imputation_MMM(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = imputation_MMM(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week", "relationship", "race"],
        method_type="mode",
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    assert odf.where(F.col("age").isNull()).count() == 0
    assert odf.where(F.col("fnlwgt").isNull()).count() == 0
    assert odf.where(F.col("hours-per-week").isNull()).count() == 0
    assert odf.where(F.col("race").isNull()).count() == 0
    assert odf.where(F.col("relationship").isNull()).count() == 0
    assert odf.where(F.col("logfnl").isNull()).count() == 10214
    assert odf.where(F.col("education").isNull()).count() == 258

    try:
        odf = imputation_MMM(
            spark_session,
            df,
            list_of_cols=["education-num"],
            method_type="mode",
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = imputation_MMM(spark_session, df, list_of_cols=[], method_type="mode")
    assert odf.where(F.col("age").isNull()).count() == 30
    assert odf.where(F.col("fnlwgt").isNull()).count() == 8
    assert odf.where(F.col("hours-per-week").isNull()).count() == 59
    assert odf.where(F.col("race").isNull()).count() == 162
    assert odf.where(F.col("relationship").isNull()).count() == 4

    odf = imputation_MMM(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week", "relationship", "race"],
        method_type="mean",
        output_mode="append",
    )
    assert len(odf.columns) == 22


def test_auto_imputation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = auto_imputation(
        spark_session,
        df,
        list_of_cols=["education-num", "relationship", "race"],
        id_col="ifa",
    )
    assert len(odf.columns) == 18
    assert odf.where(F.col("education-num").isNull()).count() == 0
    assert odf.where(F.col("race").isNull()).count() == 0
    assert odf.where(F.col("relationship").isNull()).count() == 0
    assert odf.where(F.col("logfnl").isNull()).count() == 10207
    assert odf.where(F.col("education").isNull()).count() == 254


def test_auto_imputation_with_empty_list_of_cols(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = auto_imputation(spark_session, df, list_of_cols=[], id_col="ifa")
    assert odf.where(F.col("age").isNull()).count() == 30
    assert odf.where(F.col("fnlwgt").isNull()).count() == 8
    assert odf.where(F.col("race").isNull()).count() == 162
    assert odf.where(F.col("relationship").isNull()).count() == 4


def test_auto_imputation_with_appended_output(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = auto_imputation(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week", "relationship", "race"],
        id_col="ifa",
        output_mode="append",
    )
    assert len(odf.columns) == 21


# latent_features
def test_PCA_latentFeatures(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = PCA_latentFeatures(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "logfnl", "education-num", "hours-per-week"],
        explained_variance_cutoff=0.3,
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) < len(df.columns)
    assert len(odf.columns) == 13

    try:
        odf = PCA_latentFeatures(
            spark_session,
            df,
            list_of_cols=["education-num"],
            explained_variance_cutoff=0.3,
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = PCA_latentFeatures(
        spark_session, df, list_of_cols=[], explained_variance_cutoff=0.3
    )
    assert len(odf.columns) == len(df.columns)
    assert len(odf.columns) == 17

    odf = PCA_latentFeatures(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "logfnl", "education-num", "hours-per-week"],
        explained_variance_cutoff=0.3,
        output_mode="append",
    )
    assert len(odf.columns) > len(df.columns)
    assert len(odf.columns) == 18
    assert odf.where(F.col("education").isNull()).count() == 91
    assert odf.where(F.col("race").isNull()).count() == 58
    assert odf.where(F.col("latent_0").isNull()).count() == 0


def test_autoencoder_latentFeatures(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = autoencoder_latentFeatures(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "logfnl", "education-num", "hours-per-week"],
        epochs=20,
        reduction_params=0.5,
        model_path="unit_testing/models/",
    )
    if "arm64" not in platform.version().lower():
        assert len(odf.columns) < len(df.columns)
        assert len(odf.columns) == 14
    else:
        assert odf == df

    try:
        odf = autoencoder_latentFeatures(
            spark_session,
            df,
            list_of_cols=["education-num"],
            epochs=20,
            reduction_params=0.5,
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = autoencoder_latentFeatures(
        spark_session, df, list_of_cols=[], epochs=20, reduction_params=0.5
    )
    if "arm64" not in platform.version().lower():
        assert len(odf.columns) == len(df.columns)
        assert len(odf.columns) == 17
    else:
        assert odf == df

    odf = autoencoder_latentFeatures(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "logfnl", "education-num", "hours-per-week"],
        epochs=20,
        reduction_params=0.5,
        output_mode="append",
    )
    if "arm64" not in platform.version().lower():
        assert len(odf.columns) > len(df.columns)
        assert len(odf.columns) == 19
        assert odf.where(F.col("latent_0").isNull()).count() == 0
        assert odf.where(F.col("latent_1").isNull()).count() == 0
    else:
        assert odf == df


# feature_transformation
def test_feature_transformation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = feature_transformation(df, list_of_cols=["age", "fnlwgt", "hours-per-week"])
    assert len(odf.columns) == 17
    odf_pd = odf.where(F.col("ifa") == "27520a").toPandas()
    assert approx(odf_pd["age"][0]) == 7.14142842854285
    assert approx(odf_pd["fnlwgt"][0]) == 399.6936326738268
    assert approx(odf_pd["hours-per-week"][0]) == 4.47213595499958

    odf = feature_transformation(
        df, list_of_cols=["age", "fnlwgt", "hours-per-week"], output_mode="append"
    )
    assert len(odf.columns) == 20


def test_boxcox_transformation(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = boxcox_transformation(
        df, list_of_cols=["age", "fnlwgt", "hours-per-week"], boxcox_lambda=0.5
    )
    assert len(odf.columns) == 17
    odf_pd = odf.where(F.col("ifa") == "27520a").toPandas()
    assert approx(odf_pd["age"][0]) == 7.14142842854285
    assert approx(odf_pd["fnlwgt"][0]) == 399.6936326738268
    assert approx(odf_pd["hours-per-week"][0]) == 4.47213595499958

    odf = boxcox_transformation(
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        boxcox_lambda=0.5,
        output_mode="append",
    )
    assert len(odf.columns) == 20


def test_outlier_categories(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = outlier_categories(
        spark_session,
        df,
        list_of_cols=[
            "workclass",
            "education",
            "relationship",
            "race",
            "native-country",
        ],
        max_category=12,
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    assert odf.select("workclass").distinct().count() == 10
    assert odf.select("education").distinct().count() == 13
    assert odf.select("relationship").distinct().count() == 9
    assert odf.select("native-country").distinct().count() == 12
    assert odf.select("race").distinct().count() == 10
    assert odf.select("occupation").distinct().count() == 16
    assert odf.select("sex").distinct().count() == 4
    assert odf.select("marital-status").distinct().count() == 8

    try:
        odf = outlier_categories(
            spark_session,
            df,
            list_of_cols=["occupation"],
            max_category=12,
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = outlier_categories(spark_session, df, list_of_cols=[], max_category=12)
    assert (
        odf.select("workclass").distinct().count()
        == df.select("workclass").distinct().count()
    )
    assert (
        odf.select("education").distinct().count()
        == df.select("education").distinct().count()
    )
    assert (
        odf.select("relationship").distinct().count()
        == df.select("relationship").distinct().count()
    )
    assert (
        odf.select("native-country").distinct().count()
        == df.select("native-country").distinct().count()
    )
    assert odf.select("race").distinct().count() == df.select("race").distinct().count()
    assert (
        odf.select("occupation").distinct().count()
        == df.select("occupation").distinct().count()
    )
    assert odf.select("sex").distinct().count() == df.select("sex").distinct().count()
    assert (
        odf.select("marital-status").distinct().count()
        == df.select("marital-status").distinct().count()
    )

    odf = outlier_categories(
        spark_session,
        df,
        list_of_cols=[
            "workclass",
            "education",
            "relationship",
            "race",
            "native-country",
        ],
        max_category=12,
        output_mode="append",
    )
    assert len(odf.columns) == 22


# binning
def test_attribute_binning(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = attribute_binning(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        bin_size=20,
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["age"][0])) == 1
    assert round(float(odf_min_dict["fnlwgt"][0])) == 1
    assert round(float(odf_min_dict["hours-per-week"][0])) == 1
    assert round(float(odf_min_dict["logfnl"][0])) != 1
    assert round(float(odf_max_dict["age"][0])) == 20
    assert round(float(odf_max_dict["fnlwgt"][0])) == 20
    assert round(float(odf_max_dict["hours-per-week"][0])) == 20
    assert round(float(odf_max_dict["logfnl"][0])) != 20

    try:
        odf = attribute_binning(
            spark_session,
            df,
            list_of_cols=["education-num"],
            bin_size=20,
            pre_existing_model=True,
            model_path="unit_testing/models/",
        )
    except Exception as error:
        assert str(error) == "list index out of range"

    odf = attribute_binning(spark_session, df, list_of_cols=[], bin_size=20)
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    df_min_dict = (
        df.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    df_max_dict = (
        df.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["age"][0])) == round(float(df_min_dict["age"][0]))
    assert round(float(odf_min_dict["fnlwgt"][0])) == round(
        float(df_min_dict["fnlwgt"][0])
    )
    assert round(float(odf_max_dict["age"][0])) == round(float(df_max_dict["age"][0]))
    assert round(float(odf_max_dict["fnlwgt"][0])) == round(
        float(df_max_dict["fnlwgt"][0])
    )

    odf = attribute_binning(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        bin_size=20,
        output_mode="append",
    )
    assert len(odf.columns) == 20


def test_monotonic_binning(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = monotonic_binning(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        label_col="income",
        event_label="<=50K",
        bin_method="equal_range",
        bin_size=10,
    )
    assert len(odf.columns) == 17
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["age"][0])) == 1
    assert round(float(odf_min_dict["fnlwgt"][0])) == 1
    assert round(float(odf_min_dict["hours-per-week"][0])) == 1
    assert round(float(odf_min_dict["logfnl"][0])) != 1
    assert round(float(odf_max_dict["age"][0])) == 10
    assert round(float(odf_max_dict["fnlwgt"][0])) == 10
    assert round(float(odf_max_dict["hours-per-week"][0])) == 10
    assert round(float(odf_max_dict["logfnl"][0])) != 10

    odf = monotonic_binning(
        spark_session,
        df,
        list_of_cols=[],
        label_col="income",
        event_label="<=50K",
        bin_method="equal_range",
        bin_size=10,
    )
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    df_min_dict = (
        df.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    df_max_dict = (
        df.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["age"][0])) == round(float(df_min_dict["age"][0]))
    assert round(float(odf_min_dict["fnlwgt"][0])) == round(
        float(df_min_dict["fnlwgt"][0])
    )
    assert round(float(odf_max_dict["age"][0])) == round(float(df_max_dict["age"][0]))
    assert round(float(odf_max_dict["fnlwgt"][0])) == round(
        float(df_max_dict["fnlwgt"][0])
    )

    odf = monotonic_binning(
        spark_session,
        df,
        list_of_cols=["age", "fnlwgt", "hours-per-week"],
        label_col="income",
        event_label="<=50K",
        bin_method="equal_range",
        bin_size=10,
        output_mode="append",
    )
    assert len(odf.columns) == 20


# categorical_to_numerical_transformation
def test_cat_to_num_unsupervised(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = cat_to_num_unsupervised(
        spark_session,
        df,
        list_of_cols=["workclass", "relationship", "marital-status"],
        drop_cols=["ifa"],
        method_type="label_encoding",
        index_order="frequencyDesc",
        cardinality_threshold=100,
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["workclass"][0])) == 0
    assert round(float(odf_min_dict["marital-status"][0])) == 0
    assert round(float(odf_min_dict["relationship"][0])) == 0
    assert odf.select("workclass").dtypes[0][1] == "int"
    assert odf.select("marital-status").dtypes[0][1] == "int"
    assert odf.select("relationship").dtypes[0][1] == "int"
    assert odf.select("education").dtypes[0][1] == "string"

    odf = cat_to_num_unsupervised(
        spark_session,
        df,
        list_of_cols=[],
        drop_cols=["ifa"],
        method_type="label_encoding",
        index_order="frequencyDesc",
        cardinality_threshold=100,
    )
    assert odf.select("workclass").dtypes[0][1] == "string"
    assert odf.select("marital-status").dtypes[0][1] == "string"
    assert odf.select("relationship").dtypes[0][1] == "string"
    assert odf.select("education").dtypes[0][1] == "string"

    odf = cat_to_num_unsupervised(
        spark_session,
        df,
        list_of_cols=["workclass", "relationship", "marital-status"],
        drop_cols=["ifa"],
        method_type="label_encoding",
        index_order="frequencyDesc",
        cardinality_threshold=100,
        output_mode="append",
    )
    assert len(odf.columns) == 20

    odf = cat_to_num_unsupervised(
        spark_session,
        df,
        drop_cols=["ifa"],
        method_type="onehot_encoding",
        cardinality_threshold=100,
    )
    odf_min_dict = (
        odf.describe().where(F.col("summary") == "min").toPandas().to_dict("list")
    )
    odf_max_dict = (
        odf.describe().where(F.col("summary") == "max").toPandas().to_dict("list")
    )
    assert round(float(odf_min_dict["relationship_0"][0])) == 0
    assert round(float(odf_min_dict["race_7"][0])) == 0
    assert round(float(odf_min_dict["marital-status_1"][0])) == 0
    assert round(float(odf_min_dict["sex_1"][0])) == 0
    assert round(float(odf_min_dict["occupation_12"][0])) == 0
    assert round(float(odf_max_dict["relationship_0"][0])) == 1
    assert round(float(odf_max_dict["race_7"][0])) == 1
    assert round(float(odf_max_dict["marital-status_1"][0])) == 1
    assert round(float(odf_max_dict["sex_1"][0])) == 1
    assert round(float(odf_max_dict["occupation_12"][0])) == 1


def test_cat_to_num_supervised(spark_session):
    df = read_dataset(spark_session, sample_parquet, "parquet")
    odf = cat_to_num_supervised(
        spark_session,
        df,
        list_of_cols=["workclass", "relationship", "marital-status"],
        drop_cols=["ifa"],
        label_col="income",
        event_label="<=50K",
        model_path="unit_testing/models/",
    )
    assert len(odf.columns) == 17
    assert odf.select("workclass").dtypes[0][1] == "double"
    assert odf.select("marital-status").dtypes[0][1] == "double"
    assert odf.select("relationship").dtypes[0][1] == "double"
    assert odf.select("education").dtypes[0][1] == "string"
    df_workclass_private = (
        df.where(F.col("workclass") == "Private")
        .select("income")
        .toPandas()
        .value_counts()
    )
    assert round(
        odf.where(F.col("ifa") == "27520a").toPandas()["workclass"][0]
    ) == round(
        df_workclass_private[0] / (df_workclass_private[0] + df_workclass_private[1])
    )
    df_workclass_local_gov = (
        df.where(F.col("workclass") == "Local-gov")
        .select("income")
        .toPandas()
        .value_counts()
    )
    assert round(
        odf.where(F.col("ifa") == "6144a").toPandas()["workclass"][0]
    ) == round(
        df_workclass_local_gov[0]
        / (df_workclass_local_gov[0] + df_workclass_local_gov[1])
    )
    df_workclass_federal_gov = (
        df.where(F.col("workclass") == "Federal-gov")
        .select("income")
        .toPandas()
        .value_counts()
    )
    assert round(
        odf.where(F.col("ifa") == "23710a").toPandas()["workclass"][0]
    ) == round(
        df_workclass_federal_gov[0]
        / (df_workclass_federal_gov[0] + df_workclass_federal_gov[1])
    )

    odf = cat_to_num_supervised(
        spark_session,
        df,
        list_of_cols=[],
        drop_cols=["ifa"],
        label_col="income",
        event_label="<=50K",
    )
    assert odf.select("workclass").dtypes[0][1] == "string"
    assert odf.select("marital-status").dtypes[0][1] == "string"
    assert odf.select("relationship").dtypes[0][1] == "string"
    assert odf.select("education").dtypes[0][1] == "string"

    odf = cat_to_num_supervised(
        spark_session,
        df,
        list_of_cols=["workclass", "relationship", "marital-status"],
        drop_cols=["ifa"],
        label_col="income",
        event_label="<=50K",
        output_mode="append",
    )
    assert len(odf.columns) == 20
