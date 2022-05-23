from pandas import util
from anovos.data_ingest.data_ingest import read_dataset, write_dataset
from pytest import raises


def test_that_csv_file_can_be_read(spark_session, tmp_path):

    df = util.testing.makeMixedDataFrame()

    file_path = tmp_path / "my_file.csv"

    df.to_csv(path_or_buf=str(file_path),
              sep=",",
              header=True,
              index=False)

    df = spark_session.createDataFrame(df)

    df_read = read_dataset(
        spark=spark_session,
        file_path=str(file_path),
        file_type="csv",
        file_configs={"header": True, "inferSchema": True, "delimiter": ","}
    )

    assert df.columns == df_read.columns
    assert df.count() == df_read.count()


def test_that_parquet_file_can_be_read(spark_session, tmp_path):

    df = util.testing.makeMixedDataFrame()

    file_path = tmp_path / "my_file.parquet"

    df.to_parquet(str(file_path), index=False)

    df_read = read_dataset(
        spark=spark_session,
        file_path=str(file_path),
        file_type="parquet"
    )

    assert df_read.count() == len(df)
    assert (df_read.columns == df.columns).all()


def test_that_csv_file_can_be_written(spark_session, tmp_path):

    df = spark_session.createDataFrame(util.testing.makeMixedDataFrame())

    file_path = tmp_path / "my_file.csv"

    write_dataset(idf=df,
                  file_path=str(file_path),
                  file_type="csv",
                  file_configs={"header": True,
                                "delimiter": ",",
                                "repartition": 5})

    df_written = spark_session.read.options(delimiter=",", header=True).csv(str(file_path))

    assert df.columns == df_written.columns


def test_that_parquet_file_can_be_written(spark_session, tmp_path):

    df = spark_session.createDataFrame(util.testing.makeMixedDataFrame())

    file_path = tmp_path / "my_file.parquet"

    write_dataset(idf=df,
                  file_path=str(file_path),
                  file_type="parquet")

    df_written = spark_session.read.option("header", "true").parquet(str(file_path))

    assert df.columns == df_written.columns


def test_that_negative_repartition_raises_exception(spark_session, tmp_path):

    df = spark_session.createDataFrame(util.testing.makeMixedDataFrame())

    file_path = tmp_path / "my_file.csv"

    with raises(Exception):

        write_dataset(idf=df,
                      file_path=str(file_path),
                      file_type="csv",
                      file_configs={"header": True,
                                    "delimiter": ",",
                                    "repartition": -2})
