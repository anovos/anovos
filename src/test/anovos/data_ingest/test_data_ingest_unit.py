import csv

from anovos.data_ingest.data_ingest import read_dataset


def test_that_csv_file_can_be_read(spark_session, tmp_path):
    # GIVEN
    lines = [["columnA", "columnB", "columnC"], [4, 5, "name"], [2, 4.5, "line"]]

    file_path = tmp_path / "my_file.csv"

    with open(file_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerows(lines)

    # WHEN
    df = read_dataset(
        spark=spark_session,
        file_path=file_path,
        file_type="csv",
        file_configs={"header": True, "inferSchema": True, "delimiter": ","},
    )

    # THEN
    assert df.count() == 2
    assert df.columns == lines[0]
