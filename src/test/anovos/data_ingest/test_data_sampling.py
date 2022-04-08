from anovos.data_ingest.data_sampling import data_sample
from anovos.data_ingest.data_ingest import read_dataset
from anovos.shared.spark import *
import pytest


@pytest.fixture
def test_sample():
    df_test = read_dataset(
        spark,
        file_path="./data/data_sample/test_data_sample.csv",
        file_type="csv",
        file_configs={"header": "True", "delimiter": ",", "inferSchema": True},
    )
    return df_test


def test_data_sampling(test_sample):
    output_1 = data_sample(
        test_sample,
        list_of_cols="all",
        label_col="gender",
        method_type="stratified",
        fraction=0.25,
    )
    assert "gender" in output_1.columns
    assert "label_1" in output_1.columns
    assert "label_2" in output_1.columns
    assert "label_3" in output_1.columns
    assert output_1.filter(output_1["gender"] == "F").count() < 10
    assert output_1.filter(output_1["gender"] == "F").count() > 4
    assert output_1.filter(output_1["gender"] == "M").count() < 5
    assert output_1.filter(output_1["gender"] == "M").count() > 1

    output_2 = data_sample(
        test_sample,
        list_of_cols="all",
        label_col="gender",
        method_type="stratified",
        fraction=0.5,
        seed_value=1,
    )
    assert "gender" in output_2.columns
    assert "label_1" in output_2.columns
    assert "label_2" in output_2.columns
    assert "label_3" in output_2.columns
    assert output_2.filter(output_2["gender"] == "F").count() < 20
    assert output_2.filter(output_2["gender"] == "F").count() > 8
    assert output_2.filter(output_2["gender"] == "M").count() < 10
    assert output_2.filter(output_2["gender"] == "M").count() > 2
