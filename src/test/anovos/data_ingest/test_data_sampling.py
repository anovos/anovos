import pytest

from anovos.data_ingest.data_ingest import read_dataset
from anovos.data_ingest.data_sampling import data_sample
from anovos.shared.spark import *


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
        strata_cols="all",
        method_type="stratified",
        fraction=0.75,
    )
    assert "gender" in output_1.columns
    assert "label_1" in output_1.columns
    assert "label_2" in output_1.columns
    assert "label_3" in output_1.columns
    assert output_1.filter(output_1["gender"] == "F").count() < 30
    assert output_1.filter(output_1["gender"] == "F").count() > 12
    assert output_1.filter(output_1["gender"] == "M").count() < 15
    assert output_1.filter(output_1["gender"] == "M").count() > 3

    output_2 = data_sample(
        test_sample,
        strata_cols="all",
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

    output_3 = data_sample(
        test_sample,
        strata_cols="all",
        method_type="random",
        fraction=0.5,
        seed_value=1,
    )
    assert "gender" in output_3.columns
    assert "label_1" in output_3.columns
    assert "label_2" in output_3.columns
    assert "label_3" in output_3.columns
    assert output_3.count() < 24
    assert output_3.count() > 12

    output_4 = data_sample(
        test_sample,
        strata_cols="all",
        method_type="stratified",
        stratified_type="balanced",
        fraction=0.75,
    )
    assert "gender" in output_4.columns
    assert "label_1" in output_4.columns
    assert "label_2" in output_4.columns
    assert "label_3" in output_4.columns
    assert output_4.filter(output_4["gender"] == "F").count() < 13
    assert output_4.filter(output_4["gender"] == "F").count() > 6
    assert output_4.filter(output_4["gender"] == "M").count() < 13
    assert output_4.filter(output_4["gender"] == "M").count() > 6
