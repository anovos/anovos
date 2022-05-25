import pandas as pd
import pytest

from anovos.feature_recommender.featrec_init import (
    camel_case_split,
    get_column_name,
    recommendation_data_prep,
)


@pytest.fixture
def example_attr_2():
    return pd.read_csv("./data/feature_recommender/test_input_fr_2.csv")


def test_get_column_name(example_attr_2):
    feature_name, feature_desc, industry, usecase = get_column_name(example_attr_2)
    assert feature_name == "Name"
    assert feature_desc == "Desc"
    assert industry == "Industry"
    assert usecase == "Usecase"


def test_camel_case_split():
    test_input_1 = "accountWeeks"
    test_output_1 = camel_case_split(test_input_1)
    assert test_output_1 == "account Weeks "

    test_input_2 = "account Weeks"
    test_output_2 = camel_case_split(test_input_2)
    assert test_output_2 == "account Weeks "

    test_input_3 = "AccountWeeksLock"
    test_output_3 = camel_case_split(test_input_3)
    assert test_output_3 == "Account Weeks Lock "


def test_recommendation_data_prep(example_attr_2):
    list_test, test_df = recommendation_data_prep(example_attr_2, "Name", "Desc")
    for i in range(len(list_test)):
        assert "_" not in list_test[i]

    for j in range(len(test_df)):
        assert "_" not in test_df.iloc[:, 0][j]
        assert test_df.iloc[:, 0][j].strip() == test_df.iloc[:, 0][j]
