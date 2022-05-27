import pandas as pd
import pytest

from anovos.feature_recommender.featrec_init import model_download
from anovos.feature_recommender.feature_exploration import (
    list_all_industry,
    list_all_pair,
    list_all_usecase,
    list_feature_by_industry,
    list_feature_by_pair,
    list_feature_by_usecase,
    list_industry_by_usecase,
    list_usecase_by_industry,
    process_industry,
    process_usecase,
)

model_download()


def test_list_all_industry():
    test_df = list_all_industry()
    assert len(test_df) > 0
    assert "Industry" in test_df.columns
    assert test_df.iloc[:, 0].nunique() == len(test_df)
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 0]
    assert "telecommunication" in test_df.iloc[:, 0].to_list()
    assert "healthcare" in test_df.iloc[:, 0].to_list()


def test_list_all_usecase():
    test_df = list_all_usecase()
    assert len(test_df) > 0
    assert "Usecase" in test_df.columns
    assert test_df.iloc[:, 0].nunique() == len(test_df)
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 0]
    assert "customer churn prediction" in test_df.iloc[:, 0].to_list()
    assert "fraud detection" in test_df.iloc[:, 0].to_list()


def test_list_all_pair():
    test_df = list_all_pair()
    assert len(test_df) > 0
    assert "Industry" in test_df.columns
    assert "Usecase" in test_df.columns
    assert test_df.groupby(["Industry", "Usecase"]).ngroups == len(test_df)
    for i in range(len(test_df)):
        if "telco-based credit scoring" in test_df.iloc[i, 1]:
            assert "telecommunication" in test_df.iloc[i, 0]
        if "stock price prediction" in test_df.iloc[i, 1]:
            assert "banking financial service and insurance" in test_df.iloc[i, 0]


def test_process_usecase():
    assert process_usecase("fraud", semantic=True) == "fraud detection"
    assert process_usecase("fraud", semantic=False) == "fraud"
    assert not process_usecase("churn", semantic=True) == "churn"
    assert not process_usecase("churn", semantic=False) == "autogenerate_random"


def test_process_industry():
    assert process_industry("telco", semantic=True) == "telecommunication"
    assert process_industry("telco", semantic=False) == "telco"
    assert not process_industry("bank", semantic=True) == "bank"
    assert not process_industry("bank", semantic=False) == "autogenerate_random"


def test_list_usecase_by_industry():
    test_df = list_usecase_by_industry("telecommunication")
    assert len(test_df) > 0
    assert "Usecase" in test_df.columns
    assert test_df.iloc[:, 0].nunique() == len(test_df)
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 0]

    test_df_2 = list_usecase_by_industry("health")
    assert len(test_df_2) > 0
    assert "Usecase" in test_df_2.columns
    assert test_df_2.iloc[:, 0].nunique() == len(test_df_2)
    for i in range(len(test_df_2)):
        assert "," not in test_df_2.iloc[i, 0]

    test_df_3 = list_usecase_by_industry("care", semantic=False)
    assert len(test_df_3) == 0


def test_list_industry_by_usecase():
    test_df = list_industry_by_usecase("customer churn prediction")
    assert len(test_df) > 0
    assert "Industry" in test_df.columns
    assert test_df.iloc[:, 0].nunique() == len(test_df)
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 0]

    test_df_2 = list_industry_by_usecase("fraud")
    assert len(test_df_2) > 0
    assert "Industry" in test_df_2.columns
    assert test_df_2.iloc[:, 0].nunique() == len(test_df_2)
    for i in range(len(test_df_2)):
        assert "," not in test_df_2.iloc[i, 0]

    test_df_3 = list_industry_by_usecase("prediction", semantic=False)
    assert len(test_df_3) == 0


def test_list_feature_by_industry():
    test_df = list_feature_by_industry("telecommunication", semantic=False)
    assert len(test_df) > 0
    assert "Usecase" in test_df.columns
    assert "Industry" in test_df.columns
    assert "Feature_Name" in test_df.columns
    assert "Feature_Description" in test_df.columns
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 2]
        assert "," not in test_df.iloc[i, 3]
        assert "telecommunication" in test_df.iloc[i, 2]

    test_df_2 = list_feature_by_industry("bank", num_of_feat=10)
    assert len(test_df_2) == 10
    assert "Usecase" in test_df_2.columns
    assert "Industry" in test_df_2.columns
    assert "Feature_Name" in test_df_2.columns
    assert "Feature_Description" in test_df_2.columns
    for i in range(len(test_df_2)):
        assert "," not in test_df_2.iloc[i, 2]
        assert "," not in test_df_2.iloc[i, 3]
        assert "banking" in test_df_2.iloc[i, 2]

    test_df_3 = list_feature_by_industry("game", semantic=False)
    assert len(test_df_3) == 0


def test_list_feature_by_usecase():
    test_df = list_feature_by_usecase("customer churn prediction", semantic=False)
    assert len(test_df) > 0
    assert "Usecase" in test_df.columns
    assert "Industry" in test_df.columns
    assert "Feature_Name" in test_df.columns
    assert "Feature_Description" in test_df.columns
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 2]
        assert "," not in test_df.iloc[i, 3]
        assert "customer churn prediction" in test_df.iloc[i, 3]

    test_df_2 = list_feature_by_usecase("affinity", num_of_feat="all")
    assert len(test_df_2) > 0
    assert "Usecase" in test_df_2.columns
    assert "Industry" in test_df_2.columns
    assert "Feature_Name" in test_df_2.columns
    assert "Feature_Description" in test_df_2.columns
    for i in range(len(test_df_2)):
        assert "," not in test_df_2.iloc[i, 2]
        assert "," not in test_df_2.iloc[i, 3]
        assert "brand" in test_df_2.iloc[i, 3]

    test_df_3 = list_feature_by_usecase("affinity", semantic=False)
    assert len(test_df_3) == 0


def test_list_feature_by_pair():
    test_df = list_feature_by_pair(
        "telecommunication", "customer churn prediction", semantic=False
    )
    assert len(test_df) > 0
    assert "Usecase" in test_df.columns
    assert "Industry" in test_df.columns
    assert "Feature_Name" in test_df.columns
    assert "Feature_Description" in test_df.columns
    for i in range(len(test_df)):
        assert "," not in test_df.iloc[i, 2]
        assert "," not in test_df.iloc[i, 3]
        assert "telecommunication" in test_df.iloc[i, 2]
        assert "customer churn prediction" in test_df.iloc[i, 3]

    test_df_2 = list_feature_by_pair("bank", "fraud detection", num_of_feat=10)
    assert len(test_df_2) == 10
    assert "Usecase" in test_df_2.columns
    assert "Industry" in test_df_2.columns
    assert "Feature_Name" in test_df_2.columns
    assert "Feature_Description" in test_df_2.columns
    for i in range(len(test_df_2)):
        assert "," not in test_df_2.iloc[i, 2]
        assert "," not in test_df_2.iloc[i, 3]
        assert "banking" in test_df_2.iloc[i, 2]
        assert "fraud" in test_df_2.iloc[i, 3]

    test_df_3 = list_feature_by_pair("sale", "affinity", semantic=False)
    assert len(test_df_3) == 0
