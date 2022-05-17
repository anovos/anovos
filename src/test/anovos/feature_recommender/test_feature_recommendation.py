import pandas as pd
import pytest

from anovos.feature_recommender.featrec_init import model_download
from anovos.feature_recommender.feature_recommendation import (
    feature_recommendation,
    find_attr_by_relevance,
    sankey_visualization,
)

model_download()


@pytest.fixture
def example_attr_1():
    return pd.read_csv("./data/feature_recommender/test_input_fr.csv")


@pytest.fixture
def example_attr_2():
    return pd.read_csv("./data/feature_recommender/test_input_fr_2.csv")


def test_feature_recommendation(example_attr_1, example_attr_2):
    test_df = feature_recommendation(
        example_attr_1,
        name_column="Attribute Name",
        desc_column="Attribute Description",
    )
    assert len(test_df) > 0
    assert "Usecase" in test_df.columns
    assert "Industry" in test_df.columns
    assert "Recommended_Feature_Name" in test_df.columns
    assert "Recommended_Feature_Description" in test_df.columns
    assert "Input_Attribute_Name" in test_df.columns
    assert "Input_Attribute_Description" in test_df.columns
    assert "Feature_Similarity_Score" in test_df.columns
    assert "churn" in test_df.iloc[0, 0]
    assert "churn" in test_df.iloc[1, 0]
    assert "AccountWeeks" in test_df.iloc[2, 0]
    assert "ContractRenewal" in test_df.iloc[4, 0]
    for i in range(len(test_df)):
        assert "N/A" in test_df.iloc[i, 4] or float(test_df.iloc[i, 4]) >= 0.3
        assert float(test_df.iloc[i, 4]) <= 1

    test_df_2 = feature_recommendation(
        example_attr_2, desc_column="Desc", threshold=0.0
    )
    assert len(test_df_2) > 0
    assert "Usecase" in test_df_2.columns
    assert "Industry" in test_df_2.columns
    assert "Recommended_Feature_Name" in test_df_2.columns
    assert "Recommended_Feature_Description" in test_df_2.columns
    assert "Input_Attribute_Name" not in test_df_2.columns
    assert "Input_Attribute_Description" in test_df_2.columns
    assert "Feature_Similarity_Score" in test_df_2.columns
    assert "unique identifier" in test_df_2.iloc[0, 0]
    assert "unique identifier" in test_df_2.iloc[1, 0]
    assert "cost of each trip" in test_df_2.iloc[2, 0]
    assert "date and time" in test_df_2.iloc[4, 0]
    for i in range(len(test_df_2)):
        assert "N/A" not in test_df_2.iloc[i, 3]
        assert float(test_df_2.iloc[i, 3]) >= 0
        assert float(test_df_2.iloc[i, 3]) <= 1

    test_df_3 = feature_recommendation(
        example_attr_1,
        name_column="Attribute Name",
        desc_column="Attribute Description",
        suggested_industry="telecoms",
        suggested_usecase="churn prediction",
    )
    assert len(test_df_3) > 0
    assert "Usecase" in test_df_3.columns
    assert "Industry" in test_df_3.columns
    assert "Recommended_Feature_Name" in test_df_3.columns
    assert "Recommended_Feature_Description" in test_df_3.columns
    assert "Input_Attribute_Name" in test_df_3.columns
    assert "Input_Attribute_Description" in test_df_3.columns
    assert "Feature_Similarity_Score" in test_df_3.columns
    assert "churn" in test_df_3.iloc[0, 0]
    assert "churn" in test_df_3.iloc[1, 0]
    assert "AccountWeeks" in test_df_3.iloc[2, 0]
    assert "ContractRenewal" in test_df_3.iloc[4, 0]
    for i in range(len(test_df_3)):
        if "N/A" in test_df_3.iloc[i, 4]:
            assert "N/A" in test_df_3.iloc[i, 2]
            assert "N/A" in test_df_3.iloc[i, 3]
        else:
            assert "telecommunication" in test_df_3.iloc[i, 5]
            assert "churn" in test_df_3.iloc[i, 6]

    test_df_4 = feature_recommendation(
        example_attr_2, name_column="Name", threshold=0.0
    )
    assert len(test_df_4) > 0
    assert "Usecase" in test_df_4.columns
    assert "Industry" in test_df_4.columns
    assert "Recommended_Feature_Name" in test_df_4.columns
    assert "Recommended_Feature_Description" in test_df_4.columns
    assert "Input_Attribute_Name" in test_df_4.columns
    assert "Input_Attribute_Description" not in test_df_4.columns
    assert "Feature_Similarity_Score" in test_df_4.columns
    assert "key" in test_df_4.iloc[0, 0]
    assert "key" in test_df_4.iloc[1, 0]
    assert "fare_amount" in test_df_4.iloc[2, 0]
    assert "pickup_datetime" in test_df_4.iloc[4, 0]

    test_df_5 = feature_recommendation(
        example_attr_1,
        name_column="Attribute Name",
        desc_column="Attribute Description",
        suggested_industry="telecoms",
        suggested_usecase="churn prediction",
        semantic=False,
    )
    assert len(test_df_5) == 0


def test_find_attr_by_relevance(example_attr_1, example_attr_2):
    test_feature_corpus = [
        "number of customers using products",
        "number of call customer make daily",
    ]
    test_feature_corpus_2 = [
        "usual location of the customer",
        "how much customer spend for each trip on average",
    ]
    test_feature_corpus_3 = ["autogenerate303030", "autogenerate495979"]
    test_df = find_attr_by_relevance(
        example_attr_1,
        building_corpus=test_feature_corpus,
        name_column="Attribute Name",
        desc_column="Attribute Description",
    )
    assert len(test_df) > 0
    assert "Recommended_Input_Attribute_Name" in test_df.columns
    assert "Recommended_Input_Attribute_Description" in test_df.columns
    assert "Input_Feature_Description" in test_df.columns
    assert "Input_Attribute_Similarity_Score" in test_df.columns
    test_order_unique_list = test_df.iloc[:, 0].to_list()
    assert test_feature_corpus[0] == list(dict.fromkeys(test_order_unique_list))[0]
    assert test_feature_corpus[1] == list(dict.fromkeys(test_order_unique_list))[1]
    for i in range(len(test_df)):
        assert "N/A" in test_df.iloc[i, 3] or float(test_df.iloc[i, 3]) >= 0.3
        assert float(test_df.iloc[i, 3]) <= 1

    test_df_2 = find_attr_by_relevance(
        example_attr_2,
        building_corpus=test_feature_corpus_2,
        desc_column="Desc",
        threshold=0.25,
    )
    assert len(test_df_2) > 0
    assert "Recommended_Input_Attribute_Name" not in test_df_2.columns
    assert "Recommended_Input_Attribute_Description" in test_df_2.columns
    assert "Input_Feature_Description" in test_df_2.columns
    assert "Input_Attribute_Similarity_Score" in test_df_2.columns
    for i in range(len(test_df)):
        assert "N/A" in test_df.iloc[i, 3] or float(test_df.iloc[i, 3]) >= 0.25
        assert float(test_df.iloc[i, 3]) <= 1

    test_df_3 = find_attr_by_relevance(
        example_attr_1,
        building_corpus=test_feature_corpus_3,
        name_column="Attribute Name",
    )
    assert len(test_df_3) > 0
    assert "Recommended_Input_Attribute_Name" in test_df_3.columns
    assert "Recommended_Input_Attribute_Description" not in test_df_3.columns
    assert "Input_Feature_Description" in test_df_3.columns
    assert "Input_Attribute_Similarity_Score" in test_df_3.columns
    test_order_unique_list_3 = test_df_3.iloc[:, 0].to_list()
    assert test_feature_corpus_3[0] == list(dict.fromkeys(test_order_unique_list_3))[0]
    assert test_feature_corpus_3[1] == list(dict.fromkeys(test_order_unique_list_3))[1]
    for i in range(len(test_df_3)):
        if "N/A" in test_df_3.iloc[i, 2]:
            assert "N/A" in test_df_3.iloc[i, 1]


def test_sankey_visualization(example_attr_1, example_attr_2):
    test_feature_corpus = [
        "number of customers using products",
        "number of call customer make daily",
    ]
    test_df = feature_recommendation(
        example_attr_1,
        name_column="Attribute Name",
        desc_column="Attribute Description",
    )
    test_df_2 = feature_recommendation(
        example_attr_2,
        desc_column="Desc",
        suggested_industry="banking",
        suggested_usecase="credit risk",
    )
    test_df_3 = find_attr_by_relevance(
        example_attr_1,
        building_corpus=test_feature_corpus,
        name_column="Attribute Name",
        desc_column="Attribute Description",
    )

    sankey_plot = sankey_visualization(test_df)
    assert "churn" in sankey_plot.data[0]["node"]["label"]
    assert "AccountWeeks" in sankey_plot.data[0]["node"]["label"]
    assert "telecommunication" not in sankey_plot.data[0]["node"]["label"]
    assert (
        "banking financial service and insurance"
        not in sankey_plot.data[0]["node"]["label"]
    )

    sankey_plot_2 = sankey_visualization(
        test_df_2, industry_included=True, usecase_included=True
    )
    assert "the cost of each trip in usd" in sankey_plot_2.data[0]["node"]["label"]
    assert (
        "banking financial service and insurance"
        in sankey_plot_2.data[0]["node"]["label"]
    )
    assert "credit risk modeling" in sankey_plot_2.data[0]["node"]["label"]

    sankey_plot_3 = sankey_visualization(test_df_3)
    assert (
        "number of customers using products" in sankey_plot_3.data[0]["node"]["label"]
    )
    assert (
        "number of call customer make daily" in sankey_plot_3.data[0]["node"]["label"]
    )
