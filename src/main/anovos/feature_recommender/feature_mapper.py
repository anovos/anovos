"""Feature mapper maps attributes to features based on ingested data dictionary by the user."""
import copy
import random
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import util

from anovos.feature_recommender.featrec_init import (
    EmbeddingsTrainFer,
    camel_case_split,
    feature_recommendation_prep,
    get_column_name,
    model_fer,
    recommendation_data_prep,
)
from anovos.feature_recommender.feature_explorer import (
    list_usecase_by_industry,
    process_industry,
    process_usecase,
)

list_train_fer, df_rec_fer = feature_recommendation_prep()
list_embedding_train_fer = EmbeddingsTrainFer(list_train_fer)
(
    feature_name_column,
    feature_desc_column,
    industry_column,
    usecase_column,
) = get_column_name(df_rec_fer)


def feature_mapper(
    df,
    name_column=None,
    desc_column=None,
    suggested_industry="all",
    suggested_usecase="all",
    semantic=True,
    top_n=2,
    threshold=0.3,
):
    """Matches features for users based on their input attributes, and their goal industry and/or use case

    Parameters
    ----------
    df : DataFrame
        Input DataFrame - Users' Data dictionary. It is expected to consist of attribute name and/or attribute description
    name_column : str
        Input, column name of Attribute Name in Input DataFrame. Default is None.
    desc_column : str
        Input, column name of Attribute Description in Input DataFrame. Default is None.
    suggested_industry : str
        Input, Industry of interest to the user (if any) to be filtered out. Default is 'all', meaning all Industries available.
    suggested_usecase : str
        Input, Usecase of interest to the user (if any) to be filtered out. Default is 'all', meaning all Usecases available.
    semantic : bool
        Input semantic - Whether the input needs to go through semantic similarity or not. Default is True.
    top_n : int
        Number of features displayed. Default is 2
    threshold : float
        Input threshold value. Default is 0.3

    Returns
    -------
    DataFrame
        Columns are:

        - Input Attribute Name: Name of the input Attribute
        - Input Attribute Description: Description of the input Attribute
        - Matched Feature Name: Name of the matched Feature
        - Matched Feature Description: Description of the matched Feature
        - Feature Similarity Score: Semantic similarity score between input Attribute and matched Feature
        - Industry: Industry name of the matched Feature
        - Usecase: Usecase name of the matched Feature

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Invalid input for df")
    if type(top_n) != int or top_n < 0:
        raise TypeError("Invalid input for top_n")
    if top_n > len(list_train_fer):
        raise TypeError("top_n value is too large")
    if type(threshold) != float:
        raise TypeError("Invalid input for threshold")
    if threshold < 0 or threshold > 1:
        raise TypeError(
            "Invalid input for threshold. Threshold value is between 0 and 1"
        )
    list_user, df_user = recommendation_data_prep(df, name_column, desc_column)

    if suggested_industry != "all" and suggested_industry == "all":
        suggested_industry = process_industry(suggested_industry, semantic)
        df_rec_fr = df_rec_fer[df_rec_fer.iloc[:, 2].str.contains(suggested_industry)]
        list_keep = list(df_rec_fr.index)
        list_embedding_train_fr = [
            list_embedding_train_fer.get.tolist()[x] for x in list_keep
        ]
        df_rec_fr = df_rec_fr.reset_index(drop=True)
    elif suggested_usecase != "all" and suggested_industry == "all":
        suggested_usecase = process_usecase(suggested_usecase, semantic)
        df_rec_fr = df_rec_fer[df_rec_fer.iloc[:, 3].str.contains(suggested_usecase)]
        list_keep = list(df_rec_fr.index)
        list_embedding_train_fr = [
            list_embedding_train_fer.get.tolist()[x] for x in list_keep
        ]
        df_rec_fr = df_rec_fr.reset_index(drop=True)
    elif suggested_usecase != "all" and suggested_industry != "all":
        suggested_industry = process_industry(suggested_industry, semantic)
        suggested_usecase = process_usecase(suggested_usecase, semantic)
        df_rec_fr = df_rec_fer[
            df_rec_fer.iloc[:, 2].str.contains(suggested_industry)
            & df_rec_fer.iloc[:, 3].str.contains(suggested_usecase)
        ]
        if len(df_rec_fr) > 0:
            list_keep = list(df_rec_fr.index)
            list_embedding_train_fr = [
                list_embedding_train_fer.get.tolist()[x] for x in list_keep
            ]
            df_rec_fr = df_rec_fr.reset_index(drop=True)
        else:
            df_out = pd.DataFrame(
                columns=[
                    "Input_Attribute_Name",
                    "Input_Attribute_Description",
                    "Matched_Feature_Name",
                    "Matched_Feature_Description",
                    "Feature_Similarity_Score",
                    "Industry",
                    "Usecase",
                ]
            )
            print("Industry/Usecase pair does not exist.")
            return df_out
    else:
        df_rec_fr = df_rec_fer
        list_embedding_train_fr = list_embedding_train_fer.get

    if name_column is None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Attribute_Description",
                "Matched_Feature_Name",
                "Matched_Feature_Description",
                "Feature_Similarity_Score",
                "Industry",
                "Usecase",
            ]
        )
    elif desc_column is None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Attribute_Name",
                "Matched_Feature_Name",
                "Matched_Feature_Description",
                "Feature_Similarity_Score",
                "Industry",
                "Usecase",
            ]
        )
    else:
        df_out = pd.DataFrame(
            columns=[
                "Input_Attribute_Name",
                "Input_Attribute_Description",
                "Matched_Feature_Name",
                "Matched_Feature_Description",
                "Feature_Similarity_Score",
                "Industry",
                "Usecase",
            ]
        )
    list_embedding_user = model_fer.model.encode(list_user, convert_to_tensor=True)
    for i, feature in enumerate(list_user):
        cos_scores = util.pytorch_cos_sim(list_embedding_user, list_embedding_train_fr)[
            i
        ]
        top_results = np.argpartition(-cos_scores, range(top_n))[0:top_n]
        for idx in top_results[0:top_n]:
            single_score = "%.4f" % (cos_scores[idx])
            if name_column is None:
                if float(single_score) >= threshold:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[desc_column].iloc[i],
                                df_rec_fr[feature_name_column].iloc[int(idx)],
                                df_rec_fr[feature_desc_column].iloc[int(idx)],
                                "%.4f" % (cos_scores[idx]),
                                df_rec_fr[industry_column].iloc[int(idx)],
                                df_rec_fr[usecase_column].iloc[int(idx)],
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Description",
                            "Matched_Feature_Name",
                            "Matched_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                        ],
                    )
                else:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[desc_column].iloc[i],
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Description",
                            "Matched_Feature_Name",
                            "Matched_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                        ],
                    )
            elif desc_column is None:
                if float(single_score) >= threshold:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[name_column].iloc[i],
                                df_rec_fr[feature_name_column].iloc[int(idx)],
                                df_rec_fr[feature_desc_column].iloc[int(idx)],
                                "%.4f" % (cos_scores[idx]),
                                df_rec_fr[industry_column].iloc[int(idx)],
                                df_rec_fr[usecase_column].iloc[int(idx)],
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Matched_Feature_Name",
                            "Matched_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                        ],
                    )
                else:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[name_column].iloc[i],
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Matched_Feature_Name",
                            "Matched_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                        ],
                    )
            else:
                if float(single_score) >= threshold:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[name_column].iloc[i],
                                df_user[desc_column].iloc[i],
                                df_rec_fr[feature_name_column].iloc[int(idx)],
                                df_rec_fr[feature_desc_column].iloc[int(idx)],
                                "%.4f" % (cos_scores[idx]),
                                df_rec_fr[industry_column].iloc[int(idx)],
                                df_rec_fr[usecase_column].iloc[int(idx)],
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Input_Attribute_Description",
                            "Matched_Feature_Name",
                            "Matched_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                        ],
                    )
                else:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[name_column].iloc[i],
                                df_user[desc_column].iloc[i],
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Input_Attribute_Description",
                            "Matched_Feature_Name",
                            "Matched_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                        ],
                    )
            df_out = pd.concat(
                [df_out, df_append], ignore_index=True, axis=0, join="outer"
            )
    return df_out


def find_attr_by_relevance(
    df, building_corpus, name_column=None, desc_column=None, threshold=0.3
):
    """Provide a comprehensive mapping method from users&#39; input attributes to their own feature corpus,
     and therefore, help with the process of creating features in cold-start problems


    Parameters
    ----------
    df : DataFrame
        Input DataFrame - Users' Data dictionary. It is expected to consist of attribute name and/or attribute description
    building_corpus : list
        Input Feature Description
    name_column : str
        Input, column name of Attribute Name in Input DataFrame. Default is None.
    desc_column : str
        Input, column name of Attribute Description in Input DataFrame. Default is None.
    threshold : float
        Input threshold value Default is 0.3

    Returns
    -------
    DataFrame
        Columns are:

        - Input Feature Desc: Description of the input Feature
        - Recommended Input Attribute Name: Name of the recommended Feature
        - Recommended Input Attribute Description: Description of the recommended Feature
        - Input Attribute Similarity Score: Semantic similarity score between input Attribute and recommended Feature


    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Invalid input for df")
    if type(building_corpus) != list:
        raise TypeError("Invalid input for building_corpus")
    if type(threshold) != float:
        raise TypeError("Invalid input for building_corpus")
    if threshold < 0 or threshold > 1:
        raise TypeError(
            "Invalid input for threshold. Threshold value is between 0 and 1"
        )
    for i in range(len(building_corpus)):
        if type(building_corpus[i]) != str:
            raise TypeError("Invalid input inside building_corpus:", building_corpus[i])
        building_corpus[i] = re.sub("[^A-Za-z0-9]+", " ", building_corpus[i])
        building_corpus[i] = camel_case_split(building_corpus[i])
        building_corpus[i] = building_corpus[i].lower().strip()
    if name_column is None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Feature_Description",
                "Recommended_Input_Attribute_Description",
                "Input_Attribute_Similarity_Score",
            ]
        )
    elif desc_column is None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Feature_Description",
                "Recommended_Input_Attribute_Name",
                "Input_Attribute_Similarity_Score",
            ]
        )
    else:
        df_out = pd.DataFrame(
            columns=[
                "Input_Feature_Description",
                "Recommended_Input_Attribute_Name",
                "Recommended_Input_Attribute_Description",
                "Input_Attribute_Similarity_Score",
            ]
        )
    list_user, df_user = recommendation_data_prep(df, name_column, desc_column)
    list_embedding_user = model_fer.model.encode(list_user, convert_to_tensor=True)
    list_embedding_building = model_fer.model.encode(
        building_corpus, convert_to_tensor=True
    )
    for i, feature in enumerate(building_corpus):
        if name_column is None:
            df_append = pd.DataFrame(
                columns=[
                    "Input_Feature_Description",
                    "Recommended_Input_Attribute_Description",
                    "Input_Attribute_Similarity_Score",
                ]
            )
        elif desc_column is None:
            df_append = pd.DataFrame(
                columns=[
                    "Input_Feature_Description",
                    "Recommended_Input_Attribute_Name",
                    "Input_Attribute_Similarity_Score",
                ]
            )
        else:
            df_append = pd.DataFrame(
                columns=[
                    "Input_Feature_Description",
                    "Recommended_Input_Attribute_Name",
                    "Recommended_Input_Attribute_Description",
                    "Input_Attribute_Similarity_Score",
                ]
            )
        cos_scores = util.pytorch_cos_sim(list_embedding_building, list_embedding_user)[
            i
        ]
        top_results = np.argpartition(-cos_scores, range(len(list_user)))[
            0 : len(list_user)
        ]
        for idx in top_results[0 : len(list_user)]:
            single_score = "%.4f" % (cos_scores[idx])
            if float(single_score) >= threshold:
                if name_column is None:
                    df_append.loc[len(df_append.index)] = [
                        feature,
                        df_user[desc_column].iloc[int(idx)],
                        single_score,
                    ]
                elif desc_column is None:
                    df_append.loc[len(df_append.index)] = [
                        feature,
                        df_user[name_column].iloc[int(idx)],
                        single_score,
                    ]
                else:
                    df_append.loc[len(df_append.index)] = [
                        feature,
                        df_user[name_column].iloc[int(idx)],
                        df_user[desc_column].iloc[int(idx)],
                        single_score,
                    ]
        if len(df_append) == 0:
            if name_column is None:
                df_append.loc[len(df_append.index)] = [feature, "N/A", "N/A"]
            elif desc_column is None:
                df_append.loc[len(df_append.index)] = [feature, "N/A", "N/A"]
            else:
                df_append.loc[len(df_append.index)] = [feature, "N/A", "N/A", "N/A"]
        df_out = pd.concat([df_out, df_append], ignore_index=True, axis=0, join="outer")
    return df_out


def sankey_visualization(df, industry_included=False, usecase_included=False):
    """Visualize Feature Mapper functions through Sankey plots

    Parameters
    ----------
    df : DataFrame
        Input DataFrame. This DataFrame needs to be output of feature_mapper or find_attr_by_relevance, or in the same format.
    industry_included : bool
        Whether the plot needs to include industry mapping or not. Default is False
    usecase_included : bool
        Whether the plot needs to include usecase mapping or not. Default is False

    Returns
    -------
    A `plotly` graph object.


    """
    fr_proper_col_list = [
        "Matched_Feature_Name",
        "Matched_Feature_Description",
        "Feature_Similarity_Score",
        "Industry",
        "Usecase",
    ]
    attr_proper_col_list = [
        "Input_Feature_Description",
        "Input_Attribute_Similarity_Score",
    ]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Invalid input for df")
    if not all(x in list(df.columns) for x in fr_proper_col_list) and not all(
        x in list(df.columns) for x in attr_proper_col_list
    ):
        raise TypeError(
            "df is not output DataFrame of Feature Recommendation functions"
        )
    if type(industry_included) != bool:
        raise TypeError("Invalid input for industry_included")
    if type(usecase_included) != bool:
        raise TypeError("Invalid input for usecase_included")
    if "Feature_Similarity_Score" in df.columns:
        if "Input_Attribute_Name" in df.columns:
            name_source = "Input_Attribute_Name"
        else:
            name_source = "Input_Attribute_Description"
        name_target = "Matched_Feature_Name"
        name_score = "Feature_Similarity_Score"
    else:
        name_source = "Input_Feature_Description"
        if "Recommended_Input_Attribute_Name" in df.columns:
            name_target = "Recommended_Input_Attribute_Name"
        else:
            name_target = "Recommended_Input_Attribute_Description"
        name_score = "Input_Attribute_Similarity_Score"
        if industry_included or usecase_included:
            print(
                "Input is find_attr_by_relevance output DataFrame. There is no suggested Industry and/or Usecase."
            )
        industry_included = False
        usecase_included = False
    industry_target = "Industry"
    usecase_target = "Usecase"
    df_iter = copy.deepcopy(df)
    for i in range(len(df_iter)):
        if str(df_iter[name_score][i]) == "N/A":
            df = df.drop([i])
    df = df.reset_index(drop=True)
    source = []
    target = []
    value = []
    if not industry_included and not usecase_included:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        label = source_list + target_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
    elif not industry_included and usecase_included:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        raw_usecase_list = df[usecase_target].unique().tolist()
        usecase_list = []
        for i, item in enumerate(raw_usecase_list):
            if ", " in raw_usecase_list[i]:
                raw_usecase_list[i] = raw_usecase_list[i].split(", ")
                for j, sub_item in enumerate(raw_usecase_list[i]):
                    usecase_list.append(sub_item)
            else:
                usecase_list.append(item)
        label = source_list + target_list + usecase_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
            temp_list = df[usecase_target][i].split(", ")
            for k, item in enumerate(temp_list):
                source.append(label.index(str(df[name_target][i])))
                target.append(label.index(str(item)))
                value.append(float(1))
    elif industry_included and not usecase_included:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        raw_industry_list = df[industry_target].unique().tolist()
        industry_list = []
        for i, item in enumerate(raw_industry_list):
            if ", " in raw_industry_list[i]:
                raw_industry_list[i] = raw_industry_list[i].split(", ")
                for j, sub_item in enumerate(raw_industry_list[i]):
                    industry_list.append(sub_item)
            else:
                industry_list.append(item)
        label = source_list + target_list + industry_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
            temp_list = df[industry_target][i].split(", ")
            for k, item in enumerate(temp_list):
                source.append(label.index(str(df[name_target][i])))
                target.append(label.index(str(item)))
                value.append(float(1))
    else:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        raw_industry_list = df[industry_target].unique().tolist()
        raw_usecase_list = df[usecase_target].unique().tolist()
        industry_list = []
        for i, item in enumerate(raw_industry_list):
            if ", " in raw_industry_list[i]:
                raw_industry_list[i] = raw_industry_list[i].split(", ")
                for j, sub_item in enumerate(raw_industry_list[i]):
                    industry_list.append(sub_item)
            else:
                industry_list.append(item)

        usecase_list = []
        for i, item in enumerate(raw_usecase_list):
            if ", " in raw_usecase_list[i]:
                raw_usecase_list[i] = raw_usecase_list[i].split(", ")
                for j, sub_item in enumerate(raw_usecase_list[i]):
                    usecase_list.append(sub_item)
            else:
                usecase_list.append(item)

        label = source_list + target_list + industry_list + usecase_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
            temp_list_industry = df[industry_target][i].split(", ")
            temp_list_usecase = df[usecase_target][i].split(", ")
            for k, item_industry in enumerate(temp_list_industry):
                source.append(label.index(str(df[name_target][i])))
                target.append(label.index(str(item_industry)))
                value.append(float(1))
                for j, item_usecase in enumerate(temp_list_usecase):
                    if (
                        item_usecase
                        in list_usecase_by_industry(item_industry)[
                            usecase_column
                        ].tolist()
                    ):
                        source.append(label.index(str(item_industry)))
                        target.append(label.index(str(item_usecase)))
                        value.append(float(1))
    line_color = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for k in range(len(value))
    ]
    label_color = [
        "#" + "".join([random.choice("0123456789ABCDEF") for e in range(6)])
        for f in range(len(label))
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color=line_color, width=0.5),
                    label=label,
                    color=label_color,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(title_text="Feature Mapper Sankey Visualization", font_size=10)
    return fig
