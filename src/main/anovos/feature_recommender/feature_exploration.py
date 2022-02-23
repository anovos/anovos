from anovos.feature_recommender.featrec_init import *
from sentence_transformers import util
import pandas as pd
import numpy as np

model_fer = init_model()
df_input_fer = feature_exploration_prep()
feature_name_column, feature_desc_column, industry_column, usecase_column, source_column = get_column_name(df_input_fer)


def list_all_industry():
    """

    Returns
    -------
    DataFrame of all the supported industries as part of feature exploration/recommendation
    """
    odf_uni = df_input_fer.iloc[:, 2].unique()
    odf = pd.DataFrame(odf_uni, columns=["Industry"])
    return odf


def list_all_usecase():
    """

    Returns
    -------
    DataFrame of all the supported usecases as part of feature exploration/recommendation
    """
    odf_uni = df_input_fer.iloc[:, 3].unique()
    odf = pd.DataFrame(odf_uni, columns=["Usecase"])
    return odf


def list_all_pair():
    """
    Returns
    -------
    DataFrame of all the supported Industry/Usecase pairs as part of feature exploration/recommendation
    """
    odf = df_input_fer.iloc[:, [2, 3]].drop_duplicates(keep="last", ignore_index=True)
    return odf


def process_usecase(usecase: str, semantic: bool):
    """

    Parameters
    ----------
    usecase : str
        Input usecase
    semantic : bool
        Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    if type(semantic) != bool:
        raise TypeError("Invalid input for semantic")
    if type(usecase) != str:
        raise TypeError("Invalid input for usecase")
    usecase = usecase.lower().strip()
    usecase = usecase.replace("[^A-Za-z0-9 ]+", " ")
    all_usecase = list_all_usecase()["Usecase"].to_list()
    if semantic and usecase not in all_usecase:
        all_usecase_embeddings = model_fer.encode(all_usecase, convert_to_tensor=True)
        usecase_embeddings = model_fer.encode(usecase, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(usecase_embeddings, all_usecase_embeddings)[0]
        first_match_index = int(np.argpartition(-cos_scores, 0)[0])
        processed_usecase = all_usecase[first_match_index]
        print(
            "Given input Usecase is not available. Showing the most semantically relevant Usecase result: ",
            processed_usecase,
        )
    else:
        processed_usecase = usecase
    return processed_usecase


def process_industry(industry: str, semantic: bool):
    """

    Parameters
    ----------
    industry : str
        Input industry
    semantic : bool
        Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    if type(semantic) != bool:
        raise TypeError("Invalid input for semantic")
    if type(industry) != str:
        raise TypeError("Invalid input for industry")
    industry = industry.lower().strip()
    industry = industry.replace("[^A-Za-z0-9 ]+", " ")
    all_industry = list_all_industry()["Industry"].to_list()
    if semantic and industry not in all_industry:
        all_industry_embeddings = model_fer.encode(all_industry, convert_to_tensor=True)
        industry_embeddings = model_fer.encode(industry, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(industry_embeddings, all_industry_embeddings)[
            0
        ]
        first_match_index = int(np.argpartition(-cos_scores, 0)[0])
        processed_industry = all_industry[first_match_index]
        print(
            "Given input Industry is not available. Showing the most semantically relevant Industry result: ",
            processed_industry,
        )
    else:
        processed_industry = industry
    return processed_industry


def list_usecase_by_industry(industry, semantic=True):
    """

    Parameters
    ----------
    industry
        Input industry (string)
    semantic
        Input semantic (boolean) - Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    industry = process_industry(industry, semantic)
    odf = pd.DataFrame(df_input_fer.loc[df_input_fer.iloc[:, 2] == industry].iloc[:, 3])
    odf = odf.drop_duplicates(keep="last", ignore_index=True)
    return odf


def list_industry_by_usecase(usecase, semantic=True):
    """

    Parameters
    ----------
    usecase
        Input usecase (string)
    semantic
        Input semantic (boolean) - Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    usecase = process_usecase(usecase, semantic)
    odf = pd.DataFrame(df_input_fer.loc[df_input_fer.iloc[:, 3] == usecase].iloc[:, 2])
    odf = odf.drop_duplicates(keep="last", ignore_index=True)
    return odf


def list_feature_by_industry(industry, num_of_feat=100, semantic=True):
    """

    Parameters
    ----------
    industry
        Input industry (string)
    num_of_feat
        Number of features to be displayed in the output.
        Value can be either integer, or 'all' - display all features matched with the input. Default is 100.
    semantic
        Input semantic (boolean) - Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    if type(num_of_feat) != int or num_of_feat < 0:
        if num_of_feat != "all":
            raise TypeError("Invalid input for num_of_feat")
    industry = process_industry(industry, semantic)
    odf = df_input_fer.loc[df_input_fer.iloc[:, 2] == industry].drop_duplicates(
        keep="last", ignore_index=True
    )
    if len(odf) > 0:
        odf["count"] = odf.groupby(usecase_column)[usecase_column].transform("count")
        odf.sort_values("count", inplace=True, ascending=False)
        odf = odf.drop("count", axis=1)
        if num_of_feat != "all":
            odf = odf.head(num_of_feat).reset_index(drop=True)
        else:
            odf = odf.reset_index(drop=True)
    return odf


def list_feature_by_usecase(usecase, num_of_feat=100, semantic=True):
    """

    Parameters
    ----------
    usecase
        Input usecase (string)
    num_of_feat
        Number of features to be displayed in the output.
        Value can be either integer, or 'all' - display all features matched with the input.  Default is 100.
    semantic
        Input semantic (boolean) - Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    if type(num_of_feat) != int or num_of_feat < 0:
        if num_of_feat != "all":
            raise TypeError("Invalid input for num_of_feat")
    usecase = process_usecase(usecase, semantic)
    odf = df_input_fer.loc[df_input_fer.iloc[:, 3] == usecase].drop_duplicates(
        keep="last", ignore_index=True
    )
    if len(odf) > 0:
        odf["count"] = odf.groupby(industry_column)[industry_column].transform("count")
        odf.sort_values("count", inplace=True, ascending=False)
        odf = odf.drop("count", axis=1)
        if num_of_feat != "all":
            odf = odf.head(num_of_feat).reset_index(drop=True)
        else:
            odf = odf.reset_index(drop=True)
    return odf


def list_feature_by_pair(industry, usecase, num_of_feat=100, semantic=True):
    """

    Parameters
    ----------
    industry
        Input industry (string)
    usecase
        Input usecase (string)
    num_of_feat
        Number of features to be displayed in the output.
        Value can be either integer, or 'all' - display all features matched with the input.  Default is 100.
    semantic
        Input semantic (boolean) - Whether the input needs to go through semantic similarity or not. Default is True.

    Returns
    -------

    """
    if type(num_of_feat) != int or num_of_feat < 0:
        if num_of_feat != "all":
            raise TypeError("Invalid input for num_of_feat")
    industry = process_industry(industry, semantic)
    usecase = process_usecase(usecase, semantic)
    if num_of_feat != "all":
        odf = (
            df_input_fer.loc[
                (df_input_fer.iloc[:, 2] == industry)
                & (df_input_fer.iloc[:, 3] == usecase)
            ]
            .drop_duplicates(keep="last", ignore_index=True)
            .head(num_of_feat)
        )
    else:
        odf = df_input_fer.loc[
            (df_input_fer.iloc[:, 2] == industry) & (df_input_fer.iloc[:, 3] == usecase)
        ].drop_duplicates(keep="last", ignore_index=True)
    return odf
