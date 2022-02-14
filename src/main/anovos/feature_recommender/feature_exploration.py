import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer('all-mpnet-base-v2')
input_path = 'https://raw.githubusercontent.com/anovos/anovos/feature_recommender_beta/data/feature_recommender/flatten_fr_db.csv'
df_input = pd.read_csv(input_path)


def list_all_industry():
    """
    :return: DataFrame
    """
    odf_uni = df_input['Industry'].unique()
    odf = pd.DataFrame(odf_uni, columns=['Industry'])
    return odf


def list_all_usecase():
    """
    :return: DataFrame
    """
    odf_uni = df_input['Usecase'].unique()
    odf = pd.DataFrame(odf_uni, columns=['Usecase'])
    return odf


def list_all_pair():
    """
    :return: DataFrame
    """
    odf = df_input[['Industry', 'Usecase']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def process_usecase(usecase, semantic):
    """
    :param usecase: Input usecase (string)
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: Processed Usecase(string)
    """
    if type(semantic) != bool:
        raise TypeError('Invalid input for semantic')
    if type(usecase) != str:
        raise TypeError('Invalid input for usecase')
    usecase = usecase.lower().strip()
    usecase = usecase.replace("[^A-Za-z0-9 ]+", " ")
    all_usecase = list_all_usecase()['Usecase'].to_list()
    if semantic and usecase not in all_usecase:
        all_usecase_embeddings = model.encode(all_usecase, convert_to_tensor=True)
        usecase_embeddings = model.encode(usecase, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(usecase_embeddings, all_usecase_embeddings)[0]
        first_match_index = int(np.argpartition(-cos_scores, 0)[0])
        processed_usecase = all_usecase[first_match_index]
        print("Input Usecase not available. Showing the most semantically relevant Usecase result: ", processed_usecase)
    else:
        processed_usecase = usecase
    return processed_usecase


def process_industry(industry, semantic):
    """
    :param industry: Input industry (string)
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: Processed Industry(string)
    """
    if type(semantic) != bool:
        raise TypeError('Invalid input for semantic')
    if type(industry) != str:
        raise TypeError('Invalid input for industry')
    industry = industry.lower().strip()
    industry = industry.replace("[^A-Za-z0-9 ]+", " ")
    all_industry = list_all_industry()['Industry'].to_list()
    if semantic and industry not in all_industry:
        all_industry_embeddings = model.encode(all_industry, convert_to_tensor=True)
        industry_embeddings = model.encode(industry, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(industry_embeddings, all_industry_embeddings)[0]
        first_match_index = int(np.argpartition(-cos_scores, 0)[0])
        processed_industry = all_industry[first_match_index]
        print("Input Industry not available. Showing the most semantically relevant Usecase result: ",
              processed_industry)
    else:
        processed_industry = industry
    return processed_industry


def list_usecase_by_industry(industry, semantic=True):
    """
    :param industry: Input industry (string)
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: DataFrame
    """
    industry = process_industry(industry, semantic)
    odf = df_input.loc[df_input['Industry'] == industry][['Usecase']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def list_industry_by_usecase(usecase, semantic=True):
    """
    :param usecase: Input usecase (string)
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: DataFrame
    """
    usecase = process_usecase(usecase, semantic)
    odf = df_input.loc[df_input['Usecase'] == usecase][['Industry']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def list_feature_by_industry(industry, num_of_feat=100, semantic=True):
    """
    :param industry: Input industry (string)
    :param num_of_feat: Number of features displayed (integer). Default is 100.
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: DataFrame
    """
    if type(num_of_feat) != int or num_of_feat < 0:
        if num_of_feat != 'all':
            raise TypeError('Invalid input for num_of_feat')
    industry = process_industry(industry, semantic)
    odf = df_input.loc[df_input['Industry'] == industry].drop_duplicates(keep='last', ignore_index=True)
    if len(odf) > 0:
        odf['count'] = odf.groupby('Usecase')['Usecase'].transform('count')
        odf.sort_values('count', inplace=True, ascending=False)
        odf = odf.drop('count', axis=1)
        if num_of_feat != 'all':
            odf = odf.head(num_of_feat).reset_index(drop=True)
        else:
            odf = odf.reset_index(drop=True)
    return odf


def list_feature_by_usecase(usecase, num_of_feat=100, semantic=True):
    """
    :param usecase: Input usecase (string)
    :param num_of_feat: Number of features displayed (integer). Default is 100.
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: DataFrame
    """
    if type(num_of_feat) != int or num_of_feat < 0:
        if num_of_feat != 'all':
            raise TypeError('Invalid input for num_of_feat')
    usecase = process_usecase(usecase, semantic)
    odf = df_input.loc[df_input['Usecase'] == usecase].drop_duplicates(keep='last', ignore_index=True)
    if len(odf) > 0:
        odf['count'] = odf.groupby('Industry')['Industry'].transform('count')
        odf.sort_values('count', inplace=True, ascending=False)
        odf = odf.drop('count', axis=1)
        if num_of_feat != 'all':
            odf = odf.head(num_of_feat).reset_index(drop=True)
        else:
            odf = odf.reset_index(drop=True)
    return odf


def list_feature_by_pair(industry, usecase, num_of_feat=100, semantic=True):
    """
    :param industry: Input industry (string)
    :param usecase: Input usecase (string)
    :param num_of_feat: Number of features displayed (integer). Default is 100.
    :param semantic: Input semantic (boolean) - Whether the input needs to go through semantic matching or not. Default is True.
    :return: DataFrame
    """
    if type(num_of_feat) != int or num_of_feat < 0:
        if num_of_feat != 'all':
            raise TypeError('Invalid input for num_of_feat')
    industry = process_industry(industry, semantic)
    usecase = process_usecase(usecase, semantic)
    if num_of_feat != 'all':
        odf = df_input.loc[(df_input['Industry'] == industry) & (df_input['Usecase'] == usecase)].drop_duplicates(
            keep='last', ignore_index=True).head(num_of_feat)
    else:
        odf = df_input.loc[(df_input['Industry'] == industry) & (df_input['Usecase'] == usecase)].drop_duplicates(
            keep='last', ignore_index=True)
    return odf
