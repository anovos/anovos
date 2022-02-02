from anovos.feature_recommender.featrec_init import *
from sentence_transformers import util
import numpy as np
import pandas as pd
import copy
import re


def list_allIndustry():
    odf_uni = df_input['Industry'].unique()
    odf = pd.DataFrame(odf_uni, columns=['Industry'])
    return odf


def list_allUsecase():
    odf_uni = df_input['Usecase'].unique()
    odf = pd.DataFrame(odf_uni, columns=['Usecase'])
    return odf


def list_allPair():
    odf = df_input[['Industry', 'Usecase']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def processUsecase(usecase, semantic):
    if type(semantic) != bool:
        raise TypeError('Invalid input for semantic')
    if type(usecase) != str:
        raise TypeError('Invalid input for usecase')
    usecase = usecase.lower().strip()
    usecase = usecase.replace("[^A-Za-z0-9 ]+", " ")
    all_usecase = list_allUsecase()['Usecase'].to_list()
    if semantic and usecase not in all_usecase:
        all_usecase_embeddings = model.encode(all_usecase, convert_to_tensor=True)
        usecase_embeddings = model.encode(usecase, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(usecase_embeddings, all_usecase_embeddings)[0]
        first_match_index = int(np.argpartition(-cos_scores, 0)[0])
        processed_usecase = all_usecase[first_match_index]
        print("Input Usecase not available. Showing closest Usecase semantic result for: ", processed_usecase)
    else:
        processed_usecase = usecase
    return processed_usecase


def processIndustry(industry, semantic):
    if type(semantic) != bool:
        raise TypeError('Invalid input for semantic')
    if type(industry) != str:
        raise TypeError('Invalid input for industry')
    industry = industry.lower().strip()
    industry = industry.replace("[^A-Za-z0-9 ]+", " ")
    all_industry = list_allIndustry()['Industry'].to_list()
    if semantic and industry not in all_industry:
        all_industry_embeddings = model.encode(all_industry, convert_to_tensor=True)
        industry_embeddings = model.encode(industry, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(industry_embeddings, all_industry_embeddings)[0]
        first_match_index = int(np.argpartition(-cos_scores, 0)[0])
        processed_industry = all_industry[first_match_index]
        print("Input Industry not available. Showing closest Industry semantic result for: ", processed_industry)
    else:
        processed_industry = industry
    return processed_industry


def list_usecasebyIndustry(industry, semantic=True):
    industry = processIndustry(industry, semantic)
    odf = df_input.loc[df_input['Industry'] == industry][['Usecase']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def list_industrybyUsecase(usecase, semantic=True):
    usecase = processUsecase(usecase, semantic)
    odf = df_input.loc[df_input['Usecase'] == usecase][['Industry']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def list_featurebyIndustry(industry, num_of_feat=100, semantic=True):
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    industry = processIndustry(industry, semantic)
    odf = df_input.loc[df_input['Industry'] == industry].drop_duplicates(keep='last', ignore_index=True)
    if len(odf) > 0:
        odf['count'] = odf.groupby('Usecase')['Usecase'].transform('count')
        odf.sort_values('count', inplace=True, ascending=False)
        odf = odf.drop('count', axis=1)
        odf = odf.head(num_of_feat).reset_index(drop=True)
    return odf


def list_featurebyUsecase(usecase, num_of_feat=100, semantic=True):
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    usecase = processUsecase(usecase, semantic)
    odf = df_input.loc[df_input['Usecase'] == usecase].drop_duplicates(keep='last', ignore_index=True)
    if len(odf) > 0:
        odf['count'] = odf.groupby('Industry')['Industry'].transform('count')
        odf.sort_values('count', inplace=True, ascending=False)
        odf = odf.drop('count', axis=1)
        odf = odf.head(num_of_feat).reset_index(drop=True)
    return odf


def list_featurebyPair(industry, usecase, num_of_feat=100, semantic=True):
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    industry = processIndustry(industry, semantic)
    usecase = processUsecase(usecase, semantic)
    odf = df_input.loc[(df_input['Industry'] == industry) & (df_input['Usecase'] == usecase)].drop_duplicates(
        keep='last', ignore_index=True).head(num_of_feat)
    return odf


def feature_recommendation(df, name_column=None, desc_column=None, suggested_industry='all', suggested_usecase='all',
                           semantic=True, top_n=2):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Invalid input for df')
    if type(top_n) != int or top_n < 0:
        raise TypeError('Invalid input for top_n')
    if top_n > len(list_train):
        raise TypeError('top_n value is too large')
    list_user, df_user = recommendationDataPrep(df, name_column, desc_column)
    if name_column == None:
        df_out = pd.DataFrame(columns=['Input Attribute Description', 'Recommended Feature Name',
                                       'Recommended Feature Description', 'Feature Similarity Score', 'Goal Industry',
                                       'Goal Usecase',
                                       'Source'])
    elif desc_column == None:
        df_out = pd.DataFrame(columns=['Input Attribute Name', 'Recommended Feature Name',
                                       'Recommended Feature Description', 'Feature Similarity Score', 'Goal Industry',
                                       'Goal Usecase',
                                       'Source'])
    else:
        df_out = pd.DataFrame(columns=['Input Attribute Name', 'Input Attribute Description', 'Recommended Feature Name',
                                       'Recommended Feature Description', 'Feature Similarity Score', 'Goal Industry',
                                       'Goal Usecase',
                                       'Source'])
    list_embedding_user = model.encode(list_user, convert_to_tensor=True)
    for i, feature in enumerate(list_user):
        cos_scores = util.pytorch_cos_sim(list_embedding_user, list_embedding_train)[i]
        top_results = np.argpartition(-cos_scores, range(top_n))[0:top_n]
        for idx in top_results[0:top_n]:
            if name_column == None:
                df_append = pd.DataFrame([[df_user[desc_column].iloc[i],
                                           df_rec['Feature Name'].iloc[int(idx)],
                                           df_rec['Feature Description'].iloc[int(idx)], "%.4f" % (cos_scores[idx]),
                                           df_rec['Industry'].iloc[int(idx)], df_rec['Usecase'].iloc[int(idx)],
                                           df_rec['Source'].iloc[int(idx)]]],
                                         columns=['Input Attribute Description', 'Recommended Feature Name',
                                                  'Recommended Feature Description', 'Feature Similarity Score',
                                                  'Goal Industry',
                                                  'Goal Usecase', 'Source'])
            elif desc_column == None:
                df_append = pd.DataFrame([[df_user[name_column].iloc[i],
                                           df_rec['Feature Name'].iloc[int(idx)],
                                           df_rec['Feature Description'].iloc[int(idx)], "%.4f" % (cos_scores[idx]),
                                           df_rec['Industry'].iloc[int(idx)], df_rec['Usecase'].iloc[int(idx)],
                                           df_rec['Source'].iloc[int(idx)]]],
                                         columns=['Input Attribute Name', 'Recommended Feature Name',
                                                  'Recommended Feature Description', 'Feature Similarity Score',
                                                  'Goal Industry',
                                                  'Goal Usecase', 'Source'])
            else:
                df_append = pd.DataFrame([[df_user[name_column].iloc[i], df_user[desc_column].iloc[i],
                                           df_rec['Feature Name'].iloc[int(idx)],
                                           df_rec['Feature Description'].iloc[int(idx)], "%.4f" % (cos_scores[idx]),
                                           df_rec['Industry'].iloc[int(idx)], df_rec['Usecase'].iloc[int(idx)],
                                           df_rec['Source'].iloc[int(idx)]]],
                                         columns=['Input Attribute Name', 'Input Attribute Description', 'Recommended Feature Name',
                                                  'Recommended Feature Description', 'Feature Similarity Score',
                                                  'Goal Industry',
                                                  'Goal Usecase', 'Source'])
            df_out = df_out.append(df_append, ignore_index=True)
    if suggested_industry != 'all':
        suggested_industry = processIndustry(suggested_industry, semantic)
        df_out = df_out[df_out['Goal Industry'].str.contains(suggested_industry)].reset_index(drop=True)
    if suggested_usecase != 'all':
        suggested_usecase = processUsecase(suggested_usecase, semantic)
        df_out = df_out[df_out['Goal Usecase'].str.contains(suggested_usecase)].reset_index(drop=True)
    return df_out


def fine_attrByRelevance(df, building_corpus, name_column=None, desc_column=None, threshold=0.4):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Invalid input for df')
    if type(building_corpus) != list:
        raise TypeError('Invalid input for building_corpus')
    for i in range(len(building_corpus)):
        if type(building_corpus[i]) != str:
            raise TypeError('Invalid input inside building_corpus:', building_corpus[i])
        building_corpus[i] = re.sub("[^A-Za-z0-9]+", " ", building_corpus[i])
        building_corpus[i] = camel_case_split(building_corpus[i])
        building_corpus[i] = building_corpus[i].lower().strip()
    if name_column == None:
        df_out = pd.DataFrame(columns=['Input Feature Desc',
                                       'Recommended Input Attribute Description', 'Input Attribute Similarity Score'])
    elif desc_column == None:
        df_out = pd.DataFrame(
            columns=['Input Feature Desc', 'Recommended Input Attribute Name', 'Input Attribute Similarity Score'])
    else:
        df_out = pd.DataFrame(columns=['Input Feature Desc', 'Recommended Input Attribute Name',
                                       'Recommended Input Attribute Description', 'Input Attribute Similarity Score'])
    list_user, df_user = recommendationDataPrep(df, name_column, desc_column)
    list_embedding_user = model.encode(list_user, convert_to_tensor=True)
    list_embedding_building = model.encode(building_corpus, convert_to_tensor=True)
    for i, feature in enumerate(building_corpus):
        if name_column == None:
            df_append = pd.DataFrame(columns=['Input Feature Desc',
                                              'Recommended Input Attribute Description', 'Input Attribute Similarity Score'])
        elif desc_column == None:
            df_append = pd.DataFrame(
                columns=['Input Feature Desc', 'Recommended Input Attribute Name', 'Input Attribute Similarity Score'])
        else:
            df_append = pd.DataFrame(columns=['Input Feature Desc', 'Recommended Input Attribute Name',
                                              'Recommended Input Attribute Description', 'Input Attribute Similarity Score'])
        cos_scores = util.pytorch_cos_sim(list_embedding_building, list_embedding_user)[i]
        top_results = np.argpartition(-cos_scores, range(len(list_user)))[0:len(list_user)]
        for idx in top_results[0:len(list_user)]:
            single_score = "%.4f" % (cos_scores[idx])
            if float(single_score) >= threshold:
                if name_column == None:
                    df_append = df_append.append(
                        {'Input Feature Desc': feature,
                         'Recommended Input Attribute Description': df_user[desc_column].iloc[int(idx)],
                         'Input Attribute Similarity Score': single_score}, ignore_index=True)
                elif desc_column == None:
                    df_append = df_append.append(
                        {'Input Feature Desc': feature,
                         'Recommended Input Attribute Name': df_user[name_column].iloc[int(idx)],
                         'Input Attribute Similarity Score': single_score}, ignore_index=True)
                else:
                    df_append = df_append.append(
                        {'Input Feature Desc': feature,
                         'Recommended Input Attribute Name': df_user[name_column].iloc[int(idx)],
                         'Recommended Input Attribute Description': df_user[desc_column].iloc[int(idx)],
                         'Input Attribute Similarity Score': single_score}, ignore_index=True)
        if len(df_append) == 0:
            if name_column == None:
                df_append = df_append.append(
                    {'Input Feature Desc': feature,
                     'Recommended Input Attribute Description': 'Null',
                     'Input Attribute Similarity Score': 'Null'}, ignore_index=True)
            elif desc_column == None:
                df_append = df_append.append(
                    {'Input Feature Desc': feature,
                     'Recommended Input Attribute Name': 'Null',
                     'Input Attribute Similarity Score': 'Null'}, ignore_index=True)
            else:
                df_append = df_append.append(
                    {'Input Feature Desc': feature, 'Recommended Input Attribute Name': 'Null',
                     'Recommended Input Attribute Description': 'Null',
                     'Input Attribute Similarity Score': 'Null'}, ignore_index=True)
        df_out = df_out.append(df_append, ignore_index=True)
    return df_out
