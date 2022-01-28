from anovos.feature_recommender.featrec_init import *
from sentence_transformers import util
import numpy as np
import pandas as pd
import copy


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


def list_featurebyIndustry(industry, num_of_feat=100, mw_avail='all', semantic=True):
    if mw_avail != 'all' and mw_avail != 'yes':
        raise TypeError('Invalid input for mw_avail')
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    industry = processIndustry(industry, semantic)
    if mw_avail == 'yes':
        odf = df_input.loc[df_input['Industry'] == industry].loc[(df_input['Available in Mobilewalla'] == 'fm') | (
                df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                   ignore_index=True)
        if len(odf) > 0:
            odf['count'] = odf.groupby('Usecase')['Usecase'].transform('count')
            odf.sort_values('count', inplace=True, ascending=False)
            odf = odf.drop('count', axis=1)
            odf = odf.head(num_of_feat)
    else:
        odf = df_input.loc[df_input['Industry'] == industry].drop_duplicates(keep='last', ignore_index=True)
        if len(odf) > 0:
            odf['count'] = odf.groupby('Usecase')['Usecase'].transform('count')
            odf.sort_values('count', inplace=True, ascending=False)
            odf = odf.drop('count', axis=1)
            odf = odf.head(num_of_feat)
    return odf


def list_featurebyUsecase(usecase, num_of_feat=100, mw_avail='all', semantic=True):
    if mw_avail != 'all' and mw_avail != 'yes':
        raise TypeError('Invalid input for mw_avail')
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    usecase = processUsecase(usecase, semantic)
    if mw_avail == 'yes':
        odf = df_input.loc[df_input['Usecase'] == usecase].loc[(df_input['Available in Mobilewalla'] == 'fm') | (
                df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                   ignore_index=True)
        if len(odf) > 0:
            odf['count'] = odf.groupby('Industry')['Industry'].transform('count')
            odf.sort_values('count', inplace=True, ascending=False)
            odf = odf.drop('count', axis=1)
            odf = odf.head(num_of_feat)
    else:
        odf = df_input.loc[df_input['Usecase'] == usecase].drop_duplicates(keep='last', ignore_index=True)
        if len(odf) > 0:
            odf['count'] = odf.groupby('Industry')['Industry'].transform('count')
            odf.sort_values('count', inplace=True, ascending=False)
            odf = odf.drop('count', axis=1)
            odf = odf.head(num_of_feat)
    return odf


def list_featurebyPair(industry, usecase, num_of_feat=100, mw_avail='all', semantic=True):
    if mw_avail != 'all' and mw_avail != 'yes':
        raise TypeError('Invalid input for mw_avail')
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    industry = processIndustry(industry, semantic)
    usecase = processUsecase(usecase, semantic)
    if mw_avail == 'yes':
        odf = df_input.loc[(df_input['Industry'] == industry) & (df_input['Usecase'] == usecase)].loc[
            (df_input['Available in Mobilewalla'] == 'fm') | (
                    df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                       ignore_index=True).head(
            num_of_feat)
    else:
        odf = df_input.loc[(df_input['Industry'] == industry) & (df_input['Usecase'] == usecase)].drop_duplicates(
            keep='last', ignore_index=True).head(num_of_feat)
    return odf


def featureAvailable(num_of_feat=100):
    if type(num_of_feat) != int or num_of_feat < 0:
        raise TypeError('Invalid input for num_of_feat')
    odf = df_input.loc[
        (df_input['Available in Mobilewalla'] == 'fm') | (
                df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                   ignore_index=True).head(
        num_of_feat)
    return odf


def featureRecommendation(df, name_column, desc_column, top_n):
    df_out = pd.DataFrame(columns=['Attribute Name', 'Attribute Description', 'Recommended Feature Name',
                                   'Recommended Feature Description', 'Feature Similarity Score', 'Industry', 'Usecase'])
    list_user, df_user = recommendationDataPrep(df, name_column, desc_column)
    list_embedding_train = model.encode(list_train, convert_to_tensor=True)
    list_embedding_user = model.encode(list_user, convert_to_tensor=True)
    for i, feature in enumerate(list_user):
        cos_scores = util.pytorch_cos_sim(list_embedding_user, list_embedding_train)[i]
        top_results = np.argpartition(-cos_scores, range(top_n))[0:top_n]
        for idx in top_results[0:top_n]:
            df_append = pd.DataFrame([[df_user[name_column].iloc[i], df_user[desc_column].iloc[i],
                                       df_rec['Feature Name'].iloc[int(idx)],
                                       df_rec['Feature Description'].iloc[int(idx)], "%.4f" % (cos_scores[idx]),
                                       df_rec['Industry'].iloc[int(idx)], df_rec['Usecase'].iloc[int(idx)]]],
                                     columns=['Attribute Name', 'Attribute Description', 'Recommended Feature Name',
                                              'Recommended Feature Description', 'Feature Similarity Score', 'Industry', 'Usecase'])
            df_out = df_out.append(df_append, ignore_index=True)
    return df_out
