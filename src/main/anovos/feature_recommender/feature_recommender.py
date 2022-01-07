from anovos.feature_recommender.featrec_init import *
import pandas as pd


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


def list_usecasebyIndustry(industry):
    industry = industry.lower().strip()
    odf = df_input.loc[df_input['Industry'] == industry][['Usecase']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def list_industrybyUsecase(usecase):
    usecase = usecase.lower().strip()
    odf = df_input.loc[df_input['Usecase'] == usecase][['Industry']].drop_duplicates(keep='last', ignore_index=True)
    return odf


def list_featurebyIndustry(industry, num_of_feat=100, mw_avail='all'):
    industry = industry.lower().strip()
    if mw_avail == 'yes':
        odf = df_input.loc[df_input['Industry'] == industry].loc[(df_input['Available in Mobilewalla'] == 'fm') | (
                    df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                       ignore_index=True).head(
            num_of_feat)
    else:
        odf = df_input.loc[df_input['Industry'] == industry].drop_duplicates(keep='last', ignore_index=True).head(
            num_of_feat)
    return odf


def list_featurebyUsecase(usecase, num_of_feat=100, mw_avail='all'):
    usecase = usecase.lower().strip()
    if mw_avail == 'yes':
        odf = df_input.loc[df_input['Usecase'] == usecase].loc[(df_input['Available in Mobilewalla'] == 'fm') | (
                    df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                       ignore_index=True).head(
            num_of_feat)
    else:
        odf = df_input.loc[df_input['Usecase'] == usecase].drop_duplicates(keep='last', ignore_index=True).head(
            num_of_feat)
    return odf


def list_featurebyPair(industry, usecase, num_of_feat=100, mw_avail='all'):
    industry = industry.lower().strip()
    usecase = usecase.lower().strip()
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
    odf = df_input.loc[
        (df_input['Available in Mobilewalla'] == 'fm') | (
                    df_input['Available in Mobilewalla'] == 'custom')].drop_duplicates(keep='last',
                                                                                       ignore_index=True).head(
        num_of_feat)
    return odf
