import pandas as pd
from sentence_transformers import SentenceTransformer
from re import finditer
import copy


def init_featureRecommender(input_path):
    df_final = pd.read_csv(input_path)
    return df_final


def camel_case_split(identifier):
    a = ''
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    for m in matches:
        a += str(m.group(0)) + str(' ')
    return a


def recommendationDataPrep(df, name_column, desc_column):
    df_prep = copy.deepcopy(df)
    df_prep[name_column] = df_prep[name_column].str.replace('_', ' ')
    df_prep[name_column] = df_prep[name_column].astype(str)
    df_prep[desc_column] = df_prep[desc_column].astype(str)
    df_prep['combine'] = df_prep[[name_column, desc_column]].agg(' '.join, axis=1)
    df_prep_com = df_prep['combine']
    df_prep_com = df_prep_com.replace({"[^A-Za-z0-9 ]+": " "}, regex=True)
    for i in range(len(df_prep_com)):
        df_prep_com[i] = df_prep_com[i].strip()
        df_prep_com[i] = camel_case_split(df_prep_com[i])
    list_corpus = df_prep_com.to_list()
    df_prep = df_prep.drop('combine', axis=1)
    return list_corpus, df_prep


model = SentenceTransformer('all-mpnet-base-v2')
input_path = 'https://raw.githubusercontent.com/anovos/anovos/feature_recommender_beta/data/feature_recommender/flatten_fr_db.csv'
df_input = init_featureRecommender(input_path)
df_groupby = df_input.groupby(['Feature Name', 'Feature Description']).agg(
    {'Industry': lambda x: ' '.join(set(x.dropna())), 'Usecase': lambda x: ' '.join(set(x.dropna()))}).reset_index()
list_train, df_rec = recommendationDataPrep(df_groupby, 'Feature Name', 'Feature Description')

