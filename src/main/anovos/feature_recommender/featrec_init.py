from anovos.feature_recommender.feature_exploration import *
from re import finditer
import copy


def camel_case_split(identifier):
    a = ''
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    for m in matches:
        a += str(m.group(0)) + str(' ')
    return a


def recommendation_data_prep(df, name_column, desc_column):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Invalid input for df')
    if name_column not in df.columns and name_column != None:
        raise TypeError('Invalid input for name_column')
    if desc_column not in df.columns and desc_column != None:
        raise TypeError('Invalid input for desc_column')
    if name_column == None and desc_column == None:
        raise TypeError('Need at least one input for either name_column or desc_column')
    df_prep = copy.deepcopy(df)
    if name_column == None:
        df_prep[desc_column] = df_prep[desc_column].astype(str)
        df_prep_com = df_prep[desc_column]
    elif desc_column == None:
        df_prep[name_column] = df_prep[name_column].astype(str)
        df_prep_com = df_prep[name_column]
    else:
        df_prep[name_column] = df_prep[name_column].str.replace('_', ' ')
        df_prep[name_column] = df_prep[name_column].astype(str)
        df_prep[desc_column] = df_prep[desc_column].astype(str)
        df_prep_com = df_prep[[name_column, desc_column]].agg(' '.join, axis=1)
    df_prep_com = df_prep_com.replace({"[^A-Za-z0-9 ]+": " "}, regex=True)
    for i in range(len(df_prep_com)):
        df_prep_com[i] = df_prep_com[i].strip()
        df_prep_com[i] = camel_case_split(df_prep_com[i])
    list_corpus = df_prep_com.to_list()
    return list_corpus, df_prep


df_groupby = df_input.groupby(['Feature Name', 'Feature Description']).agg(
    {'Industry': lambda x: ', '.join(set(x.dropna())), 'Usecase': lambda x: ', '.join(set(x.dropna())),
     'Source': lambda x: ', '.join(set(x.dropna()))}).reset_index()
list_train, df_rec = recommendation_data_prep(df_groupby, 'Feature Name', 'Feature Description')
list_embedding_train = model.encode(list_train, convert_to_tensor=True)
