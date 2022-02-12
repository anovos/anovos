from anovos.feature_recommender.featrec_init import *
import re
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt


def feature_recommendation(df, name_column=None, desc_column=None, suggested_industry='all', suggested_usecase='all',
                           semantic=True, top_n=2, threshold=0.3):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Invalid input for df')
    if type(top_n) != int or top_n < 0:
        raise TypeError('Invalid input for top_n')
    if top_n > len(list_train):
        raise TypeError('top_n value is too large')
    list_user, df_user = recommendation_data_prep(df, name_column, desc_column)

    if suggested_industry != 'all' and suggested_industry == 'all':
        suggested_industry = process_industry(suggested_industry, semantic)
        df_rec_fr = df_rec[df_rec['Industry'].str.contains(suggested_industry)]
        list_keep = list(df_rec_fr.index)
        list_embedding_train_fr = [list_embedding_train.tolist()[x] for x in list_keep]
        df_rec_fr = df_rec_fr.reset_index(drop=True)
    elif suggested_usecase != 'all' and suggested_industry == 'all':
        suggested_usecase = process_usecase(suggested_usecase, semantic)
        df_rec_fr = df_rec[df_rec['Usecase'].str.contains(suggested_usecase)]
        list_keep = list(df_rec_fr.index)
        list_embedding_train_fr = [list_embedding_train.tolist()[x] for x in list_keep]
        df_rec_fr = df_rec_fr.reset_index(drop=True)
    elif suggested_usecase != 'all' and suggested_industry != 'all':
        suggested_industry = process_industry(suggested_industry, semantic)
        suggested_usecase = process_usecase(suggested_usecase, semantic)
        df_rec_fr = df_rec[
            df_rec['Industry'].str.contains(suggested_industry) & df_rec['Usecase'].str.contains(suggested_usecase)]
        if len(df_rec_fr) > 0:
            list_keep = list(df_rec_fr.index)
            list_embedding_train_fr = [list_embedding_train.tolist()[x] for x in list_keep]
            df_rec_fr = df_rec_fr.reset_index(drop=True)
        else:
            df_out = pd.DataFrame(
                columns=['Input Attribute Name', 'Input Attribute Description', 'Recommended Feature Name',
                         'Recommended Feature Description', 'Feature Similarity Score', 'Industry',
                         'Usecase',
                         'Source'])
            print('Industry/Usecase pair does not exist.')
            return df_out
    else:
        df_rec_fr = df_rec
        list_embedding_train_fr = list_embedding_train

    if name_column == None:
        df_out = pd.DataFrame(columns=['Input Attribute Description', 'Recommended Feature Name',
                                       'Recommended Feature Description', 'Feature Similarity Score', 'Industry',
                                       'Usecase',
                                       'Source'])
    elif desc_column == None:
        df_out = pd.DataFrame(columns=['Input Attribute Name', 'Recommended Feature Name',
                                       'Recommended Feature Description', 'Feature Similarity Score', 'Industry',
                                       'Usecase',
                                       'Source'])
    else:
        df_out = pd.DataFrame(
            columns=['Input Attribute Name', 'Input Attribute Description', 'Recommended Feature Name',
                     'Recommended Feature Description', 'Feature Similarity Score', 'Industry',
                     'Usecase',
                     'Source'])
    list_embedding_user = model.encode(list_user, convert_to_tensor=True)
    for i, feature in enumerate(list_user):
        cos_scores = util.pytorch_cos_sim(list_embedding_user, list_embedding_train_fr)[i]
        top_results = np.argpartition(-cos_scores, range(top_n))[0:top_n]
        for idx in top_results[0:top_n]:
            single_score = "%.4f" % (cos_scores[idx])
            if name_column == None:
                if float(single_score) >= threshold:
                    df_append = pd.DataFrame([[df_user[desc_column].iloc[i],
                                               df_rec_fr['Feature Name'].iloc[int(idx)],
                                               df_rec_fr['Feature Description'].iloc[int(idx)],
                                               "%.4f" % (cos_scores[idx]),
                                               df_rec_fr['Industry'].iloc[int(idx)],
                                               df_rec_fr['Usecase'].iloc[int(idx)],
                                               df_rec_fr['Source'].iloc[int(idx)]]],
                                             columns=['Input Attribute Description', 'Recommended Feature Name',
                                                      'Recommended Feature Description', 'Feature Similarity Score',
                                                      'Industry',
                                                      'Usecase', 'Source'])
                else:
                    df_append = pd.DataFrame(
                        [[df_user[desc_column].iloc[i], 'Null', 'Null', 'Null', 'Null', 'Null', 'Null']],
                        columns=['Input Attribute Description', 'Recommended Feature Name',
                                 'Recommended Feature Description', 'Feature Similarity Score',
                                 'Industry',
                                 'Usecase', 'Source'])
            elif desc_column == None:
                if float(single_score) >= threshold:
                    df_append = pd.DataFrame([[df_user[name_column].iloc[i],
                                               df_rec_fr['Feature Name'].iloc[int(idx)],
                                               df_rec_fr['Feature Description'].iloc[int(idx)],
                                               "%.4f" % (cos_scores[idx]),
                                               df_rec_fr['Industry'].iloc[int(idx)],
                                               df_rec_fr['Usecase'].iloc[int(idx)],
                                               df_rec_fr['Source'].iloc[int(idx)]]],
                                             columns=['Input Attribute Name', 'Recommended Feature Name',
                                                      'Recommended Feature Description', 'Feature Similarity Score',
                                                      'Industry',
                                                      'Usecase', 'Source'])
                else:
                    df_append = pd.DataFrame(
                        [[df_user[desc_column].iloc[i], 'Null', 'Null', 'Null', 'Null', 'Null', 'Null']],
                        columns=['Input Attribute Name', 'Recommended Feature Name',
                                 'Recommended Feature Description', 'Feature Similarity Score',
                                 'Industry',
                                 'Usecase', 'Source'])
            else:
                if float(single_score) >= threshold:
                    df_append = pd.DataFrame([[df_user[name_column].iloc[i], df_user[desc_column].iloc[i],
                                               df_rec_fr['Feature Name'].iloc[int(idx)],
                                               df_rec_fr['Feature Description'].iloc[int(idx)],
                                               "%.4f" % (cos_scores[idx]),
                                               df_rec_fr['Industry'].iloc[int(idx)],
                                               df_rec_fr['Usecase'].iloc[int(idx)],
                                               df_rec_fr['Source'].iloc[int(idx)]]],
                                             columns=['Input Attribute Name', 'Input Attribute Description',
                                                      'Recommended Feature Name',
                                                      'Recommended Feature Description', 'Feature Similarity Score',
                                                      'Industry',
                                                      'Usecase', 'Source'])
                else:
                    df_append = pd.DataFrame(
                        [[df_user[name_column].iloc[i], df_user[desc_column].iloc[i], 'Null', 'Null', 'Null', 'Null',
                          'Null', 'Null']],
                        columns=['Input Attribute Name', 'Input Attribute Description', 'Recommended Feature Name',
                                 'Recommended Feature Description', 'Feature Similarity Score',
                                 'Industry',
                                 'Usecase', 'Source'])
            df_out = df_out.append(df_append, ignore_index=True)
    return df_out


def fine_attr_by_relevance(df, building_corpus, name_column=None, desc_column=None, threshold=0.3):
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
    list_user, df_user = recommendation_data_prep(df, name_column, desc_column)
    list_embedding_user = model.encode(list_user, convert_to_tensor=True)
    list_embedding_building = model.encode(building_corpus, convert_to_tensor=True)
    for i, feature in enumerate(building_corpus):
        if name_column == None:
            df_append = pd.DataFrame(columns=['Input Feature Desc',
                                              'Recommended Input Attribute Description',
                                              'Input Attribute Similarity Score'])
        elif desc_column == None:
            df_append = pd.DataFrame(
                columns=['Input Feature Desc', 'Recommended Input Attribute Name', 'Input Attribute Similarity Score'])
        else:
            df_append = pd.DataFrame(columns=['Input Feature Desc', 'Recommended Input Attribute Name',
                                              'Recommended Input Attribute Description',
                                              'Input Attribute Similarity Score'])
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


def sankey_visualization(df, industry_included=False, usecase_included=False):
    fr_proper_col_list = [
        'Recommended Feature Name', 'Recommended Feature Description',
        'Feature Similarity Score', 'Industry', 'Usecase', 'Source'
    ]
    attr_proper_col_list = [
        'Input Feature Desc', 'Input Attribute Similarity Score'
    ]
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Invalid input for df')
    if not all(x in list(df.columns) for x in fr_proper_col_list) and not all(
            x in list(df.columns) for x in attr_proper_col_list):
        raise TypeError(
            'df is not output DataFrame of Feature Recommendation functions')
    if type(industry_included) != bool:
        raise TypeError('Invalid input for industry_included')
    if type(usecase_included) != bool:
        raise TypeError('Invalid input for usecase_included')
    if 'Feature Similarity Score' in df.columns:
        if 'Input Attribute Name' in df.columns:
            name_source = 'Input Attribute Name'
        else:
            name_source = 'Input Attribute Description'
        name_target = 'Recommended Feature Name'
        name_score = 'Feature Similarity Score'
    else:
        name_source = 'Input Feature Desc'
        if 'Recommended Input Attribute Name' in df.columns:
            name_target = 'Recommended Input Attribute Name'
        else:
            name_target = 'Recommender Input Attribute Description'
        name_score = 'Input Attribute Similarity Score'
        if industry_included != False or usecase_included != False:
            print(
                'Input is fine_attr_by_relevance output DataFrame. There is no suggested Industry and/or Usecase.'
            )
        industry_included = False
        usecase_included = False
    industry_target = 'Industry'
    usecase_target = 'Usecase'
    df_iter = copy.deepcopy(df)
    for i in range(len(df_iter)):
        if str(df_iter[name_score][i]) == 'Null':
            df = df.drop([i])
    df = df.reset_index(drop=True)
    source = []
    target = []
    value = []
    if industry_included == False and usecase_included == False:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        label = source_list + target_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
    elif industry_included == False and usecase_included != False:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        raw_usecase_list = df[usecase_target].unique().tolist()
        usecase_list = []
        for i, item in enumerate(raw_usecase_list):
            if ', ' in raw_usecase_list[i]:
                raw_usecase_list[i] = raw_usecase_list[i].split(', ')
                for j, sub_item in enumerate(raw_usecase_list[i]):
                    usecase_list.append(sub_item)
            else:
                usecase_list.append(item)
        label = source_list + target_list + usecase_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
            temp_list = df[usecase_target][i].split(', ')
            for k, item in enumerate(temp_list):
                source.append(label.index(str(df[name_target][i])))
                target.append(label.index(str(item)))
                value.append(float(1))
    elif industry_included != False and usecase_included == False:
        source_list = df[name_source].unique().tolist()
        target_list = df[name_target].unique().tolist()
        raw_industry_list = df[industry_target].unique().tolist()
        industry_list = []
        for i, item in enumerate(raw_industry_list):
            if ', ' in raw_industry_list[i]:
                raw_industry_list[i] = raw_industry_list[i].split(', ')
                for j, sub_item in enumerate(raw_industry_list[i]):
                    industry_list.append(sub_item)
            else:
                industry_list.append(item)
        label = source_list + target_list + industry_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
            temp_list = df[industry_target][i].split(', ')
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
            if ', ' in raw_industry_list[i]:
                raw_industry_list[i] = raw_industry_list[i].split(', ')
                for j, sub_item in enumerate(raw_industry_list[i]):
                    industry_list.append(sub_item)
            else:
                industry_list.append(item)

        usecase_list = []
        for i, item in enumerate(raw_usecase_list):
            if ', ' in raw_usecase_list[i]:
                raw_usecase_list[i] = raw_usecase_list[i].split(', ')
                for j, sub_item in enumerate(raw_usecase_list[i]):
                    usecase_list.append(sub_item)
            else:
                usecase_list.append(item)

        label = source_list + target_list + industry_list + usecase_list
        for i in range(len(df)):
            source.append(label.index(str(df[name_source][i])))
            target.append(label.index(str(df[name_target][i])))
            value.append(float(df[name_score][i]))
            temp_list_industry = df[industry_target][i].split(', ')
            temp_list_usecase = df[usecase_target][i].split(', ')
            for k, item_industry in enumerate(temp_list_industry):
                source.append(label.index(str(df[name_target][i])))
                target.append(label.index(str(item_industry)))
                value.append(float(1))
                for j, item_usecase in enumerate(temp_list_usecase):
                    if item_usecase in list_usecase_by_industry(
                            item_industry)['Usecase'].tolist():
                        source.append(label.index(str(item_industry)))
                        target.append(label.index(str(item_usecase)))
                        value.append(float(1))
    line_color = [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for k in range(len(value))
    ]
    label_color = [
        "#" + ''.join([random.choice('0123456789ABCDEF') for e in range(6)])
        for f in range(len(label))
    ]
    fig = go.Figure(data=[
        go.Sankey(node=dict(pad=15,
                            thickness=20,
                            line=dict(color=line_color, width=0.5),
                            label=label,
                            color=label_color),
                  link=dict(source=source, target=target, value=value))
    ])
    fig.update_layout(title_text="Feature Recommendation Sankey Visualization",
                      font_size=10)
    return fig