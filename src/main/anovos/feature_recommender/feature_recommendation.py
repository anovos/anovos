from anovos.feature_recommender.featrec_init import *
import re
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt


def feature_recommendation(
    df,
    name_column=None,
    desc_column=None,
    suggested_industry="all",
    suggested_usecase="all",
    semantic=True,
    top_n=2,
    threshold=0.3,
):
    """

    Parameters
    ----------
    df :
        Input DataFrame - Users' Data dictionary. It is expected to consist of attribute name and/or attribute description
    name_column :
        Input, column name of Attribute Name in Input DataFrame (string). Default is None.
    desc_column :
        Input, column name of Attribute Description in Input DataFrame (string). Default is None.
    suggested_industry :
        Input, Industry of interest to the user (if any) to be filtered out (string). Default is 'all', meaning all Industries available.
    suggested_usecase :
        Input, Usecase of interest to the user (if any) to be filtered out (string). Default is 'all', meaning all Usecases available.
    semantic :
        Input semantic (boolean) - Whether the input needs to go through semantic similarity or not. Default is True.
    top_n :
        Number of features displayed (int). Default is 2
    threshold :
        Input threshold value (float). Default is 0.3

    Returns
    -------
    type
        DataFrame with Recommended Features with the Input DataFrame and/or Users' Industry/Usecase of interest

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Invalid input for df")
    if type(top_n) != int or top_n < 0:
        raise TypeError("Invalid input for top_n")
    if top_n > len(list_train_fer):
        raise TypeError("top_n value is too large")
    if type(threshold) != float:
        raise TypeError("Invalid input for building_corpus")
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
            list_embedding_train_fer.tolist()[x] for x in list_keep
        ]
        df_rec_fr = df_rec_fr.reset_index(drop=True)
    elif suggested_usecase != "all" and suggested_industry == "all":
        suggested_usecase = process_usecase(suggested_usecase, semantic)
        df_rec_fr = df_rec_fer[df_rec_fer.iloc[:, 3].str.contains(suggested_usecase)]
        list_keep = list(df_rec_fr.index)
        list_embedding_train_fr = [
            list_embedding_train_fer.tolist()[x] for x in list_keep
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
                list_embedding_train_fer.tolist()[x] for x in list_keep
            ]
            df_rec_fr = df_rec_fr.reset_index(drop=True)
        else:
            df_out = pd.DataFrame(
                columns=[
                    "Input_Attribute_Name",
                    "Input_Attribute_Description",
                    "Recommended_Feature_Name",
                    "Recommended_Feature_Description",
                    "Feature_Similarity_Score",
                    "Industry",
                    "Usecase",
                    "Source",
                ]
            )
            print("Industry/Usecase pair does not exist.")
            return df_out
    else:
        df_rec_fr = df_rec_fer
        list_embedding_train_fr = list_embedding_train_fer

    if name_column == None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Attribute_Description",
                "Recommended_Feature_Name",
                "Recommended_Feature_Description",
                "Feature_Similarity_Score",
                "Industry",
                "Usecase",
                "Source",
            ]
        )
    elif desc_column == None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Attribute_Name",
                "Recommended_Feature_Name",
                "Recommended_Feature_Description",
                "Feature_Similarity_Score",
                "Industry",
                "Usecase",
                "Source",
            ]
        )
    else:
        df_out = pd.DataFrame(
            columns=[
                "Input_Attribute_Name",
                "Input_Attribute_Description",
                "Recommended_Feature_Name",
                "Recommended_Feature_Description",
                "Feature_Similarity_Score",
                "Industry",
                "Usecase",
                "Source",
            ]
        )
    list_embedding_user = model_fer.encode(list_user, convert_to_tensor=True)
    for i, feature in enumerate(list_user):
        cos_scores = util.pytorch_cos_sim(list_embedding_user, list_embedding_train_fr)[
            i
        ]
        top_results = np.argpartition(-cos_scores, range(top_n))[0:top_n]
        for idx in top_results[0:top_n]:
            single_score = "%.4f" % (cos_scores[idx])
            if name_column == None:
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
                                df_rec_fr[source_column].iloc[int(idx)],
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Description",
                            "Recommended_Feature_Name",
                            "Recommended_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                            "Source",
                        ],
                    )
                else:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[desc_column].iloc[i],
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Description",
                            "Recommended_Feature_Name",
                            "Recommended_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                            "Source",
                        ],
                    )
            elif desc_column == None:
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
                                df_rec_fr[source_column].iloc[int(idx)],
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Recommended_Feature_Name",
                            "Recommended_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                            "Source",
                        ],
                    )
                else:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[desc_column].iloc[i],
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Recommended_Feature_Name",
                            "Recommended_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                            "Source",
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
                                df_rec_fr[source_column].iloc[int(idx)],
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Input_Attribute_Description",
                            "Recommended_Feature_Name",
                            "Recommended_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                            "Source",
                        ],
                    )
                else:
                    df_append = pd.DataFrame(
                        [
                            [
                                df_user[name_column].iloc[i],
                                df_user[desc_column].iloc[i],
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                                "Null",
                            ]
                        ],
                        columns=[
                            "Input_Attribute_Name",
                            "Input_Attribute_Description",
                            "Recommended_Feature_Name",
                            "Recommended_Feature_Description",
                            "Feature_Similarity_Score",
                            "Industry",
                            "Usecase",
                            "Source",
                        ],
                    )
            df_out = df_out.append(df_append, ignore_index=True)
    return df_out


def find_attr_by_relevance(
    df, building_corpus, name_column=None, desc_column=None, threshold=0.3
):
    """

    Parameters
    ----------
    df :
        Input DataFrame - Users' Data dictionary. It is expected to consist of attribute name and/or attribute description
    building_corpus :
        Input Feature Description (list)
    name_column :
        Input, column name of Attribute Name in Input DataFrame (string). Default is None.
    desc_column :
        Input, column name of Attribute Description in Input DataFrame (string). Default is None.
    threshold :
        Input threshold value (float). Default is 0.3

    Returns
    -------
    type
        DataFrame with Input Feature Description and Input Attribute matching

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
    if name_column == None:
        df_out = pd.DataFrame(
            columns=[
                "Input_Feature_Description",
                "Recommended_Input_Attribute_Description",
                "Input_Attribute_Similarity_Score",
            ]
        )
    elif desc_column == None:
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
    list_embedding_user = model_fer.encode(list_user, convert_to_tensor=True)
    list_embedding_building = model_fer.encode(building_corpus, convert_to_tensor=True)
    for i, feature in enumerate(building_corpus):
        if name_column == None:
            df_append = pd.DataFrame(
                columns=[
                    "Input_Feature_Description",
                    "Recommended_Input_Attribute_Description",
                    "Input_Attribute_Similarity_Score",
                ]
            )
        elif desc_column == None:
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
                if name_column == None:
                    df_append = df_append.append(
                        {
                            "Input_Feature_Description": feature,
                            "Recommended_Input_Attribute_Description": df_user[
                                desc_column
                            ].iloc[int(idx)],
                            "Input_Attribute_Similarity_Score": single_score,
                        },
                        ignore_index=True,
                    )
                elif desc_column == None:
                    df_append = df_append.append(
                        {
                            "Input_Feature_Description": feature,
                            "Recommended_Input_Attribute_Name": df_user[
                                name_column
                            ].iloc[int(idx)],
                            "Input_Attribute_Similarity_Score": single_score,
                        },
                        ignore_index=True,
                    )
                else:
                    df_append = df_append.append(
                        {
                            "Input_Feature_Description": feature,
                            "Recommended_Input_Attribute_Name": df_user[
                                name_column
                            ].iloc[int(idx)],
                            "Recommended_Input_Attribute_Description": df_user[
                                desc_column
                            ].iloc[int(idx)],
                            "Input_Attribute_Similarity_Score": single_score,
                        },
                        ignore_index=True,
                    )
        if len(df_append) == 0:
            if name_column == None:
                df_append = df_append.append(
                    {
                        "Input_Feature_Description": feature,
                        "Recommended_Input_Attribute_Description": "Null",
                        "Input_Attribute_Similarity_Score": "Null",
                    },
                    ignore_index=True,
                )
            elif desc_column == None:
                df_append = df_append.append(
                    {
                        "Input_Feature_Description": feature,
                        "Recommended_Input_Attribute_Name": "Null",
                        "Input_Attribute_Similarity_Score": "Null",
                    },
                    ignore_index=True,
                )
            else:
                df_append = df_append.append(
                    {
                        "Input_Feature_Description": feature,
                        "Recommended_Input_Attribute_Name": "Null",
                        "Recommended_Input_Attribute_Description": "Null",
                        "Input_Attribute_Similarity_Score": "Null",
                    },
                    ignore_index=True,
                )
        df_out = df_out.append(df_append, ignore_index=True)
    return df_out


def sankey_visualization(df, industry_included=False, usecase_included=False):
    """

    Parameters
    ----------
    df :
        Input DataFrame. This DataFrame needs to be output of feature_recommendation or find_attr_by_relevance, or in the same format.
    industry_included :
        Whether the plot needs to include industry mapping or not (boolean). Default is False
    usecase_included :
        Whether the plot needs to include usecase mapping or not (boolean). Default is False

    Returns
    -------
    type
        Sankey plot

    """
    fr_proper_col_list = [
        "Recommended_Feature_Name",
        "Recommended_Feature_Description",
        "Feature_Similarity_Score",
        "Industry",
        "Usecase",
        "Source",
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
        name_target = "Recommended_Feature_Name"
        name_score = "Feature_Similarity_Score"
    else:
        name_source = "Input_Feature_Description"
        if "Recommended_Input_Attribute_Name" in df.columns:
            name_target = "Recommended_Input_Attribute_Name"
        else:
            name_target = "Recommended_Input_Attribute_Description"
        name_score = "Input_Attribute_Similarity_Score"
        if industry_included != False or usecase_included != False:
            print(
                "Input is find_attr_by_relevance output DataFrame. There is no suggested Industry and/or Usecase."
            )
        industry_included = False
        usecase_included = False
    industry_target = "Industry"
    usecase_target = "Usecase"
    df_iter = copy.deepcopy(df)
    for i in range(len(df_iter)):
        if str(df_iter[name_score][i]) == "Null":
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
    elif industry_included != False and usecase_included == False:
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
    fig.update_layout(
        title_text="Feature Recommendation Sankey Visualization", font_size=10
    )
    return fig
