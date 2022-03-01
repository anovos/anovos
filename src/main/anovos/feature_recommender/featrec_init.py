import copy
import os
from re import finditer

import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home


def detect_model_path():
    """

    Returns
    -------
    Local Feature Explorer and Recommender semantic model path (if the model is pre-downloaded)
    """
    transformers_path = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if transformers_path is None:
        try:
            torch_home = _get_torch_home()
        except ImportError:
            torch_home = os.path.expanduser(
                os.getenv(
                    "TORCH_HOME",
                    os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"),
                )
            )
        transformers_path = os.path.join(torch_home, "sentence_transformers")
    model_path = os.path.join(
        transformers_path, "sentence-transformers_all-mpnet-base-v2"
    )
    return model_path


def model_download():
    print("Starting the Semantic Model download")
    SentenceTransformer("all-mpnet-base-v2")
    print("Model downloading finished")


class _TransformerModel:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            model_path = detect_model_path()
            if os.path.exists(model_path):
                self._model = SentenceTransformer(model_path)
            else:
                raise FileNotFoundError(
                    "Model has not been downloaded. Please use model_download() function to download the model first"
                )
        return self._model


model_fer = _TransformerModel()


def init_input_fer():
    """

    Returns
    -------
    Loading the Feature Explorer and Recommender (FER) Input DataFrame (FER corpus)
    """
    input_path_fer = "https://raw.githubusercontent.com/anovos/anovos/main/data/feature_recommender/flatten_fr_db.csv"
    df_input_fer = pd.read_csv(input_path_fer)
    return df_input_fer


def get_column_name(df):
    """

    Parameters
    ----------
    df
        Input DataFrame

    Returns
    -------
    feature_name_column
        Column name of Feature Name in the input DataFrame (string)
    feature_desc_column
        Column name of Feature Description in the input DataFrame (string)
    industry_column
        Column name of Industry in the input DataFrame (string)
    usecase_column
        Column name of Usecase in the input DataFrame (string)
    """
    feature_name_column = str(df.columns.tolist()[0])
    feature_desc_column = str(df.columns.tolist()[1])
    industry_column = str(df.columns.tolist()[2])
    usecase_column = str(df.columns.tolist()[3])
    return (
        feature_name_column,
        feature_desc_column,
        industry_column,
        usecase_column,
    )


def camel_case_split(input):
    """

    Parameters
    ----------
    input
        Input (string) which requires cleaning

    Returns
    -------

    """
    processed_input = ""
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", input)
    for m in matches:
        processed_input += str(m.group(0)) + str(" ")
    return processed_input


def recommendation_data_prep(df, name_column, desc_column):
    """

    Parameters
    ----------
    df
        Input DataFrame
    name_column
        Column name of Input DataFrame attribute/ feature name (string)
    desc_column
        Column name of Input DataFrame attribute/ feature description (string)


    Returns
    -------
    list_corpus
        List of prepared data for Feature Recommender functions
    return df_prep
        Processed DataFrame for Feature Recommender functions

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Invalid input for df")
    if name_column not in df.columns and name_column is not None:
        raise TypeError("Invalid input for name_column")
    if desc_column not in df.columns and desc_column is not None:
        raise TypeError("Invalid input for desc_column")
    if name_column is None and desc_column is None:
        raise TypeError("Need at least one input for either name_column or desc_column")
    df_prep = copy.deepcopy(df)
    if name_column is None:
        df_prep[desc_column] = df_prep[desc_column].astype(str)
        df_prep_com = df_prep[desc_column]
    elif desc_column is None:
        df_prep[name_column] = df_prep[name_column].astype(str)
        df_prep_com = df_prep[name_column]
    else:
        df_prep[name_column] = df_prep[name_column].str.replace("_", " ")
        df_prep[name_column] = df_prep[name_column].astype(str)
        df_prep[desc_column] = df_prep[desc_column].astype(str)
        df_prep_com = df_prep[[name_column, desc_column]].agg(" ".join, axis=1)
    df_prep_com = df_prep_com.replace({"[^A-Za-z0-9 ]+": " "}, regex=True)
    for i in range(len(df_prep_com)):
        df_prep_com[i] = df_prep_com[i].strip()
        df_prep_com[i] = camel_case_split(df_prep_com[i])
    list_corpus = df_prep_com.to_list()
    return list_corpus, df_prep


def feature_exploration_prep():
    """

    Returns
    -------
    df_input_fer
        DataFrame used in Feature Exploration functions
    """
    df_input_fer = init_input_fer()
    df_input_fer = df_input_fer.rename(columns=lambda x: x.strip().replace(" ", "_"))
    return df_input_fer


def feature_recommendation_prep():
    """

    Returns
    -------
    list_train_fer
        List of prepared data for Feature Recommendation functions
    df_red_fer
        DataFrame used in Feature Recommendation functions
    list_embedding_train_fer
        List of embedding tensor for Feature Recommendation functions
    """
    df_input_fer = init_input_fer()
    (
        feature_name_column,
        feature_desc_column,
        industry_column,
        usecase_column,
    ) = get_column_name(df_input_fer)
    df_groupby_fer = (
        df_input_fer.groupby([feature_name_column, feature_desc_column])
        .agg(
            {
                industry_column: lambda x: ", ".join(set(x.dropna())),
                usecase_column: lambda x: ", ".join(set(x.dropna())),
            }
        )
        .reset_index()
    )
    list_train_fer, df_rec_fer = recommendation_data_prep(
        df_groupby_fer, feature_name_column, feature_name_column
    )
    list_embedding_train_fer = model_fer.model.encode(
        list_train_fer, convert_to_tensor=True
    )
    return list_train_fer, df_rec_fer, list_embedding_train_fer
