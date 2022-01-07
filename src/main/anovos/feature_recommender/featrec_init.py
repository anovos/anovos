import pandas as pd

def init_featureRecommender(input_path):
    df_final = pd.read_csv(input_path)
    return df_final

# demo_featrec location goes here
input_path = '..../data/feature_recommender/demo_featrec.csv'
df_input = init_featureRecommender(input_path)