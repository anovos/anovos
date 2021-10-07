import yaml
import subprocess
import copy
import os
import sys
import pandas as pd
import numpy as np
import datapane as dp
import plotly
import plotly.express as px
from plotly.figure_factory import create_distplot
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def ends_with(string, end_str="/"):
    '''
    :param string: "s3:mw-bucket"
    :param end_str: "/"
    :return: "s3:mw-bucket/"
    '''
    string = str(string)
    if string.endswith(end_str):
        return string
    return string + end_str


def remove_u_score(col):
    col_ = col.split("_")
    bl=[]
    for i in col_:
        bl.append(i[0].upper()+i[1:])
    
    return " ".join(bl)


global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
global_plot_bg_color = 'rgba(0,0,0,0)'
global_paper_bg_color = 'rgba(0,0,0,0)'


config_file = "configs.yaml"
config_file = open(config_file, 'r')

args = yaml.load(config_file, yaml.SafeLoader)

list_tabs = args.get('report_gen_inter',None).get('output_pandas_df',None).get('list_tabs').split(",")
list_tab1_arr = args.get('report_gen_inter',None).get('output_pandas_df',None).get('list_tab1').split(",")
list_tab2_arr = args.get('report_gen_inter',None).get('output_pandas_df',None).get('list_tab2').split(",")
list_tab3_arr = args.get('report_gen_inter',None).get('output_pandas_df',None).get('list_tab3').split(",")
islabel = args.get('report_gen_inter',None).get('output_pandas_df',None).get('islabel',None)
data_drift_check = args.get('report_gen_inter',None).get('data_drift').get('driftcheckrequired')
label = args.get('report_gen_inter',None).get('charts_to_objects',None).get('label',None)
base_path = args.get('report_gen_final',None).get('base_path')
data_dictionary_path = args.get('report_gen_final',None).get('data_dictionary_path',None)
id_col = args.get('report_gen_inter',None).get('charts_to_objects',None).get('id_col',None)
t1 = args.get('drift_detector',None).get('drift_statistics',None).get('threshold')
t2 = args.get('report_gen_final',None).get('threshold')

corr_threshold = args.get('report_gen_final',None).get('corr_threshold',None)
iv_threshold = args.get('report_gen_final',None).get('iv_threshold',None)

remove_list = ['IV_calculation','IG_calculation']

if islabel == False:
    list_tab3_arr = [x for x in list_tab3_arr if x not in remove_list]
else:
    pass

list_tabs_all = [list_tab1_arr,list_tab2_arr,list_tab3_arr]


blank_chart = go.Figure()
blank_chart.update_layout(autosize=False,width=10,height=10)
blank_chart.layout.plot_bgcolor = global_plot_bg_color
blank_chart.layout.paper_bgcolor = global_paper_bg_color
blank_chart.update_xaxes(visible=False)
blank_chart.update_yaxes(visible=False)
blank_chart


df_stability = pd.read_csv("./output/stability/historical_metrics.csv")
df_stability['idx']=df_stability['idx'].astype(str).apply(lambda x: 'idx'+x)


def line_chart_gen_stability(df1,df2,col):
    
    def val_cat(val):
        if val>=3.5:
            return "Very Stable"
        elif val>=3 and val<3.5:
            return "Stable"
        elif val>=2 and val<3:
            return "Marginally Stable"
        elif val>=1 and val<2:
            return "Unstable"
        elif val>=0 and val<1:
            return "Very Unstable"
        else:
            return "Out of Range"

    df1 = df1[df1["attribute"]==col]
    
    val_si = list(df2[df2["attribute"]==col].stability_index.values)[0]
    
    f1 = go.Figure()
    f1.add_trace(go.Indicator(
    mode = "gauge+number",
    value = val_si,
    gauge = {
        'axis': {'range': [None, 4], 'tickwidth':1, 'tickcolor': "black"},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 1], 'color': px.colors.sequential.Reds[7]},
            {'range': [1, 2], 'color': px.colors.sequential.Reds[6]},
            {'range': [2, 3], 'color': px.colors.sequential.Oranges[4]},
            {'range': [3, 3.5], 'color': px.colors.sequential.BuGn[7]},
            {'range': [3.5,4], 'color': px.colors.sequential.BuGn[8]}],
        'threshold': {
            'line': {'color': "black", 'width': 3},
            'thickness':1,
            'value': val_si},
        'bar': {'color': global_plot_bg_color}},
        title = {'text': "Order of Stability : " + val_cat(val_si)}))

    f1.update_layout(height = 400,font = {'color': "black", 'family': "Arial"})
    
    
    
    f2 = px.line(df1, x='idx', y='mean',markers=True)
    f2.update_traces(line_color=global_theme[2])
    f2.layout.plot_bgcolor = global_plot_bg_color
    f2.layout.paper_bgcolor = global_paper_bg_color
    

    f3 = px.line(df1, x='idx', y='kurtosis',markers=True)
    f3.update_traces(line_color=global_theme[4])
    f3.layout.plot_bgcolor = global_plot_bg_color
    f3.layout.paper_bgcolor = global_paper_bg_color


    f4 = px.line(df1, x='idx', y='stddev',markers=True)
    f4.update_traces(line_color=global_theme[6])
    f4.layout.plot_bgcolor = global_plot_bg_color
    f4.layout.paper_bgcolor = global_paper_bg_color
    
    f5 = "Distribution of Stability for " + str(col.upper())
    
#     f4 = go.Figure()
#     f4.add_trace(go.Indicator(
#         mode = "gauge+number", value = val_rand,
#         domain = {'x': [0.05, 1], 'y': [0.7, 1]},
#         title = {'text' :"Stability Index"},
#         gauge = {
#             'shape': "bullet",
#             'axis': {'range': [0,4]},
#             'threshold': {
#                 'line': {'color': "black", 'width': 4},
#                 'thickness': 1,
#                 'value': val_rand},
#                 'steps': [
#             {'range': [0, 0.03], 'color': px.colors.sequential.Reds[7]},
#             {'range': [0.03, 0.1], 'color': px.colors.sequential.Reds[6]},
#             {'range': [0.1, 0.2], 'color': px.colors.sequential.Oranges[4]},
#             {'range': [0.2, 0.5], 'color': px.colors.sequential.BuGn[7]},
#             {'range': [0.5,1], 'color': px.colors.sequential.BuGn[8]}],
#             'bar': {'color': global_plot_bg_color}}))
    


    return dp.Group(dp.Text("#"),dp.Text(f5), dp.Plot(f1),dp.Group(dp.Plot(f2),dp.Plot(f3),dp.Plot(f4),columns=3),rows=4,label=col)



var_stability = list(set(df_stability.attribute.values))


df_si_ = pd.read_csv("./output/stability/stability_index.csv")
df_si = df_si_[["attribute","stability_index","mean_si","stddev_si","kurtosis_si","mean_cv","stddev_cv","kurtosis_cv"]]
stability_attr = list(df_si_[df_si_.flagged.values==1].attribute.values)
total_stability_attr = list(df_si_.attribute.values)
len(stability_attr)/len(total_stability_attr)



line_chart_list = []
for i in var_stability:
    line_chart_list.append(line_chart_gen_stability(df_stability,df_si,i))



data = [['1','Descriptor Statistics','','Used to summarize basic and statistical infromation of the datasets'],\
        ['1.1','Global Summary','int, string','Summarize global information about the datasets like number of rows and columns, name and number of categorical and numerical attributes'],\
        ['1.2','Statistics by Metric Type','',''],\
        ['','Measures of Shape','float','Describe the distribution(or pattern) of different attributes in the datasets using skewness and kurtosis'],\
        ['','Measures of Central Tendency','int, float, double,  string','Describe the central position of each attributes in datasets by finding basic measures like mean, median and mode'],\
        ['','Measure Of Percentiles','int','Indicate the value below which a given percentage of data of given attribute falls'],\
        ['','Measures Of Dispersion','int, float','Measure the spread of data about the mean e.g., Standard Deviation, Variance, Covariance, IQR and range of each attribute of datasets'],\
        ['','Measures Of Cardinality','int, float','Measure the count of unique values present in each attribute'],\
        ['','Measures Of Counts','int, float','Measure the sparsity of the datasets, e.g., fill count and percenatge, missing value count and percentage and nonzero count and percenatge'],\
        ['1.3','Attribute Visualization','',''],\
        ['','Numerical','histogram','Visualize the distributions of Numerical attributes using Histograms'],\
        ['','Categorical','bar plot','Visualize the distributions of Categorical attributes using Barplot'],\
        ['2','Quality check','','Used to check the quality of a datasets both at column level and row level. '],\
        ['2.1','Column Level','',''],\
        ['','IDness Detection','',''],\
        ['','Null Detection','int, float','Detect the sparsity of the datasets, e.g., count and percentage of missing value of attributes'],\
        ['','Baisedness Detection','int, float, double, string','Detect the baisedness of the attributes by finding the mode and its percenatge(value that is most frequent in the data)'],\
        ['','InvalidEntries Detection','int, float, string','Detect the entries and count of invalid values or noise present in the datasets'],\
        ['','Outlier Detection','int, float, double, plot','Used to detect and visualize the outlier present in numerical attributes of the datasets'],\
        ['2.2','Row Level','',''],\
        ['','Duplicate Detection','int','Measure number of rows in the datasets that have same value for each attribute'],\
        ['','Rows WMissingFeats','int, float','Measure the count/percentage of rows which have missing/null attributes'],\
        ['3','Association & Interactions','','Used to find interesting associations and relationships among attributes of daatsets'],\
        ['3.1','Association Matrix','',''],\
        ['','Correlation Matrix','float','Measure the strength of relationship among each attribute by finding correlation coefficient having range -1.0 to 1.0.'],\
        ['','IV Calculations',' float','Information Value Calculations- Used to rank variables on the basis of their importance.Greater the value of IV higher rthe attribute importance. IV less than 0.02 is not useful for prediction'],\
        ['','IG Calculations','float','Information Gain- Measures the reduction in entropy by splitting a dataset according to given value of a attribute.'],\
        ['','Varibale clustering','set of clusters','Divides the numerical attributes into disjoint or hierarchical clusters based on linear relationship of attributes'],\
        ['3.2','Association Plot','',''],\
        ['','Correlation Matrix','heat map','Used to Visualize the strength of relationship among attributes by ploting heat map'],\
        ['','IV Calculations','bar plot','Used to Visualize attribute importance of in increasing or decreasing order using barplot'],\
        ['','IG Calculations','bar plot','Used to visualize the purity of attributes in dataset using barplot i.e., how a change to the datasets impact the distribution of data.'],\
        ['','Varibale clustering','pie chart','Used to visualize how numerical featrures are group together into different clusters formed by variable clustering'],\
        ['3.3','Attribute to Target Association','',''],\
        ['','Numerical ','histogram','Used to Visualize the bucket-wise percentage/ditribution of data for Numerical attributes having Target Variable greater/less than or equal to given threshold value'],\
        ['','Categorical','bar plot','Used to Visualize the Category-wise percentage/distributions of data for categorical attributes having Target Varibale greater/less than or equal to  given threshold value.'],\
        ['4','Data Drift ','','Used to monitor differences between target and source datasets '],\
        ['4.1','Data Drift Analyzer','',''],\
        ['','PSI','int, float','Population Stability Index- Measure how much a attribute has shifted in distribution between two sample of dataset(target and source datasets )'],\
        ['','JSD','float','Jensen-Shannon Divergence- Used to quantify the difference(or similarity) between distributions of two sample data(target and source datasets). It ranges between 0-1. smaller the score of JSD higher the similarity of two datasets. '],\
        ['','HD','float','Hellinger Distance- Measure the similarity in distribution between two sample of dataset(target and source datasets ). Smaller the value of hellinger distance higher the similarity in distribution of two sample dataset.'],\
        ['','KS','float','kolmogorov-Smirnov Test- It quantifies a distance between distribution of two sample dataset to check the similarity between them. Greater the value of K-S test p-value higher the similarity between two dataset.']]

metric_dict = pd.DataFrame(data, columns = ['Module No.', 'Module Name','Metric type','Metric definitions'])


def data_analyzer_output(p2,tab_name):

    df_list=[]
    txt_list =[]
    plot_list=[]
    idx = list_tabs.index(tab_name)

    for index,i in enumerate(list_tabs_all[idx]):

        if tab_name == "quality_checker":

            df_list.append([dp.Text("### " + str(remove_u_score(i))),dp.DataTable(pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(i) + ".csv").round(3))])
        
        elif tab_name == "association_evaluator":
            
            for j in list_tabs_all[idx]:
                
                if j == "correlation_matrix":

                    df_list_ = pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(j) + ".csv").round(3)
                    feats_order = list(df_list_["attribute"].values)
                    df_list_ = df_list_.round(3)
                    fig = px.imshow(df_list_[feats_order],y=feats_order,color_continuous_scale=global_theme,aspect="auto")
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
#                     fig.update_layout(title_text=str("Correlation Plot "))
                    df_list.append(dp.DataTable(df_list_[["attribute"]+feats_order],label= remove_u_score(j)))
                    plot_list.append(dp.Plot(fig,label=remove_u_score(j)))
                    
                    
                elif j == "variable_clustering":

                    df_list_ = pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(j) + ".csv").round(3)
                    fig = px.sunburst(df_list_, path=['Cluster', 'Attribute'], values='RS_Ratio',color_discrete_sequence=global_theme)
#                     fig.update_layout(title_text=str("Distribution of homogenous variable across Clusters"))
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
#                     fig.update_layout(title_text=str("Variable Clustering Plot "))
                    fig.layout.autosize=True
                    df_list.append(dp.DataTable(df_list_,label=remove_u_score(j)))
                    plot_list.append(dp.Plot(fig,label=remove_u_score(j)))
                
                else:

                    try:
                        df_list_ = pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(j) + ".csv").round(3)
                        col_nm = [x for x in list(df_list_.columns) if "attribute" not in x]
                        df_list_ = df_list_.sort_values(col_nm[0], ascending=True)
                        fig = px.bar(df_list_,x=col_nm[0],y='attribute',orientation='h',color_discrete_sequence=global_theme)
                        fig.layout.plot_bgcolor = global_plot_bg_color
                        fig.layout.paper_bgcolor = global_paper_bg_color
#                         fig.update_layout(title_text=str("Representation of " + str(remove_u_score(j))))
                        fig.layout.autosize=True
                        df_list.append(dp.DataTable(df_list_,label=remove_u_score(j)))
                        plot_list.append(dp.Plot(fig,label=remove_u_score(j)))
                    except:
                        pass

            return df_list,plot_list
        else:
            
            df_list.append(dp.DataTable(pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(i) + ".csv").round(3),label=remove_u_score(list_tabs_all[idx][index])))

    return df_list

data = [['Descriptor Statistics','Global Summary','','Summarize global information about the datasets like number of rows and columns, name and number of categorical and numerical attributes'],
['Descriptor Statistics','Statistics by Metric Type','Measures of Shape','Describe the distribution(or pattern) of different attributes in the datasets using skewness and kurtosis'],
['Descriptor Statistics','Statistics by Metric Type','Measures of Central Tendency','Describe the central position of each attributes in datasets by finding basic measures like mean, median and mode'],
['Descriptor Statistics','Statistics by Metric Type','Measure Of Percentiles','Indicate the value below which a given percentage of data of given attribute falls'],
['Descriptor Statistics','Statistics by Metric Type','Measures Of Dispersion','Measure the spread of data about the mean e.g., Standard Deviation, Variance, Covariance, IQR and range of each attribute of datasets'],
['Descriptor Statistics','Statistics by Metric Type','Measures Of Cardinality','Measure the count of unique values present in each attribute'],
['Descriptor Statistics','Statistics by Metric Type','Measures Of Counts','Measure the sparsity of the datasets, e.g., fill count and percenatge, missing value count and percentage and nonzero count and percenatge'],
['Descriptor Statistics','Attribute Visualization','Numerical','Visualize the distributions of Numerical attributes using Histograms'],
['Descriptor Statistics','Attribute Visualization','Categorical','Visualize the distributions of Categorical attributes using Barplot'],
['Quality check','Column Level','IDness Detection',''],
['Quality check','Column Level','Null Detection','Detect the sparsity of the datasets, e.g., count and percentage of missing value of attributes'],
['Quality check','Column Level','Baisedness Detection','Detect the baisedness of the attributes by finding the mode and its percenatge(value that is most frequent in the data)'],
['Quality check','Column Level','InvalidEntries Detection','Detect the entries and count of invalid values or noise present in the datasets'],
['Quality check','Column Level','Outlier Detection','Used to detect and visualize the outlier present in numerical attributes of the datasets'],
['Quality check','Row Level','Duplicate Detection','Measure number of rows in the datasets that have same value for each attribute'],
['Quality check','Row Level','Rows WMissingFeats','Measure the count/percentage of rows which have missing/null attributes'],
['Association & Interactions','Association Matrix','Correlation Matrix','Measure the strength of relationship among each attribute by finding correlation coefficient having range -1.0 to 1.0.'],
['Association & Interactions','Association Matrix','IV Calculations','Information Value Calculations- Used to rank variables on the basis of their importance.Greater the value of IV higher rthe attribute importance. IV less than 0.02 is not useful for prediction'],
['Association & Interactions','Association Matrix','IG Calculations','Information Gain- Measures the reduction in entropy by splitting a dataset according to given value of a attribute.'],
['Association & Interactions','Association Matrix','Varibale clustering','Divides the numerical attributes into disjoint or hierarchical clusters based on linear relationship of attributes'],
['Association & Interactions','','',''],
['Association & Interactions','Association Plot','Correlation Matrix','Used to Visualize the strength of relationship among attributes by ploting heat map'],
['Association & Interactions','Association Plot','IV Calculations','Used to Visualize attribute importance of in increasing or decreasing order using barplot'],
['Association & Interactions','Association Plot','IG Calculations','Used to visualize the purity of attributes in dataset using barplot i.e., how a change to the datasets impact the distribution of data.'],
['Association & Interactions','Association Plot','Varibale clustering','Used to visualize how numerical featrures are group together into different clusters formed by variable clustering'],
['Association & Interactions','Attribute to Target Association','Numerical ','Used to Visualize the bucket-wise percentage/ditribution of data for Numerical attributes having Target Variable greater/less than or equal to given threshold value'],
['Association & Interactions','Attribute to Target Association','Categorical','Used to Visualize the Category-wise percentage/distributions of data for categorical attributes having Target Varibale greater/less than or equal to  given threshold value.'],
['Data Drift & Data Stability','Data Drift Analyzer','PSI','Population Stability Index- Measure how much a attribute has shifted in distribution between two sample of dataset(target and source datasets )'],
['Data Drift & Data Stability','Data Drift Analyzer','JSD','Jensen-Shannon Divergence- Used to quantify the difference(or similarity) between distributions of two sample data(target and source datasets). It ranges between 0-1. smaller the score of JSD higher the similarity of two datasets. '],
['Data Drift & Data Stability','Data Drift Analyzer','HD','Hellinger Distance- Measure the similarity in distribution between two sample of dataset(target and source datasets ). Smaller the value of hellinger distance higher the similarity in distribution of two sample dataset.'],
['Data Drift & Data Stability','Data Drift Analyzer','KS','kolmogorov-Smirnov Test- It quantifies a distance between distribution of two sample dataset to check the similarity between them. Greater the value of K-S test p-value higher the similarity between two dataset.']]

metric_dict = pd.DataFrame(data, columns = ['Module Name', 'Sub Module Name','Metric Name','Metric Definition'])

def main(base_path):
    
    p1 = ends_with(base_path) + ends_with("chart_objects")
    p2 = ends_with(base_path) + ends_with("pandas_df")
    p3 = ends_with(base_path) + ends_with("report")

    Path(p3).mkdir(parents=True, exist_ok=True)
    
    datatype_df = pd.read_csv(ends_with(base_path) + ends_with("pandas_df") + "data_type_df.csv")
    
    if data_dictionary_path is None:
        data_dict = datatype_df
    else:
        data_definitions_df = pd.read_csv(data_dictionary_path)
        data_dict = data_definitions_df.merge(datatype_df,how="outer",on="Attributes")
    
    
    
    global_summary_df = pd.read_csv(ends_with(base_path) + ends_with("pandas_df") + "global_summary_df.csv").reindex([2,3,4,1,5,6])
    
    
    #Drift Chart - 1

    metric_drift = ["PSI","JSD","HD","KS"]
    drift_df = pd.read_csv(ends_with(p2) + "drift_statistics.csv")
    drift_df = drift_df[drift_df.attribute.values!=id_col]
    len_feats = drift_df.shape[0]
    drift_df_stats = drift_df[drift_df.flagged.values==1]\
                            .melt(id_vars="attribute",value_vars=["PSI","JSD","HD","KS"])\
                            .sort_values(by=['variable','value'], ascending=False)
    
    drifted_feats = drift_df[drift_df.flagged.values==1].shape[0]
    
#     fig_metric_drift = px.sunburst(drift_df_stats, path=['variable','attribute'], values='value',color_discrete_sequence= global_theme)
#     fig_metric_drift.update_layout(margin=dict(t=10, b=10, r=10, l=10))

    
    fig_metric_drift = go.Figure()
    fig_metric_drift.add_trace(go.Scatter(x=list(drift_df[drift_df.flagged.values==1][metric_drift[0]].values),
                                          y=list(drift_df[drift_df.flagged.values==1].attribute.values),
                                          marker=dict(color=global_theme[1], size=14),
                                          mode="markers",
                                          name=metric_drift[0]))
    
    fig_metric_drift.add_trace(go.Scatter(x=list(drift_df[drift_df.flagged.values==1][metric_drift[1]].values),
                             y=list(drift_df[drift_df.flagged.values==1].attribute.values),
                             marker=dict(color=global_theme[3], size=14),
                             mode="markers",
                             name=metric_drift[1]))

    fig_metric_drift.add_trace(go.Scatter(x=list(drift_df[drift_df.flagged.values==1][metric_drift[2]].values),
                             y=list(drift_df[drift_df.flagged.values==1].attribute.values),
                             marker=dict(color=global_theme[5], size=14),
                             mode="markers",
                             name=metric_drift[2]))

    fig_metric_drift.add_trace(go.Scatter(x=list(drift_df[drift_df.flagged.values==1][metric_drift[3]].values),
                             y=list(drift_df[drift_df.flagged.values==1].attribute.values),
                             marker=dict(color=global_theme[7], size=14),
                             mode="markers",
                             name=metric_drift[3]))

    fig_metric_drift.add_vrect(x0=0,x1=t1,fillcolor=global_theme[7], opacity=0.1,layer="below", line_width=1),

    fig_metric_drift.update_layout(legend=dict(orientation="h",x=0.5, yanchor="top",xanchor="center"))
    fig_metric_drift.layout.plot_bgcolor = global_plot_bg_color
    fig_metric_drift.layout.paper_bgcolor = global_paper_bg_color
    fig_metric_drift.update_xaxes(showline=True, linewidth=2, gridcolor=px.colors.sequential.Greys[1])
    fig_metric_drift.update_yaxes(showline=True, linewidth=2, gridcolor=px.colors.sequential.Greys[2])
    
#     Drift Chart - 2

    fig_gauge_drift = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = drifted_feats,
                        mode = "gauge+number+delta",
                        title = {'text': ""},
                        delta = {'reference': (int(t2*len_feats)*-1)+(2*drifted_feats)},
                        gauge = {'axis': {'range': [None, len_feats]},
                                 'bar': {'color': px.colors.sequential.Reds[7]},
                                 'steps' : [
                                     {'range': [0,int(t2*len_feats)], 'color': px.colors.sequential.Reds[1]},\
                                     {'range': [int(t2*len_feats),drifted_feats], 'color': px.colors.sequential.Reds[8]},\
                                     {'range': [drifted_feats,len_feats], 'color': px.colors.sequential.Greens[8]}],\
                                 'threshold' : {'line': {'color': 'black', 'width': 3}, 'thickness': 1, 'value': int(t2*len_feats)}}))
    
    fig_gauge_drift.update_layout(font = {'color': "black", 'family': "Arial"})
    fig_gauge_drift.update_layout(height = 400)

    
    def drift_text_gen(drifted_feats,len_feats,t2):
        if drifted_feats == 0:
            text = "*The above barometer does not indicate any drift in the underlying data. Please refer to the metric values as displayed in the above table & comparison plot for better understanding*"
        elif drifted_feats>0:
            if int(t2*len_feats)<drifted_feats:
                text = "*The above barometer indicates that " + str(drifted_feats) + " out of " + str(len_feats) + " (" + str(np.round((100*drifted_feats/len_feats),2)) + "%) attributes were found to be deviating from its base data behaviour. Further, it has computed the overall data health based on data drift across the attributes of interest as well. It can be inferred that based on the safe netting threshold of " + str(t2*100) + "% chosen (approx " + str(int(t2*len_feats)) + ") we could see it to be deviated by " + str(drifted_feats-int(t2*len_feats)) + " attributes which contributes to (-)" + str(np.round(100*(drifted_feats-int(t2*len_feats))/(len_feats),2)) + "% below the mark. Please refer to the metric values as displayed in the above table & comparison plot for better understanding*"
            else:
                text = "*The above barometer indicates that " + str(drifted_feats) + " out of " + str(len_feats) + " (" + str(np.round((100*drifted_feats/len_feats),2)) + "%) attributes were found to be deviating from its base data behaviour. Further, it has computed the overall data health based on data drift across the attributes of interest as well. It can be inferred that based on the safe netting threshold of " + str(t2*100) + "% chosen (approx " + str(int(t2*len_feats)) + ") we could not see any attribute to be deviated below the mark. Please refer to the metric values as displayed in the above table & comparison plot for better understanding*"
        else:
            text = ""
        return text
      
    all_charts = os.listdir(p1)
    all_charts_cat_1 = [x for x in all_charts if "cat_f1" in x]
    all_charts_cat_2 = [x for x in all_charts if "cat_f2" in x]
    all_charts_num_1 = [x for x in all_charts if "num_f1" in x]
    all_charts_num_2 = [x for x in all_charts if "num_f2" in x]
    all_charts_num_3 = [x for x in all_charts if "num_f3" in x]
    all_drift_charts = [x for x in all_charts if "drift_feats" in x]

    all_charts_num_1_,all_charts_num_2_,all_charts_num_3_,all_charts_cat_1_,all_charts_cat_2_,all_drift_charts_ = [],[],[],[],[],[]

    for i in all_charts_num_1:
        
        try:
            col_name = open(ends_with(p1) + i).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
            all_charts_num_1_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + i))),label=col_name))
        except:
            pass

    for j in all_charts_num_2:

        try:
            col_name = open(ends_with(p1) + j).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
            all_charts_num_2_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + j))),label=col_name))
        except:
            pass

    for k in all_charts_num_3:
        try:
            if bool(islabel):
                col_name = open(ends_with(p1) + k).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[2].split(">")[1]
                all_charts_num_3_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + k))),label=col_name))
            else:            
                col_name = open(ends_with(p1) + k).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0][1:]
                all_charts_num_3_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + k))),label=col_name))            
        except:
            pass
        
    for l in all_charts_cat_1:

        col_name = open(ends_with(p1) + l).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
        if col_name == label:
            label_fig = dp.Plot(go.Figure(json.load(open(ends_with(p1) + l))),label=col_name)
        else:
            pass
        all_charts_cat_1_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + l))),label=col_name))

    for m in all_charts_cat_2:

        try:
            col_name = open(ends_with(p1) + m).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
            all_charts_cat_2_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + m))),label=col_name))
        except:
            pass

    for n in all_drift_charts:

        try:
            col_name = open(ends_with(p1) + n).readlines()[0].split("<br>")[0].split(" ")[-1]
            all_drift_charts_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + n))),label=col_name))
        except:
            pass
        
    if bool(islabel):
        l1 = dp.Group(dp.Text("### Attribute to Target Association"),dp.Text("*Bivariate Distribution considering the event captured across different attribute splits (or categories)*"))
        l2 = dp.Group(dp.Select(blocks=[dp.Group(dp.Select(blocks=all_charts_num_2_,type=dp.SelectType.DROPDOWN),label="Numeric"),dp.Group(dp.Select(blocks=all_charts_cat_2_,type=dp.SelectType.DROPDOWN),label="Categorical")],type=dp.SelectType.TABS))
    else:
        l1 = dp.Text("##")
        l2 = dp.Text("##")
        

    global_summary_df = pd.read_csv(ends_with(p2) + "global_summary_df.csv")
    rows_count= int(global_summary_df[global_summary_df.metric.values=="rows_count"].value.values[0])
    catcols_count = int(global_summary_df[global_summary_df.metric.values=="catcols_count"].value.values[0])
    numcols_count = int(global_summary_df[global_summary_df.metric.values=="numcols_count"].value.values[0])

    a1 = "Overall data contains **" + str(rows_count) + "** records across **" + str(numcols_count+catcols_count) + "** attributes of which **" + str(numcols_count) + "** are continuous while **" + str(catcols_count) + "** are categorical."
    if label is None:
        a2 = "There was **no** class variable considered for analysis"
        blank_chart = go.Figure()
        blank_chart.update_layout(autosize=False,width=10,height=10)
        blank_chart.layout.plot_bgcolor = global_plot_bg_color
        blank_chart.layout.paper_bgcolor = global_paper_bg_color
        blank_chart.update_xaxes(visible=False)
        blank_chart.update_yaxes(visible=False)
        label_fig_ = blank_chart
    else:
        a2 = "The class variable considered is **" + str(label) + "** "

        for l in all_charts_cat_1:

            obj_dtls = json.load(open(ends_with(p1) + l))
            col_name = open(ends_with(p1) + l).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]

            if col_name == label:
                text_val = list(list(obj_dtls.values())[0][0].items())[8][1]
                x_val = list(list(obj_dtls.values())[0][0].items())[11][1]
                y_val = list(list(obj_dtls.values())[0][0].items())[13][1]
                label_fig_ = go.Figure(data=[go.Pie(labels=x_val, values=y_val, textinfo='label+percent',
                                             insidetextorientation='radial',pull=[0, 0.1],marker_colors=global_theme)])

                label_fig_.update_traces(textposition='inside', textinfo='percent+label')
                label_fig_.update_layout(legend=dict(orientation="h",x=0.5, yanchor="bottom",xanchor="center"))

    #             label_fig_.update_layout(title_text=str('Pie Chart Distribution for ' +str(col_name.upper())))
                label_fig_.layout.plot_bgcolor = global_plot_bg_color
                label_fig_.layout.paper_bgcolor = global_paper_bg_color


            
#             label_fig_ = dp.Plot(label_fig_,label=col_name)

#     a3 = "Granular data diagnosis is provided at the subsequent sections but from a comprehensive view, listing down the following observations about the attributes - "        

#     z1 = pd.read_csv(ends_with(p2) + "stats_generator_measures_of_dispersion.csv").query("`cov`>1").attribute.values
#     if len(list(z1)) == 0:
#         z1_text = "There was **no** attribute found with a high variance"
#     else:
#         z1_text  =  "High variance : " "**" + ",".join(z1) + "**"

#     a3_1 = "In terms of shape, we've observed the following about the attributes : "
#     a3_2 =  "Positively skewed : " + "**" + ",".join(list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`skewness`>0").attribute.values)) + "**"
#     a3_3 =  "Negatively skewed : "+" **" + ",".join(list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`skewness`<0").attribute.values)) + "**"
#     a3_4 =  "Heavy tail than a normal distribution : " + "**" + ",".join(list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`kurtosis`>3").attribute.values)) + "**"
#     a3_5 =  "Lighter tail than a normal distribution : " + "**" + ",".join(list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`kurtosis`<3").attribute.values)) + "**"

    

#     z2 = pd.read_csv(ends_with(p2) + "stats_generator_measures_of_counts.csv").query("`fill_pct`<0.7").attribute.values
#     if len(list(z2)) == 0:
#         z2_text = "There was **no** attribute found with an alerting fill rate"
#     else:
#         z2_text = "Low fill rates : **" + ",".join(z2) + "**"


#     try:
#         z3 = pd.read_csv(ends_with(p2) + "quality_checker_biasedness_detection.csv").query("`flagged`>0").attribute.values
#     except:
#         z3 = []
#     if len(list(z3)) ==0:
#         z3_text = "High biasedness was **not** observed in any of the attributes"
#     else:
#         z3_text = "High biasedness **" + ",".join(z3) + "**"


#     z4 = pd.read_csv(ends_with(p2) + "quality_checker_outlier_detection.csv").attribute.values
#     if len(list(z4)) ==0:
#         z4_text = "Outlier behaviour was **not** observed in of the attributes"
#     else:
#         z4_text =  "Outlier behavior : " + "**" + ",".join(z4) + "**"

#     a4 = "In terms of Association, we've found that "

#     corr_matrx = pd.read_csv(ends_with(p2) + "/association_evaluator_correlation_matrix.csv")
#     corr_matrx = corr_matrx[list(corr_matrx.attribute.values)]
#     corr_matrx = corr_matrx.where(np.triu(np.ones(corr_matrx.shape),k=1).astype(np.bool))
#     to_drop = [column for column in corr_matrx.columns if any(corr_matrx[column] > corr_threshold)]

#     a5 = "Moderately / High correlation : " + "**" + ",".join(to_drop) + "**"

#     if bool(islabel):
#         a6 = " Significant Attributes for Modelling : " + "**" + ",".join(list(pd.read_csv(ends_with(p2) + "/association_evaluator_IV_calculation.csv").query("`iv`>" + str(iv_threshold)).attribute.values)) + "**"
#     else:
#         a6 = ""
    
    
    a3 = "Granular data diagnosis is provided at the subsequent sections but from a comprehensive view, listing down the following observations about the attributes - "

    try:
        x1 = list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_dispersion.csv").query("`cov`>1").attribute.values)
        if len(x1)>0:
            x1_1 = ["High Variance",x1]
        else:
            x1_1 = ["High Variance",None]
    except:
        x1_1 = ["High Variance",None]

    try:
        x2 =  list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`skewness`>0").attribute.values)
        if len(x2)>0:
            x2_1 = ["Positive Skewness",x2]
        else:
            x2_1 = ["Positive Skewness",None]
    except:
        x2_1 = ["Positive Skewness",None]


    try:
        x3 = list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`skewness`<0").attribute.values)
        if len(x3)>0:
            x3_1 = ["Negative Skewness",x3]
        else:
            x3_1 = ["Negative Skewness",None]
    except:
        x3_1 = ["Negative Skewness",None]

    try:
        x4 =  list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`kurtosis`>3").attribute.values)
        if len(x4)>0:
            x4_1 = ["High Kurtosis",x4]
        else:
            x4_1 = ["High Kurtosis",None]

    except:
        x4_1 =  ["High Kurtosis",None]

    try:
        x5 = list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_shape.csv").query("`kurtosis`<3").attribute.values)
        if len(x5)>0:
            x5_1 = ["Low Kurtosis",x5]
        else:
            x5_1 = ["Low Kurtosis",None]
    except:
        x5_1 = ["Low Kurtosis",None]


    try:
        x6=list(pd.read_csv(ends_with(p2) + "stats_generator_measures_of_counts.csv").query("`fill_pct`<0.7").attribute.values)
        if len(x6)>0:
            x6_1 = ["Low Fill Rates",x6]
        else:
            x6_1 = ["Low Fill Rates",None]
    except:
        x6_1 = ["Low Fill Rates",None]

    try:
        x7 = list(pd.read_csv(ends_with(p2) + "quality_checker_biasedness_detection.csv").query("`flagged`>0").attribute.values)
        if len(x7)>0:
            x7_1 = ["High Biasedness",x7]
        else:
            x7_1 = ["High Biasedness",None]
    except:
        x7_1 = ["High Biasedness",None]

    try:
        x8 = list(pd.read_csv(ends_with(p2) + "quality_checker_outlier_detection.csv").attribute.values)
        if len(x8)>0:
            x8_1 = ["Outliers",x8]
        else:
            x8_1 = ["Outliers",None]
    except:
        x8_1 = ["Outliers",None]


    try:
        corr_matrx = pd.read_csv(ends_with(p2) + "/association_evaluator_correlation_matrix.csv")
        corr_matrx = corr_matrx[list(corr_matrx.attribute.values)]
        corr_matrx = corr_matrx.where(np.triu(np.ones(corr_matrx.shape),k=1).astype(np.bool))
        to_drop = [column for column in corr_matrx.columns if any(corr_matrx[column] > corr_threshold)]
        if len(to_drop)>0:
            x9_1 = ["High Correlation",to_drop]
        else:
            x9_1 = ["High Correlation",None]  
    except:
        x9_1 = ["High Correlation", None]


    try:
        x10 = list(pd.read_csv(ends_with(p2) + "/association_evaluator_IV_calculation.csv").query("`iv`>" + str(iv_threshold)).attribute.values)
        if len(x10)>0:
            x10_1 = ["Significant Attributes",x10]
        else:
            x10_1 ["Significant Attributes",None]
    except:
        x10_1 = ["Significant Attributes",None]


    blank_list_df=[]
    for i in [x1_1,x2_1,x3_1,x4_1,x5_1,x6_1,x7_1,x8_1,x9_1,x10_1]:
        try:
            for j in i[1]:
                blank_list_df.append([i[0],j])
        except:
            pass

    x = pd.DataFrame(blank_list_df,columns=["Metric","Attribute"])
    x['Value'] = '✔'
    x1 = x.pivot(index='Attribute', columns='Metric',values='Value').fillna('✘')
    x1 = x1[["Outliers","Significant Attributes","Positive Skewness","Negative Skewness","High Variance","High Correlation","High Kurtosis","Low Kurtosis"]]
#     # x2 = x1.style.applymap(lambda x: 'background-color : green' if x>x1.iloc[0,0] else '')
#     # x2 = x2.style.applymap(lambda x: '' if x<1 else 1)
#     # x2 = x1
#     # cols_to_show = list(x2.columns)
#     # text_color = []
#     # # n = len(x2)

#     exec_summary_matrix = px.imshow(x1,color_continuous_scale=[px.colors.sequential.Greens[0],px.colors.sequential.Greens[7]],aspect="auto")
#     exec_summary_matrix.update_layout(autosize=True,width=1000,height=700)
#     exec_summary_matrix.update_coloraxes(showscale=False)
#     exec_summary_matrix.layout.plot_bgcolor = global_plot_bg_color
#     exec_summary_matrix.layout.paper_bgcolor = global_paper_bg_color
    
    
    df_var_clus_1 = pd.read_csv(ends_with(p2) + "/association_evaluator_variable_clustering.csv")
    
    num_clus = df_var_clus_1["Cluster"].nunique()
    
#     a4 = "Based on the variable clustering performed, we could observe **" + str(num_clus) + "** cluster groups formed containing a list of homogenous attributes within (details of which can be referred in the respective section) "
    
#     df_var_clus_2 = pd.DataFrame(df_var_clus_1["Cluster"].value_counts().reset_index()).rename(columns = {"index":"Cluster","Cluster":"Attribute_Count"})
    
#     f2 = px.pie(df_var_clus_2, values='Attribute_Count', names='Cluster',color_discrete_sequence=global_theme,hole=0.4)
#     f2.update_traces(textposition='inside', textinfo='percent+label')
#     f2.update_layout(legend=dict(orientation="h",x=0.5, yanchor="bottom",xanchor="center"))
    

    if bool(data_drift_check):
        
#         a5 = "In terms of data drift we observed, **" + str(drifted_feats) + "** out of **" + str(len_feats) + " (" + str(np.round((100*drifted_feats/len_feats),2)) + "**%) attributes were found to be deviating from its base data behaviour. Further computations were done (details can be referred to the data drift section) to gauge the overall data health based on data drift across the attributes of interest."
        a5 = "Data Health based on Data Drift & Data Stability : "
    else:
        a5 = ""

    if np.round(100*(drifted_feats-int(t2*len_feats))/(len_feats),2)>0:
        flag_var = False
    else:
        flag_var = True
    
    
    if bool(data_drift_check):
        dp.Report(dp.HTML('<html><img src="https://plsadaptive.s3.amazonaws.com/eco/images/partners/Y5Zm9VJkHtunAAdmazwsO2lhTwftfYNfk5aP4RJ7.png" style="height:40px;display:flex;margin:auto;float:right"></img></html>'),\
                  dp.Text("# ML-Anovos Report"),\
                  dp.Select(blocks=[
#                dp.Group(dp.Text("# "), dp.Text("1." + a1),dp.Text("2." + a2),label_fig_,dp.Text("3.Attributes Diagnosis View"), dp.Plot(f),dp.Text("4." + a4),dp.Plot(f2,label="Distribution of Clusters"),dp.Text("5." + a5),\
                           dp.Group(dp.Text("# "), dp.Text("1." + a1),dp.Text("2." + a2),label_fig_,dp.Text("3.Attributes Diagnosis View"), dp.DataTable(x1),dp.Text("4." + a5),\
#                         dp.Group(dp.Text("# "), dp.Text("1." + a1),dp.Text("2." + a2),label_fig_, dp.Text("3." + a3_1), dp.Text(" - " + a3_2), dp.Text(" - " + a3_3), dp.Text(" - " + a3_4), dp.Text(" - " + a3_5), dp.Text("4." + a3), dp.Text("  - " + z1_text),dp.Text("  - " + z2_text),dp.Text(" - " + z3_text),dp.Text(" - " + z4_text),dp.Text(" - " + a5),dp.Text(" - " + a6),dp.Text("5." + a7),dp.Plot(f2,label="Distribution of Clusters"),dp.Text("6." + a8),\
             dp.Group(dp.BigNumber(heading = "Attributes Drifted",value = str(str(drifted_feats) + " out of " + str(len_feats))),\
                      dp.BigNumber(heading = "% Drifted", value = str(np.round((100*drifted_feats/len_feats),2))+"%",change=str(np.round(100*(drifted_feats-int(t2*len_feats))/(len_feats),2))+"%", is_upward_change=flag_var),\
                      dp.BigNumber(heading = "Attributes Impacted by Stability", value = str(len(stability_attr)) + " out of " + str(len(total_stability_attr))),\
                      dp.BigNumber(heading= "% Attributes which are Unstable",value = str(np.round(100*len(stability_attr)/len(total_stability_attr),2))+"%"),columns=4),\
             dp.Text("# "),\
             dp.Text("# "),label="Executive Summary"),\
           dp.Group(dp.Select(blocks=[\
                    dp.Group(dp.Group(dp.Text("## "),dp.Text("### Data Dictionary"),dp.DataTable(data_dict)),label="Data Dictionary"),\
                    dp.Group(dp.Text("##"),dp.Text("### Metric Definitions"),dp.DataTable(metric_dict),label="Metric Dictionary")],type=dp.SelectType.TABS),label="Wiki"),\
           dp.Group(
               dp.Text("# "),\
               dp.Text("*Descriptor Statistics summarizes the basic information about the data elements and their individual distribution*"),\
               dp.Text("### Global Summary"),\
               dp.Text("*This section details about the dimension of dataset and the details of attributes across respective data type*"),\
               dp.DataTable(global_summary_df),\
               dp.Text("### Statistics by Metric Type"),\
               dp.Text("*Gives an overall representation of the data anatomy as measured across the different statistical tests*"),\
               dp.Select(blocks=data_analyzer_output(p2,"stats_generator"),\
                         type=dp.SelectType.TABS),\
               dp.Text("# "),\
               dp.Text("### Attribute Visualization"),\
               dp.Text("*Univariate representation of attributes can be studied here through the histogram (continuous) / bar plots (categorical). For a comprehensive view, a restriction is made to the number of buckets based on the user input*"),\
               dp.Group(dp.Select(blocks=\
                       [dp.Group(dp.Select(blocks=all_charts_num_1_,type=dp.SelectType.DROPDOWN),label="Numeric"),\
                        dp.Group(dp.Select(blocks=all_charts_cat_1_,type=dp.SelectType.DROPDOWN),label="Categorical")],\
                         type=dp.SelectType.TABS)),\
               label="Descriptor Statistics"),\
           dp.Group(dp.Select(blocks=[
                    dp.Group(
                        dp.Text("# "),\
                        dp.Text("*Qualitative inspection of Data at a columnar level basis checks like detection of NULL & Invalid records , Biasedness, Outlier observations*"),\
                        data_analyzer_output(p2,"quality_checker")[2][0],\
                        data_analyzer_output(p2,"quality_checker")[2][1],\
                        data_analyzer_output(p2,"quality_checker")[3][0],\
                        data_analyzer_output(p2,"quality_checker")[3][1],\
                        data_analyzer_output(p2,"quality_checker")[4][0],\
                        data_analyzer_output(p2,"quality_checker")[4][1],\
                        data_analyzer_output(p2,"quality_checker")[5][0],\
                        data_analyzer_output(p2,"quality_checker")[5][1],\
                        data_analyzer_output(p2,"quality_checker")[6][0],\
                        dp.Group(data_analyzer_output(p2,"quality_checker")[6][1],dp.Select(blocks=all_charts_num_3_,type=dp.SelectType.DROPDOWN),rows=2),label="Column Level"),\
                    dp.Group(
                        dp.Text("# "),\
                        dp.Text("*Qualitative inspection of Data at a row level basis checks like duplicate entry finding & observation of missing attributes*"),\
                        data_analyzer_output(p2 ,"quality_checker")[0][0],\
                        data_analyzer_output(p2 ,"quality_checker")[0][1],\
                        data_analyzer_output(p2 ,"quality_checker")[1][0],\
                        data_analyzer_output(p2 ,"quality_checker")[1][1],label="Row Level")],\
                   type=dp.SelectType.TABS),\
                    label="Quality Check"),\
           dp.Group(dp.Text("# "),\
                    dp.Text("*Association analysis basis different statistical checks*"),\
                    dp.Text("### Association Matrix"),\
                    dp.Select(blocks=data_analyzer_output(p2,tab_name="association_evaluator")[0],type=dp.SelectType.DROPDOWN),\
                    dp.Text("### "),\
                    dp.Text("### Association Plot"),\
                    dp.Select(blocks=data_analyzer_output(p2,"association_evaluator")[1],type=dp.SelectType.DROPDOWN),\
                    dp.Text("## "),\
                    l1,\
                    l2,\
                    label="Association & Interactions"),\
           dp.Group(dp.Text("# "),\
                    dp.Text("*Useful in capturing the underlying data drift / deviation of the attribute composition as compared to the source data used for analysis*"),\
                    dp.Text("### Data Drift Analyzer"),\
                    dp.Text("*Flagging of Data Drift basis the threshold of " + str(t1) + " chosen across the measured metrics. This means that for a given attribute, if any of the 4 metric have been found to be beyond the threshold, it could have been flagged as a " + "**drifted attribute**" + ". However, in case the user wants to manually customize the threshold, they can use the filter option (refer to a funnel-like logo by hovering on the table header)*"),\
                    dp.DataTable(drift_df),\
                    dp.Text("##"),\
                    dp.Select(blocks=all_drift_charts_,type=dp.SelectType.DROPDOWN),\
                    dp.Text("*Source & Target datasets were compared to see the % deviation. For continuous attributes, the comparison is done at each decile level while for the other type, it's done across individual category level*"),\
                    dp.Text("###  "),\
                    dp.Text("###  "),\
                    dp.Text("### Overall Data Health basis computed Drift Metrices"),\
                    dp.Group(dp.Plot(fig_metric_drift),dp.Plot(fig_gauge_drift),columns=2),\
                    dp.Group(dp.Text("*Representation of Attributes across different computed Drift Metrics*"),dp.Text(drift_text_gen(drifted_feats,len_feats,t2)),columns=2),\
                    dp.Text("## "),\
                    dp.Text("## "),\
                    dp.Text("### Analysis of Data Stability "),\
                    dp.DataTable(df_si),\
                    dp.Text("*Attribute wise Stability Index is computed by analyzing the coefficient of variation of **mean, kurtosis, standard deviation** across different periods. A weighted approach is taken to derive at the final index*"),\
                    dp.Select(blocks=line_chart_list,type=dp.SelectType.DROPDOWN),\
                    dp.Text("**{ Interpretation of Stability Index >> 0-1: Very Unstable, 1-2: Unstable, 2-3: Marginally Stable, 3-3.5: Stable, 3.5-4: Very Stable }**"),\
                    dp.Text("## "),\
                    dp.Text("## "),\
                    label="Data Drift & Data Stability")],\
                   type=dp.SelectType.TABS)).save(ends_with(p3) + "ml_anovos_report.html",open=True)
    else:
            dp.Report(dp.HTML('<html><img src="https://plsadaptive.s3.amazonaws.com/eco/images/partners/Y5Zm9VJkHtunAAdmazwsO2lhTwftfYNfk5aP4RJ7.png" style="height:40px;display:flex;margin:auto;float:right"></img></html>'),\
                  dp.Text("# ML-Anovos Report"),\
                  dp.Select(blocks=[
               dp.Group(dp.Text("# "), dp.Text("1." + a1),dp.Text("2." + a2),label_fig_, dp.Text("3." + a3),dp.Text("  - " + z1_text),dp.Text("  - " + z2_text),dp.Text(" - " + z3_text),dp.Text(" - " + z4_text),dp.Text(" - " + a5),dp.Text(" - " + a6),dp.Text("4." + a7),dp.Plot(f2,label="Distribution of Clusters"),dp.Text("5." + a8),\
             dp.Group(dp.BigNumber(heading = "Count of Attributes Analyzed", value = len_feats),\
             dp.BigNumber(heading = "Count of Attributes found to be Drifted",value = drifted_feats),\
             dp.BigNumber(heading = "Proportion of Drifted Attributes", value = str(np.round((100*drifted_feats/len_feats),2))+"%",change=str(np.round(100*(drifted_feats-int(t2*len_feats))/(len_feats),2))+"%", is_upward_change=flag_var),columns=3),\
             dp.Text("# "),\
             dp.Text("# "),label="Executive Summary"),\
           dp.Group(dp.Select(blocks=[\
                    dp.Group(dp.Group(dp.Text("## "),dp.Text("### Data Dictionary"),dp.DataTable(data_dict)),label="Data Dictionary"),\
                    dp.Group(dp.Text("##"),dp.Text("### Metric Definitions"),dp.DataTable(metric_dict),label="Metric Dictionary")],type=dp.SelectType.TABS),label="Wiki"),\
           dp.Group(
               dp.Text("# "),\
               dp.Text("*Descriptor Statistics summarizes the basic information about the data elements and their individual distribution*"),\
               dp.Text("### Global Summary"),\
               dp.Text("*This section details about the dimension of dataset and the details of attributes across respective data type*"),\
               dp.DataTable(global_summary_df),\
               dp.Text("### Statistics by Metric Type"),\
               dp.Text("*Gives an overall representation of the data anatomy as measured across the different statistical tests*"),\
               dp.Select(blocks=data_analyzer_output(p2,"stats_generator"),\
                         type=dp.SelectType.TABS),\
               dp.Text("# "),\
               dp.Text("### Attribute Visualization"),\
               dp.Text("*Univariate representation of attributes can be studied here through the histogram (continuous) / bar plots (categorical). For a comprehensive view, a restriction is made to the number of buckets based on the user input*"),\
               dp.Group(dp.Select(blocks=\
                       [dp.Group(dp.Select(blocks=all_charts_num_1_,type=dp.SelectType.DROPDOWN),label="Numeric"),\
                        dp.Group(dp.Select(blocks=all_charts_cat_1_,type=dp.SelectType.DROPDOWN),label="Categorical")],\
                         type=dp.SelectType.TABS)),\
               label="Descriptor Statistics"),\
           dp.Group(dp.Select(blocks=[
                    dp.Group(
                        dp.Text("# "),\
                        dp.Text("*Qualitative inspection of Data at a columnar level basis checks like detection of NULL & Invalid records , Biasedness, Outlier observations*"),\
                        data_analyzer_output(p2,"quality_checker")[2][0],\
                        data_analyzer_output(p2,"quality_checker")[2][1],\
                        data_analyzer_output(p2,"quality_checker")[3][0],\
                        data_analyzer_output(p2,"quality_checker")[3][1],\
                        data_analyzer_output(p2,"quality_checker")[4][0],\
                        data_analyzer_output(p2,"quality_checker")[4][1],\
                        data_analyzer_output(p2,"quality_checker")[5][0],\
                        data_analyzer_output(p2,"quality_checker")[5][1],\
                        data_analyzer_output(p2,"quality_checker")[6][0],\
                        dp.Group(data_analyzer_output(p2,"quality_checker")[6][1],dp.Select(blocks=all_charts_num_3_,type=dp.SelectType.DROPDOWN),columns=2),label="Column Level"),\
                    dp.Group(
                        dp.Text("# "),\
                        dp.Text("*Qualitative inspection of Data at a row level basis checks like duplicate entry finding & observation of missing attributes*"),\
                        data_analyzer_output(p2 ,"quality_checker")[0][0],\
                        data_analyzer_output(p2 ,"quality_checker")[0][1],\
                        data_analyzer_output(p2 ,"quality_checker")[1][0],\
                        data_analyzer_output(p2 ,"quality_checker")[1][1],label="Row Level")],\
                   type=dp.SelectType.TABS),\
                    label="Quality Check"),\
           dp.Group(dp.Text("# "),\
                    dp.Text("*Association analysis basis different statistical checks*"),\
                    dp.Text("### Association Matrix"),\
                    dp.Select(blocks=data_analyzer_output(p2,tab_name="association_evaluator")[0],type=dp.SelectType.DROPDOWN),\
                    dp.Text("### "),\
                    dp.Text("### Association Plot"),\
                    dp.Select(blocks=data_analyzer_output(p2,"association_evaluator")[1],type=dp.SelectType.DROPDOWN),\
                    dp.Text("## "),\
                    l1,\
                    l2,\
                    label="Association & Interactions")],\
                   type=dp.SelectType.TABS)).save(ends_with(p3) + "ml_anovos_report.html",open=True)



if __name__ == '__main__':
    #base_path = sys.argv[1]

    main(base_path)
