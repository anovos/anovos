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


global_theme = px.colors.sequential.Peach_r
global_theme_r = px.colors.sequential.Peach_r
global_plot_bg_color = 'rgba(0,0,0,0)'
global_paper_bg_color = 'rgba(0,0,0,0)'

config_file = "configs.yaml"
config_file = open(config_file, 'r')
args = yaml.load(config_file, yaml.SafeLoader)

base_path = args.get('report_gen_final').get('base_path')
list_tabs = args.get('report_gen_inter').get('output_pandas_df',None).get('list_tabs').split(",")
list_tab1 = args.get('report_gen_inter').get('output_pandas_df',None).get('list_tab1').split(",")
list_tab2 = args.get('report_gen_inter').get('output_pandas_df',None).get('list_tab2').split(",")
list_tab3 = args.get('report_gen_inter').get('output_pandas_df',None).get('list_tab3').split(",")
list_tabs_all = [list_tab1,list_tab2,list_tab3]

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
                    fig = px.imshow(df_list_[feats_order],y=feats_order,color_continuous_scale=global_theme_r)
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
                    fig.update_layout(title_text=str("Correlation Plot "))
                    df_list.append(dp.DataTable(df_list_[["attribute"]+feats_order],label=remove_u_score(j)))
                    plot_list.append(dp.Plot(fig,label=remove_u_score(j)))
                    
                    
                elif j == "variable_clustering":
                    df_list_ = pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(j) + ".csv").round(3)
                    fig = px.sunburst(df_list_, path=['Cluster', 'Attribute'], values='RS_Ratio',color_discrete_sequence=global_theme)
                    fig.update_layout(title_text=str("Distribution of homogenous variable across Clusters"))
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
                    fig.update_layout(title_text=str("Variable Clustering Plot "))
                    fig.layout.autosize=True
                    df_list.append(dp.DataTable(df_list_,label=remove_u_score(j)))
                    plot_list.append(dp.Plot(fig,label=remove_u_score(j)))
                
                else:
                    df_list_ = pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(j) + ".csv").round(3)
                    col_nm = [x for x in list(df_list_.columns) if "attribute" not in x]
                    df_list_ = df_list_.sort_values(col_nm[0], ascending=True)
                    fig = px.bar(df_list_,x=col_nm[0],y='attribute',orientation='h',color_discrete_sequence=global_theme_r)
                    fig.layout.plot_bgcolor = global_plot_bg_color
                    fig.layout.paper_bgcolor = global_paper_bg_color
                    fig.update_layout(title_text=str("Representation of " + str(remove_u_score(j))))
                    fig.layout.autosize=True
                    df_list.append(dp.DataTable(df_list_,label=remove_u_score(j)))
                    plot_list.append(dp.Plot(fig,label=remove_u_score(j)))

            return df_list,plot_list
        else:
            
            df_list.append(dp.DataTable(pd.read_csv(ends_with(p2) + str(tab_name)+"_" + str(i) + ".csv").round(3),label=remove_u_score(list_tabs_all[idx][index])))

    return df_list

def main(base_path):
    
    p1 = ends_with(base_path) + ends_with("chart_objects")
    p2 = ends_with(base_path) + ends_with("pandas_df")
    p3 = ends_with(base_path) + ends_with("report")

    Path(p3).mkdir(parents=True, exist_ok=True)

    dd1 = pd.read_csv(ends_with(base_path) + ends_with("data_dict") + "data.csv")
    dd2 = pd.read_csv(ends_with(base_path) + ends_with("pandas_df") + "data_type_df.csv")
    data_dict = dd1.merge(dd2,how="outer",on="Attributes")
    
    metric_dict = pd.read_csv(ends_with(base_path) + ends_with("metric_dict") + "data.csv")
    feature_mp = pd.read_csv(ends_with(base_path) + ends_with("feature_mp") + "data.csv")
    drift_stats = pd.read_csv(ends_with(base_path) + ends_with("pandas_df") + "drift_statistics.csv")
    global_summary_df = pd.read_csv(ends_with(base_path) + ends_with("pandas_df") + "global_summary_df.csv").reindex([2,3,4,1,5,1])

    
    all_charts = os.listdir(p1)
    all_charts_cat_1 = [x for x in all_charts if "cat_f1" in x]
    all_charts_cat_2 = [x for x in all_charts if "cat_f2" in x]
    all_charts_num_1 = [x for x in all_charts if "num_f1" in x]
    all_charts_num_2 = [x for x in all_charts if "num_f2" in x]

    all_charts_num_1_,all_charts_num_2_,all_charts_cat_1_,all_charts_cat_2_ = [],[],[],[]

    for i in all_charts_num_1:
        
        col_name = open(ends_with(p1) + i).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
        all_charts_num_1_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + i))),label=col_name))

    print(all_charts)
    for j in all_charts_num_2:

        col_name = open(ends_with(p1) + j).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
        all_charts_num_2_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + j))),label=col_name))
        
        
    for k in all_charts_cat_1:

        col_name = open(ends_with(p1) + k).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
        all_charts_cat_1_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + k))),label=col_name))

    for l in all_charts_cat_2:

        col_name = open(ends_with(p1) + l).readlines()[0].split("hovertemplate")[1].split(":")[1].split("=")[0].split("''")[0][1:]
        all_charts_cat_2_.append(dp.Plot(go.Figure(json.load(open(ends_with(p1) + l))),label=col_name))


    dp.Report("# ML-Sphere Report",\
       dp.Select(blocks=[
       dp.Group(dp.Select(blocks=[\
                dp.Group(dp.Group(dp.Text("## "),dp.Text("## Data Dictionary & Schema Structure"),dp.DataTable(data_dict)),label="Data Dictionary"),\
                dp.Group(dp.Text("##"),dp.Text("## Metric Definitions"),dp.DataTable(metric_dict),label="Metric Dictionary"),\
                dp.Group(dp.Text("## "),dp.Text("## Recommended Features"),dp.DataTable(feature_mp),label="Feature Marketplace")],type=dp.SelectType.TABS),label="Wiki"),\
       dp.Group(
           dp.Text("## Global Summary"),\
           dp.DataTable(global_summary_df),\
           dp.Text("## Statistics by Metric Type"),\
           dp.Select(blocks=data_analyzer_output(p2,"stats_generator"),\
                     type=dp.SelectType.TABS),\
           dp.Text("## "),\
           dp.Text("## Attribute Visualization"),\
           dp.Group(dp.Select(blocks=\
                   [dp.Group(dp.Select(blocks=all_charts_num_1_,type=dp.SelectType.DROPDOWN),label="Numeric"),\
                    dp.Group(dp.Select(blocks=all_charts_cat_1_,type=dp.SelectType.DROPDOWN),label="Categorical")],\
                     type=dp.SelectType.TABS)),\
           label="Descriptor Statistics"),\
       dp.Group(dp.Select(blocks=[
                dp.Group(
                    data_analyzer_output(p2 ,"quality_checker")[0][0],\
                    data_analyzer_output(p2 ,"quality_checker")[0][1],\
                    data_analyzer_output(p2 ,"quality_checker")[1][0],\
                    data_analyzer_output(p2 ,"quality_checker")[1][1],label="Row Level"),\
                dp.Group(
                    data_analyzer_output(p2,"quality_checker")[2][0],\
                    data_analyzer_output(p2,"quality_checker")[2][1],\
                    data_analyzer_output(p2,"quality_checker")[3][0],\
                    data_analyzer_output(p2,"quality_checker")[3][1],\
                    data_analyzer_output(p2,"quality_checker")[4][0],\
                    data_analyzer_output(p2,"quality_checker")[4][1],\
                    data_analyzer_output(p2,"quality_checker")[5][0],\
                    data_analyzer_output(p2,"quality_checker")[5][1],\
                    data_analyzer_output(p2,"quality_checker")[6][0],\
                    data_analyzer_output(p2,"quality_checker")[6][1],label="Column Level")],\
               type=dp.SelectType.TABS),\
                label="Quality Check"),\
       dp.Group(dp.Text("## Association Matrix"),\
                dp.Select(blocks=data_analyzer_output(p2,tab_name="association_evaluator")[0],type=dp.SelectType.DROPDOWN),\
                dp.Text("## "),\
                dp.Text("## Association Plot"),\
                dp.Select(blocks=data_analyzer_output(p2,"association_evaluator")[1],type=dp.SelectType.DROPDOWN),\
                dp.Text("## "),\
                dp.Text("### Attribute to Target Association"),\
                dp.Group(dp.Select(blocks=
                   [dp.Group(dp.Select(blocks=all_charts_num_2_,type=dp.SelectType.DROPDOWN),label="Numeric"),\
                    dp.Group(dp.Select(blocks=all_charts_cat_2_,type=dp.SelectType.DROPDOWN),label="Categorical")],\
                     type=dp.SelectType.TABS)),\
                label="Association & Interactions"),\
       dp.Group(dp.Text("## Data Drift Analyzer"),dp.DataTable(drift_stats),label="Data Drift")],\
       type=dp.SelectType.TABS)).save(ends_with(p3) + "ml_sphere_report.html",open=True)


if __name__ == '__main__':
    #base_path = sys.argv[1]

    main(base_path)
