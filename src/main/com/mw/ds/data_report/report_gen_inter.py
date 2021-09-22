import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
import warnings
from com.mw.ds.shared.spark import *
from com.mw.ds.shared.utils import *
from com.mw.ds.data_transformer.transformers import *
import plotly
from plotly.io import write_json
import plotly.express as px
from pathlib import Path
import pandas as pd
global_theme = px.colors.sequential.Peach
global_theme_r = px.colors.sequential.Peach_r
global_plot_bg_color = 'rgba(0,0,0,0)'
global_paper_bg_color = 'rgba(0,0,0,0)'

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


def outlier_catfeats(idf, list_of_cols, coverage, max_category=50, pre_existing_model=False, model_path="NA", output_mode='replace', print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: List of columns for outlier treatment
    coverage: Minimum % of rows mapped to actual category name and rest will be mapped to others
    max_category: Even if coverage is less, only these many categories will be mapped to actual name and rest to others
    pre_existing_model: outlier value for each feature. True if model files exists already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  param NA means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Dataframe after outlier treatment
    '''
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|') if x.strip() in idf.columns]
    else:
        list_of_cols = [e for e in list_of_cols if e in idf.columns]
        
    if output_mode not in ('replace','append'):
        raise TypeError('Invalid input for output_mode')
    if len(list_of_cols) == 0:
        raise TypeError('Invalid input for Column(s)')
    
    if pre_existing_model == True:
        df_model = sqlContext.read.csv(model_path + "/outlier_catfeats", header=True, inferSchema=True)
    else:
        for index, i in enumerate(list_of_cols):
            from pyspark.sql.window import Window
            window = Window.partitionBy().orderBy(F.desc('count_pct'))
            df_cats = idf.groupBy(i).count().dropna()\
                         .withColumn('count_pct', F.col('count')/F.sum('count').over(Window.partitionBy()))\
                         .withColumn('rank', F.rank().over(window))\
                         .withColumn('cumu', F.sum('count_pct').over(window.rowsBetween(Window.unboundedPreceding, 0)))\
                         .withColumn('lag_cumu', F.lag('cumu').over(window)).fillna(0)\
                         .where(~((F.col('cumu') >= coverage) & (F.col('lag_cumu') >= coverage)))\
                         .where(F.col('rank') <= max_category)\
                         .select(F.lit(i).alias('feature'), F.col(i).alias('parameters'))
                        
            if index == 0:
                df_model = df_cats
            else:
                df_model = df_model.union(df_cats)
    
    odf = idf
    for i in list_of_cols:
        parameters = df_model.where(F.col('feature') == i).select('parameters').rdd.flatMap(lambda x:x).collect()
        if output_mode == 'replace':
            odf = odf.withColumn(i, F.when((F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)).otherwise("others"))
        else:
            odf = odf.withColumn(i + "_outliered", F.when((F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)).otherwise("others"))
        
    # Saving model File if required
    if (pre_existing_model == False) & (model_path != "NA"):
        df_model.repartition(1).write.csv(model_path + "/outlier_catfeats", header=True, mode='overwrite')
        
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_outliered") for i in list_of_cols]
        uniquecount_computation(idf, list_of_cols).select('feature', F.col("unique_values").alias("uniqueValues_before")).show(len(list_of_cols))
        uniquecount_computation(odf, output_cols).select('feature', F.col("unique_values").alias("uniqueValues_after")).show(len(list_of_cols))
         
    return odf


def plot_gen_hist_bar(idf,col,cov=None,max_cat=50,bin_type=None):
    
    import plotly.express as px
    from plotly.figure_factory import create_distplot
    num_cols,cat_cols,other_cols = featureType_segregation(idf)
    

    #try:
    if col in cat_cols:
    
        idf = outlier_catfeats(idf,list_of_cols=col,coverage=cov,max_category=max_cat)\
                              .groupBy(col).count()\
                              .withColumn("count_%",100*(F.col("count")/F.sum("count").over(Window.partitionBy())))\
                              .orderBy("count",ascending=False)\
                              .toPandas()
        
        fig = px.bar(idf,x=col,y='count',text=idf['count_%'].apply(lambda x: '{0:1.2f}%'.format(x)),color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Bar Plot Distribution for ' +str(col.upper())))
#         fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

    elif col in num_cols:

        idf = feature_binning(idf,list_of_cols=col,method_type=bin_type,bin_size=max_cat)\
                             .groupBy(col).count()\
                             .withColumn("count_%",100*(F.col("count")/F.sum("count").over(Window.partitionBy())))\
                             .orderBy("count",ascending=False)\
                             .toPandas()
        
        fig = px.bar(idf,x=col,y='count',text=idf['count_%'].apply(lambda x: '{0:1.2f}%'.format(x)),color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Bar Plot Distribution for ' +str(col.upper())))
#         fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})


    else:
        pass

    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color

#       plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}bar_graph.html")
    
    
    return fig

    #except:
    #    return ""


def plot_gen_boxplot(idf,cont_col,cat_col=None,color_by=None,cov=None,max_cat=50,threshold=500000):
    
    import plotly.express as px
    from plotly.figure_factory import create_distplot
    
    count_df = idf.count()
    
    if (cat_col is not None) and (cont_col is not None):
        
        if count_df > threshold:
            
            group_dist = dict(sub.values() for sub in \
                               idf.groupBy(cat_col).count().fillna("NA_Missing",subset=cat_col)\
                                 .withColumn("count_%",\
                                 int(threshold)*(F.col("count")/F.sum("count")\
                                 .over(Window.partitionBy()))/F.col("count"))\
                                 .select(cat_col,"count_%").toPandas().to_dict('r'))
            
            idf = idf.fillna("NA_Missing",subset=cat_col).sampleBy(cat_col,fractions=group_dist,seed=common_seed)
        
        else:
            idf = idf.fillna("NA_Missing",subset=cat_col)
        
        idf = outlier_catfeats(idf,list_of_cols=cat_col,coverage=cov,max_category=max_cat).toPandas()
        
        fig = px.box(idf,x=cat_col,y=cont_col,color=color_by,color_discrete_sequence=global_theme)
#         fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Box Plot Analysis for ' +str(cont_col.upper()) + str(" across : " + str(cat_col.upper()))))
            
    elif (cat_col is None) and (cont_col is not None):
        
        if count_df > threshold:
            
            group_dist = dict(sub.values() for sub in \
                           idf.groupBy(cont_col).count().fillna("NA_Missing")\
                             .withColumn("count_%",int(threshold)*(F.col("count")/F.sum("count")\
                             .over(Window.partitionBy()))/F.col("count"))\
                             .select(cont_col,"count_%").toPandas().to_dict('r'))
            idf = idf.fillna("NA_Missing",subset=cont_col).sampleBy(cont_col,fractions=group_dist,seed=common_seed)
            
        else:
            pass
        
        idf = idf.select(cont_col).toPandas()
        
        fig = px.box(idf,y=cont_col,color=color_by,color_discrete_sequence=global_theme)
#         fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Box Plot Analysis for ' +str(cont_col.upper())))
        
    else:
        pass
    
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
#     plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}box_plot.html")
    
    return fig


def plot_gen_feat_analysis_label(idf,col,label,event_class,max_cat=None,bin_type=None):
    
    import plotly.express as px
    from plotly.figure_factory import create_distplot
    num_cols,cat_cols,other_cols = featureType_segregation(idf)
    
    event_class = str(event_class)
    
#     file_name_ = str(col) + "_" + str(event_class) + "_" + str(max_cat) + "_" + str(bin_type) + "_"
    
    class_cats = idf.select(label).distinct().rdd.flatMap(lambda x: x).collect()
    if col in cat_cols:
        idf = idf.groupBy(col).pivot(label).count()\
                .fillna(0,subset=class_cats)\
                .withColumn("event_rate",100*(F.col(event_class)/(F.col(class_cats[0])+F.col(class_cats[1]))))\
                .withColumn("feature_name",F.lit(col))\
                .orderBy("event_rate",ascending=False)\
                .toPandas()
                
        fig = px.bar(idf,x=col,y='event_rate',text=idf['event_rate'].apply(lambda x: '{0:1.2f}%'.format(x)),color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Event Rate Distribution for ' +str(col.upper()) + str(" [Target Variable : " + str(event_class) + str("]"))))
    
    elif col in num_cols:
        
        idf = feature_binning(idf, method_type=bin_type, bin_size=max_cat, list_of_cols=col)
        
        idf = idf.groupBy(col).pivot(label).count()\
                 .fillna(0,subset=class_cats)\
                 .withColumn("event_rate",100*(F.col(event_class)/(F.col(class_cats[0])+F.col(class_cats[1]))))\
                 .withColumn("feature_name",F.lit(col))\
                 .orderBy("event_rate",ascending=False)\
                 .toPandas()
        
        fig = px.bar(idf,x=col,y='event_rate',text=idf['event_rate'].apply(lambda x: '{0:1.2f}%'.format(x)),color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Event Rate Distribution for ' +str(col.upper()) + str(" [Target Variable : " + str(event_class) + str("]"))))

    else:
        pass
    
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
#     plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}feat_analysis_label.html")

                          
    return fig


def plot_gen_variable_clustering(idf):
    
    import plotly.express as px
    from plotly.figure_factory import create_distplot
    
    fig = px.sunburst(idf, path=['Cluster', 'feature'], values='RS_Ratio',color_discrete_sequence=global_theme)
    fig.update_layout(title_text=str("Distribution of homogenous variable across Clusters"))
    
#     plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}plot_sunburst.html")

    return fig


def plot_gen_dist(idf,col,threshold=500000, rug_chart=False):
    
    import plotly.figure_factory as ff
#     file_name_ = str("distplot") + "_" + str(col) + "_"
    group_label = [col]
    count_df = idf.count()
    
    if col in num_cols:
        
        if count_df > threshold:
            group_dist = dict(sub.values() for sub in \
                               idf.groupBy(col).count().fillna("NA_Missing",subset=col)\
                                 .withColumn("count_%",\
                                 int(threshold)*(F.col("count")/F.sum("count")\
                                 .over(Window.partitionBy()))/F.col("count"))\
                                 .select(col,"count_%").toPandas().to_dict('r'))
            
            idf = idf.select(col).dropna().sampleBy(col,fractions=group_dist,seed=common_seed)
        
        else:
            idf = idf.select(col).dropna()
        
        idf = idf.select(col).rdd.flatMap(lambda x: x).collect()
        fig = ff.create_distplot([idf],group_labels=group_label,show_rug=rug_chart,colors = global_theme)
        
        fig.layout.plot_bgcolor = global_plot_bg_color
        fig.layout.paper_bgcolor = global_paper_bg_color
        fig.update_layout(title_text=str("Distribution Plot " + str(col)))
        
#         plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}distplot.html")
        return fig

    else:
        return 0


def num_cols_chart_list(df,max_cat=10,bin_type="equal_frequency",output_path=None):
    
    num_cols_chart = []
    num_cols,cat_cols,other_cols = featureType_segregation(df)
    for index,i in enumerate(num_cols):
        
        f = plot_gen_hist_bar(idf=df,col=i,max_cat=max_cat,bin_type=bin_type)
        if output_path is None:
            f.write_json("fig_num_f1_" + str(index))
        else:
            f.write_json(ends_with(output_path) + "fig_num_f1_" + str(index))
        
        num_cols_chart.append(f)



def cat_cols_chart_list(df,id_col,max_cat=10,cov=0.9,output_path=None):

    cat_cols_chart = []
    num_cols,cat_cols,other_cols = featureType_segregation(df)
    for index,i in enumerate(cat_cols):
        if i !=id_col:
            f=plot_gen_hist_bar(idf=df,col=i,max_cat=max_cat,cov=cov)
            if output_path is None:
                f.write_json("fig_cat_f1_" + str(index))
            else:
                f.write_json(ends_with(output_path) + "fig_cat_f1_" + str(index))
                
            cat_cols_chart.append(f)
            
        else:
            pass


def num_cols_int_chart_list(df,label,event_class,bin_type="equal_range",max_cat=10,output_path=None):

    num_cols_int_chart = []
    num_cols,cat_cols,other_cols = featureType_segregation(df)
    for index,i in enumerate(num_cols):
        f = plot_gen_feat_analysis_label(idf=df,col=i,label=label,event_class=event_class,bin_type=bin_type,max_cat=max_cat)
        if output_path is None:
            f.write_json("fig_num_f2_" + str(index))
        else:
            f.write_json(ends_with(output_path) + "fig_num_f2_" + str(index))
        num_cols_int_chart.append(f)
    return num_cols_int_chart


def cat_cols_int_chart_list(df,id_col,label,event_class,output_path=None):
    
    cat_cols_int_chart = []
    num_cols,cat_cols,other_cols = featureType_segregation(df)
    for index,i in enumerate(cat_cols):
        if i!=id_col:
            f = plot_gen_feat_analysis_label(idf=df,col=i,label=label,event_class=event_class)
            if output_path is None:

                f.write_json("fig_cat_f2_" + str(index))
            else:
                f.write_json(ends_with(output_path) + "fig_cat_f2_" + str(index))
            cat_cols_int_chart.append(f)
        else:
            pass



def charts_to_objects(idf,id_col=None,max_cat=10,label=None,event_class=None,chart_output_path=None):
    
    Path(chart_output_path).mkdir(parents=True, exist_ok=True)
    
    num_cols_chart_list(idf,output_path=chart_output_path)
    cat_cols_chart_list(idf,id_col=id_col,output_path=chart_output_path)
    num_cols_int_chart_list(idf,label=label,event_class=event_class,output_path=chart_output_path)
    cat_cols_int_chart_list(idf,id_col=id_col, label=label,event_class=event_class,output_path=chart_output_path)


def output_pandas_df(idf,input_path,pandas_df_output_path, list_tabs, list_tab1, list_tab2, list_tab3):
    
    Path(pandas_df_output_path).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(idf.dtypes,columns=["Attributes","Datatype"]).to_csv(ends_with(pandas_df_output_path) + "data_type_df.csv",index=False)

    list_tabs_arr = list_tabs.split(",")
    list_tab1_arr = list_tab1.split(",")
    list_tab2_arr = list_tab2.split(",")
    list_tab3_arr = list_tab3.split(",")
    list_tabs_all = [list_tab1_arr,list_tab2_arr,list_tab3_arr]


    for index,i in enumerate(list_tabs_arr):
        for j in list_tabs_all[index]:
            if i == "stats_generator":
                spark.read.parquet(ends_with(input_path) + ends_with("data_analyzer") + ends_with(i) + ends_with(j)).toPandas().to_csv(ends_with(pandas_df_output_path) + i + "_" + j + ".csv",index=False)
            else:
                spark.read.parquet(ends_with(input_path) + ends_with("data_analyzer") + ends_with(i) + ends_with(j) + ends_with("stats")).toPandas().to_csv(ends_with(pandas_df_output_path) + i + "_" + j + ".csv",index=False)
    
    spark.read.parquet(ends_with(input_path) + ends_with("data_analyzer") +  ends_with("stats_generator") + ends_with("global_summary")).toPandas().to_csv(ends_with(pandas_df_output_path) + "global_summary_df.csv",index=False) 

def data_drift(read_file,pandas_df_output_path):
    
    Path(pandas_df_output_path).mkdir(parents=True, exist_ok=True)
    read_dataset(**read_file).toPandas().to_csv(ends_with(pandas_df_output_path) + "drift_statistics.csv",index=False)



