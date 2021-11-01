import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from com.mw.ds.shared.spark import *
from com.mw.ds.data_transformer.transformers import outlier_categories, imputation_MMM, attribute_binning
from com.mw.ds.shared.utils import attributeType_segregation
from com.mw.ds.data_analyzer.stats_generator import uniqueCount_computation
from pyspark.sql.window import Window
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import plotly
from plotly.io import write_json
import plotly.express as px
import plotly.graph_objects as go
from plotly.figure_factory import create_distplot
from io import StringIO,BytesIO 
import boto3

global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
global_plot_bg_color = 'rgba(0,0,0,0)'
global_paper_bg_color = 'rgba(0,0,0,0)'
num_cols = []
cat_cols = []

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


def save_stats(idf, master_path, function_name, reread=False,run_type="local"):
    """
    :param idf: input dataframe
    :param master_path: Path to master folder under which all statistics will be saved in a csv file format.
    :param function_name: Function Name for which statistics need to be saved. file name will be saved as csv
    :return: None, dataframe saved
    """

    if run_type == "local":

        Path(master_path).mkdir(parents=True, exist_ok=True)
        idf.toPandas().to_csv(ends_with(master_path) + function_name + ".csv",index=False)

    else:

        bucket_name = master_path.split("//")[1].split("/")[0]
        path_name = master_path.replace(master_path.split("//")[0]+"//"+master_path.split("//")[1].split("/")[0],"")[1:]
        s3_resource = boto3.resource("s3")
        csv_buffer = BytesIO()
        idf.to_csv(csv_buffer,index=False)
        s3_resource.Object(bucket_name, ends_with(path_name) + function_name + ".csv").put(Body=csv_buffer.getvalue())

    if reread:
        odf = spark.read.csv(ends_with(master_path) + function_name + ".csv", header=True, inferSchema=True)
        return odf


def edit_binRange(col):
    try:
        list_col = col.split("-")
        deduped_col = list(set(list_col))
        if len(list_col) != len(deduped_col):
            return deduped_col[0]
        else:
            return col
    except:
        pass


f_edit_binRange = F.udf(edit_binRange, T.StringType())


def binRange_to_binIdx(col, cutoffs_path):
    bin_cutoffs = sqlContext.read.parquet(cutoffs_path).where(F.col('attribute') == col).select('parameters') \
        .rdd.flatMap(lambda x: x).collect()[0]
    bin_ranges = []
    max_cat = len(bin_cutoffs) + 1
    for idx in range(0, max_cat):
        if idx == 0:
            bin_ranges.append("<= " + str(round(bin_cutoffs[idx], 4)))
        elif idx < (max_cat - 1):
            bin_ranges.append(str(round(bin_cutoffs[idx - 1], 4)) + "-" + str(round(bin_cutoffs[idx], 4)))
        else:
            bin_ranges.append("> " + str(round(bin_cutoffs[idx - 1], 4)))
    mapping = spark.createDataFrame(zip(range(1, max_cat + 1), bin_ranges), schema=["bin_idx", col])
    return mapping


def plot_frequency(idf, col, cutoffs_path):
    odf = idf.groupBy(col).count() \
        .withColumn("count_%", 100 * (F.col("count") / F.sum("count").over(Window.partitionBy()))) \
        .withColumn(col, f_edit_binRange(col))

    if col in cat_cols:
        odf_pd = odf.orderBy("count", ascending=False).toPandas().fillna("Missing")
        odf_pd.loc[odf_pd[col] == "others", col] = "others*"

    if col in num_cols:
        mapping = binRange_to_binIdx(col, cutoffs_path)
        odf_pd = odf.join(mapping, col, 'left_outer').orderBy('bin_idx').toPandas().fillna("Missing")


    
    fig = px.bar(odf_pd, x=col, y='count', text=odf_pd['count_%'].apply(lambda x: '{0:1.2f}%'.format(x)),
                 color_discrete_sequence=global_theme)
    fig.update_traces(textposition='outside')
    fig.update_layout(title_text=str('Frequency Distribution for ' + str(col.upper())))
    fig.update_xaxes(type='category')
    # fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    # plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}bar_graph.html")

    return fig


def plot_outlier(idf, col, split_var=None, sample_size=500000):
    idf_sample = idf.select(col).sample(False, min(1.0, float(sample_size) / idf.count()), 0)
    idf_sample.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    idf_imputed = imputation_MMM(idf_sample)
    idf_pd = idf_imputed.toPandas()
    fig = px.violin(idf_pd, y=col, color=split_var, box=True, points="outliers",
                    color_discrete_sequence=[global_theme_r[8], global_theme_r[4]])
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    fig.update_layout(legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center"))

    return fig


def plot_eventRate(idf, col, label_col, event_label, cutoffs_path):
    event_label = str(event_label)
    class_cats = idf.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()

    odf = idf.groupBy(col).pivot(label_col).count() \
        .fillna(0, subset=class_cats) \
        .withColumn("event_rate", 100 * (F.col(event_label) / (F.col(class_cats[0]) + F.col(class_cats[1])))) \
        .withColumn("attribute_name", F.lit(col)) \
        .withColumn(col, f_edit_binRange(col))

    if col in cat_cols:
        odf_pd = odf.orderBy("event_rate", ascending=False).toPandas()
        odf_pd.loc[odf_pd[col] == "others", col] = "others*"

    if col in num_cols:
        mapping = binRange_to_binIdx(col, cutoffs_path)
        odf_pd = odf.join(mapping, col, 'left_outer').orderBy('bin_idx').toPandas()

    

    fig = px.bar(odf_pd, x=col, y='event_rate', text=odf_pd['event_rate'].apply(lambda x: '{0:1.2f}%'.format(x)),
                 color_discrete_sequence=global_theme)
    fig.update_traces(textposition='outside')
    fig.update_layout(title_text=str(
        'Event Rate Distribution for ' + str(col.upper()) + str(" [Target Variable : " + str(event_label) + str("]"))))
    fig.update_xaxes(type='category')
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    # plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}feat_analysis_label.html")

    return fig


def plot_comparative_drift(idf, source, col, cutoffs_path):
    odf = idf.groupBy(col).agg((F.count(col) / idf.count()).alias('countpct_target')).fillna(np.nan, subset=[col])

    if col in cat_cols:
        odf_pd = odf.join(source.withColumnRenamed("p", "countpct_source").fillna(np.nan, subset=[col]), col,
                          "full_outer") \
            .orderBy("countpct_target", ascending=False).toPandas()

    if col in num_cols:
        mapping = binRange_to_binIdx(col, cutoffs_path)
        odf_pd = odf.join(mapping, col, 'left_outer').fillna(np.nan, subset=['bin_idx']) \
            .join(source.fillna(np.nan, subset=[col]).select(F.col(col).alias('bin_idx'),
                                                             F.col("p").alias("countpct_source")), 'bin_idx',
                  "full_outer") \
            .orderBy('bin_idx').toPandas()


    odf_pd.fillna({col: 'Missing', 'countpct_source': 0, 'countpct_target': 0}, inplace=True)
    odf_pd['%_diff'] = (((odf_pd['countpct_target'] / odf_pd['countpct_source']) - 1) * 100)
    fig = go.Figure()
    fig.add_bar(y=list(odf_pd.countpct_source.values), x=odf_pd[col], name="source", marker=dict(color=global_theme))
    fig.update_traces(overwrite=True, marker={"opacity": 0.7})
    fig.add_bar(y=list(odf_pd.countpct_target.values), x=odf_pd[col], name="target",
                text=odf_pd['%_diff'].apply(lambda x: '{0:0.2f}%'.format(x)), marker=dict(color=global_theme))
    fig.update_traces(textposition='outside')
    fig.update_layout(paper_bgcolor=global_paper_bg_color, plot_bgcolor=global_plot_bg_color, showlegend=False)
    fig.update_layout(title_text=str('Drift Comparison for ' + col + '<br><sup>(L->R : Source->Target)</sup>'))
    fig.update_traces(marker=dict(color=global_theme))
    fig.update_xaxes(type='category')
    fig.add_trace(go.Scatter(x=odf_pd[col], y=odf_pd.countpct_target.values, mode='lines+markers',
                             line=dict(color=px.colors.qualitative.Antique[10], width=3, dash='dot')))
    fig.update_layout(xaxis_tickfont_size=14, yaxis=dict(title='frequency', titlefont_size=16, tickfont_size=14))

    return fig


def charts_to_objects(idf, list_of_cols='all', drop_cols=[], label_col=None, event_label=1,
                      bin_method="equal_range", bin_size=10, coverage=1.0,
                      drift_detector=False, source_path="NA", master_path='',run_type="local"):
    global num_cols
    global cat_cols
    import timeit
    
    
    start = timeit.default_timer()
    
    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols,cat_cols,other_cols = attributeType_segregation(idf.select(list_of_cols))

    idf_cleaned = outlier_categories(idf, list_of_cols=cat_cols, coverage=coverage, max_category=bin_size)
    
    if drift_detector:
        encoding_model_exists = True
    else:
        encoding_model_exists = False
    idf_encoded = attribute_binning(idf_cleaned, list_of_cols=num_cols, method_type=bin_method, bin_size=bin_size, 
                                    bin_dtype="categorical", pre_existing_model=encoding_model_exists, 
                                    model_path=source_path+"/drift_statistics", output_mode='append')

    cutoffs_path = source_path+"/drift_statistics/attribute_binning"
    idf_encoded.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    
    
    if run_type == "local":

        Path(master_path).mkdir(parents=True, exist_ok=True)
        for idx, col in enumerate(list_of_cols):
            
            if col in cat_cols:
                f = plot_frequency(idf_encoded,col,cutoffs_path)
                f.write_json(ends_with(master_path) + "freqDist_" + col)

                if label_col:
                    f = plot_eventRate(idf_encoded,col,label_col,event_label,cutoffs_path)
                    f.write_json(ends_with(master_path) + "eventDist_" + col)

                if drift_detector:
                    frequency_path = source_path+"/drift_statistics/frequency_counts/" + col
                    idf_source = spark.read.csv(frequency_path, header=True, inferSchema=True)
                    f = plot_comparative_drift(idf_encoded,idf_source,col,cutoffs_path)
                    f.write_json(ends_with(master_path) + "drift_" + col)
            
            if col in num_cols:
                f = plot_outlier(idf,col,split_var=None)
                f.write_json(ends_with(master_path) + "outlier_" + col)
                f = plot_frequency(idf_encoded.drop(col).withColumnRenamed(col+"_binned",col),col,cutoffs_path)
                f.write_json(ends_with(master_path) + "freqDist_" + col)

                if label_col:
                    f = plot_eventRate(idf_encoded.drop(col).withColumnRenamed(col+"_binned",col),col,label_col,event_label,cutoffs_path)
                    f.write_json(ends_with(master_path) + "eventDist_" + col)

                if drift_detector:
                    frequency_path = source_path+"/drift_statistics/frequency_counts/" + col
                    idf_source = spark.read.csv(frequency_path, header=True, inferSchema=True)
                    f = plot_comparative_drift(idf_encoded.drop(col).withColumnRenamed(col+"_binned",col),idf_source,col,cutoffs_path)
                    f.write_json(ends_with(master_path) + "drift_" + col)

        pd.DataFrame(idf.dtypes,columns=["attribute","data_type"]).to_csv(ends_with(master_path) + "data_type.csv",index=False)

    else:

        bucket_name = master_path.split("//")[1].split("/")[0]
        path_name = master_path.replace(master_path.split("//")[0]+"//"+master_path.split("//")[1].split("/")[0],"")[1:]
        s3_resource = boto3.resource("s3")

        x = pd.DataFrame(idf.dtypes).reset_index()
        x = x.rename(columns = {'index' : 'attribute', 0:'data_type'})
        csv_buffer = BytesIO()
        x.to_csv(csv_buffer,index=False)
        s3_resource.Object(bucket_name, ends_with(path_name) + "data_type.csv").put(Body=csv_buffer.getvalue())

        for idx, col in enumerate(list_of_cols):
            
            if col in cat_cols:
                f = plot_frequency(idf_encoded,col,cutoffs_path)
                s3_resource.Object(bucket_name, ends_with(path_name) + "freqDist_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))

                if label_col:
                    f = plot_eventRate(idf_encoded,col,label_col,event_label,cutoffs_path)
                    s3_resource.Object(bucket_name, ends_with(path_name) + "eventDist_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))

                if drift_detector:
                    frequency_path = source_path+"/drift_statistics/frequency_counts/" + col
                    idf_source = spark.read.csv(frequency_path, header=True, inferSchema=True)
                    f = plot_comparative_drift(idf_encoded,idf_source,col,cutoffs_path)
                    s3_resource.Object(bucket_name, ends_with(path_name) + "drift_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))
            
            if col in num_cols:
                f = plot_outlier(idf,col,split_var=None)
                s3_resource.Object(bucket_name, ends_with(path_name) + "outlier_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))
                f = plot_frequency(idf_encoded.drop(col).withColumnRenamed(col+"_binned",col),col,cutoffs_path)
                s3_resource.Object(bucket_name, ends_with(path_name) + "freqDist_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))

                if label_col:
                    f = plot_eventRate(idf_encoded.drop(col).withColumnRenamed(col+"_binned",col),col,label_col,event_label,cutoffs_path)
                    s3_resource.Object(bucket_name, ends_with(path_name) + "eventDist_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))

                if drift_detector:
                    frequency_path = source_path+"/drift_statistics/frequency_counts/" + col
                    idf_source = spark.read.csv(frequency_path, header=True, inferSchema=True)
                    f = plot_comparative_drift(idf_encoded.drop(col).withColumnRenamed(col+"_binned",col),idf_source,col,cutoffs_path)
                    s3_resource.Object(bucket_name, ends_with(path_name) + "drift_" + col).put(Body=(bytes(json.dumps(f.to_json()).encode("UTF-8"))))

