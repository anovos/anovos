from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyspark
from com.mw.ds.data_ingest.data_ingest import *
from com.mw.ds.data_transformer.transformers import *
from com.mw.ds.shared.spark import *
from com.mw.ds.shared.utils import *
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

global_theme = px.colors.sequential.Plasma
global_theme_r = px.colors.sequential.Plasma_r
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


def remove_dups(col):
    try:
        list_col = col.split("-")
        deduped_col = list(set(list_col))
        if len(list_col) != len(deduped_col):
            return deduped_col[0]
        else:
            return col
    except:
        pass


f_remove_dups = F.udf(remove_dups, T.StringType())


def processed_df(df, drop_cols_viz):
    num_cols, cat_cols, other_cols = attributeType_segregation(df)

    df_ = df.select(num_cols + cat_cols)
    zero_var_col = []
    for i in df_.columns:
        x = df_.select(i).distinct().count()
        if x == 1:
            zero_var_col.append(i)
        else:
            pass
    df_ = df_.drop(*zero_var_col + drop_cols_viz)
    return df_


def range_generator(df, col_orig, col_binned):
    range_table = df.groupBy(col_binned) \
        .agg(F.round(F.min(col_orig), 2).alias("min"), F.round(F.max(col_orig), 2).alias("max")) \
        .withColumn("range", F.concat(F.col("min"), F.lit("-"), F.col("max"))) \
        .select(col_binned, "range")

    df_ = df.join(range_table, col_binned, "left_outer") \
        .withColumnRenamed(col_binned, str(col_orig + "_binning_number")) \
        .withColumnRenamed("range", col_binned)

    return df_


def feature_binning(idf, method_type, bin_size, list_of_cols, id_col="id", label_col="label", pre_existing_model=False,
                    model_path="NA", output_mode="replace", print_impact=False, output_type="number"):
    '''
    idf: Input Dataframe
    method_type: equal_frequency, equal_range
    bin_size: No of bins
    list_of_cols: all numerical (except ID & Label) or list of columns (in list format or string separated by |)
    id_col, label_col: Excluding ID & Label columns from binning
    pre_existing_model: True if mapping values exists already, False Otherwise. 
    model_path: If pre_existing_model is True, this argument is path for imputation model. 
                  If pre_existing_model is False, this argument can be used for saving the model. 
                  Default "NA" means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Binned Dataframe
    '''

    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = featureType_segregation(idf)
        list_of_cols = [e for e in num_cols if e not in (id_col, label_col)]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|') if
                        ((x.strip() in idf.columns) & (x.strip() not in (id_col, label_col)))]
    if isinstance(list_of_cols, list):
        list_of_cols = [e for e in list_of_cols if ((e in idf.columns) & (e not in (id_col, label_col)))]

    if method_type not in ("equal_frequency", "equal_range"):
        raise TypeError('Invalid input for method_type')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')
    if len(list_of_cols) == 0:
        raise TypeError('Invalid input for Column(s)')

    odf = idf
    for col in list_of_cols:

        if method_type == "equal_frequency":
            from pyspark.ml.feature import QuantileDiscretizer
            if pre_existing_model == True:
                discretizerModel = QuantileDiscretizer.load(model_path + "/feature_binning/" + col)
            else:
                discretizer = QuantileDiscretizer(numBuckets=bin_size, inputCol=col, outputCol=col + "_binned")
                discretizerModel = discretizer.fit(odf)
            if output_type == "number":
                odf = discretizerModel.transform(odf)
            else:
                odf = range_generator(discretizerModel.transform(odf), col, col + "_binned")

            if (pre_existing_model == False) & (model_path != "NA"):
                discretizerModel.write().overwrite().save(model_path + "/feature_binning/" + col)
        else:

            from pyspark.ml.feature import Bucketizer
            if pre_existing_model == True:
                bucketizer = Bucketizer.load(model_path + "/feature_binning/" + col)
            else:
                max_val = idf.select(F.col(col)).groupBy().max().rdd.flatMap(lambda x: x).collect()[0]
                min_val = idf.select(F.col(col)).groupBy().min().rdd.flatMap(lambda x: x).collect()[0]
                bin_width = (max_val - min_val) / bin_size
                bin_cutoff = [-float("inf")]
                for i in range(1, bin_size):
                    bin_cutoff.append(min_val + i * bin_width)
                bin_cutoff.append(float("inf"))
                bucketizer = Bucketizer(splits=bin_cutoff, inputCol=col, outputCol=col + "_binned")

                if (pre_existing_model == False) & (model_path != "NA"):
                    bucketizer.write().overwrite().save(model_path + "/feature_binning/" + col)
            if output_type == "number":
                odf = bucketizer.transform(odf)
            else:
                odf = range_generator(bucketizer.transform(odf), col, col + "_binned")

    if output_mode == 'replace':
        for col in list_of_cols:
            odf = odf.drop(col).withColumnRenamed(col + "_binned", col)

    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_binned") for i in list_of_cols]
        c(odf, output_cols).show(len(output_cols))
    return odf


def plot_gen_hist_bar(idf, col, cov=None, max_cat=50, bin_type=None):
    import plotly.express as px
    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    # try:
    if col in cat_cols:

        idf = outlier_categories(idf, list_of_cols=col, coverage=cov, max_category=max_cat) \
            .groupBy(col).count() \
            .withColumn("count_%", 100 * (F.col("count") / F.sum("count").over(Window.partitionBy()))) \
            .withColumn(col, f_remove_dups(col)) \
            .orderBy("count", ascending=False) \
            .toPandas().fillna("Missing")

        fig = px.bar(idf, x=col, y='count', text=idf['count_%'].apply(lambda x: '{0:1.2f}%'.format(x)),
                     color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Bar Plot for ' + str(col.upper())))
    #         fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

    elif col in num_cols:

        idf = feature_binning(idf, list_of_cols=col, method_type=bin_type, bin_size=max_cat, output_type="string") \
            .groupBy(str(col + "_binning_number"), col).count() \
            .withColumn("count_%", 100 * (F.col("count") / F.sum("count").over(Window.partitionBy()))) \
            .withColumn(col, f_remove_dups(col)) \
            .orderBy(str(col + "_binning_number"), ascending=True) \
            .toPandas().fillna("Missing")

        fig = px.bar(idf, x=col, y='count', text=idf['count_%'].apply(lambda x: '{0:1.2f}%'.format(x)),
                     color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Histogram for ' + str(col.upper())))
        fig.update_xaxes(type='category')
    #         fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

    else:
        pass

    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color

    #       plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}bar_graph.html")

    return fig


def plot_gen_boxplot(idf, cont_col, cat_col=None, color_by=None, cov=None, max_cat=50, threshold=500000):
    import plotly.express as px

    count_df = idf.count()

    if (cat_col is not None) and (cont_col is not None):

        if count_df > threshold:

            group_dist = dict(sub.values() for sub in \
                              idf.groupBy(cat_col).count().fillna("Missing", subset=cat_col) \
                              .withColumn("count_%", \
                                          int(threshold) * (F.col("count") / F.sum("count") \
                                                            .over(Window.partitionBy())) / F.col("count")) \
                              .select(cat_col, "count_%").toPandas().to_dict('r'))

            idf = idf.fillna("Missing", subset=cat_col).sampleBy(cat_col, fractions=group_dist, seed=common_seed)

        else:
            idf = idf.fillna("Missing", subset=cat_col)

        idf = outlier_categories(idf, list_of_cols=cat_col, coverage=cov, max_category=max_cat).toPandas()

        fig = px.box(idf, x=cat_col, y=cont_col, color=color_by, color_discrete_sequence=global_theme)
        #         fig.update_traces(textposition='outside')
        fig.update_layout(
            title_text=str('Box Plot Analysis for ' + str(cont_col.upper()) + str(" across : " + str(cat_col.upper()))))

    elif (cat_col is None) and (cont_col is not None):

        if count_df > threshold:

            group_dist = dict(sub.values() for sub in \
                              idf.groupBy(cont_col).count().fillna("Missing") \
                              .withColumn("count_%", int(threshold) * (F.col("count") / F.sum("count") \
                                                                       .over(Window.partitionBy())) / F.col("count")) \
                              .select(cont_col, "count_%").toPandas().to_dict('r'))
            idf = idf.fillna("Missing", subset=cont_col).sampleBy(cont_col, fractions=group_dist, seed=common_seed)

        else:
            pass

        idf = idf.select(cont_col).toPandas()

        fig = px.box(idf, y=cont_col, color=color_by, color_discrete_sequence=global_theme)
        #         fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Box Plot Analysis for ' + str(cont_col.upper())))

    else:
        pass

    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    #     plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}box_plot.html")

    return fig


def plot_gen_feat_analysis_label(idf, col, label, event_class, max_cat=None, bin_type=None):
    import plotly.express as px
    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    event_class = str(event_class)

    #     file_name_ = str(col) + "_" + str(event_class) + "_" + str(max_cat) + "_" + str(bin_type) + "_"

    class_cats = idf.select(label).distinct().rdd.flatMap(lambda x: x).collect()
    if col in cat_cols:
        idf = idf.groupBy(col).pivot(label).count() \
            .fillna(0, subset=class_cats) \
            .withColumn("event_rate", 100 * (F.col(event_class) / (F.col(class_cats[0]) + F.col(class_cats[1])))) \
            .withColumn("attribute_name", F.lit(col)) \
            .withColumn(col, f_remove_dups(col)) \
            .orderBy("event_rate", ascending=False) \
            .toPandas()

        fig = px.bar(idf, x=col, y='event_rate', text=idf['event_rate'].apply(lambda x: '{0:1.2f}%'.format(x)),
                     color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Event Rate Distribution for ' + str(col.upper()) + str(
            " [Target Variable : " + str(event_class) + str("]"))))
        fig.update_xaxes(type='category')

    elif col in num_cols:

        idf = feature_binning(idf, method_type=bin_type, bin_size=max_cat, list_of_cols=col, output_type="string")

        odf = idf.groupBy(col).pivot(label).count() \
            .fillna(0, subset=class_cats) \
            .withColumn("event_rate", 100 * (F.col(event_class) / (F.col(class_cats[0]) + F.col(class_cats[1])))) \
            .withColumn("attribute_name", F.lit(col)) \
            .withColumn(col, f_remove_dups(col)) \
            .orderBy("event_rate", ascending=False) \
            .toPandas()

        fig = px.bar(odf, x=col, y='event_rate', text=odf['event_rate'].apply(lambda x: '{0:1.2f}%'.format(x)),
                     color_discrete_sequence=global_theme)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text=str('Event Rate Distribution for ' + str(col.upper()) + str(
            " [Target Variable : " + str(event_class) + str("]"))))
        fig.update_xaxes(type='category')

    else:
        pass

    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    #     plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}feat_analysis_label.html")

    return fig


def plot_gen_variable_clustering(idf):
    import plotly.express as px

    fig = px.sunburst(idf, path=['Cluster', 'Attribute'], values='RS_Ratio', color_discrete_sequence=global_theme)
    #     fig.update_layout(title_text=str("Distribution of homogenous variable across Clusters"))

    #     plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}plot_sunburst.html")

    return fig


def plot_gen_dist(idf, col, threshold=500000, rug_chart=False):
    import plotly.figure_factory as ff
    #     file_name_ = str("distplot") + "_" + str(col) + "_"
    group_label = [col]
    count_df = idf.count()

    if col in num_cols:

        if count_df > threshold:
            group_dist = dict(sub.values() for sub in \
                              idf.groupBy(col).count().fillna("Missing", subset=col) \
                              .withColumn("count_%", \
                                          int(threshold) * (F.col("count") / F.sum("count") \
                                                            .over(Window.partitionBy())) / F.col("count")) \
                              .select(col, "count_%").toPandas().to_dict('r'))

            idf = idf.select(col).dropna().sampleBy(col, fractions=group_dist, seed=common_seed)

        else:
            idf = idf.select(col).dropna()

        idf = idf.select(col).rdd.flatMap(lambda x: x).collect()
        fig = ff.create_distplot([idf], group_labels=group_label, show_rug=rug_chart, colors=global_theme)

        fig.layout.plot_bgcolor = global_plot_bg_color
        fig.layout.paper_bgcolor = global_paper_bg_color
        fig.update_layout(title_text=str("Distribution Plot " + str(col)))

        #         plotly.offline.plot(fig, auto_open=False, validate=False, filename=f"{base_loc}/{file_name_}distplot.html")
        return fig

    else:
        return 0


def num_cols_chart_list(df, max_cat=10, bin_type="equal_range", output_path=None):
    num_cols_chart = []
    num_cols, cat_cols, other_cols = attributeType_segregation(df)
    for index, i in enumerate(num_cols):
        f = plot_gen_hist_bar(idf=df, col=i, max_cat=max_cat, bin_type=bin_type)
        if output_path is None:
            f.write_json("fig_num_f1_" + str(index))
        else:
            f.write_json(ends_with(output_path) + "fig_num_f1_" + str(index))

        num_cols_chart.append(f)


def cat_cols_chart_list(df, id_col, max_cat=10, cov=0.9, output_path=None):
    cat_cols_chart = []
    num_cols, cat_cols, other_cols = attributeType_segregation(df)
    for index, i in enumerate(cat_cols):
        if i != id_col:
            f = plot_gen_hist_bar(idf=df, col=i, max_cat=max_cat, cov=cov)
            if output_path is None:
                f.write_json("fig_cat_f1_" + str(index))
            else:
                f.write_json(ends_with(output_path) + "fig_cat_f1_" + str(index))

            cat_cols_chart.append(f)

        else:
            pass


def num_cols_int_chart_list(df, label, event_class, bin_type="equal_range", max_cat=10, output_path=None):
    num_cols_int_chart = []
    num_cols, cat_cols, other_cols = attributeType_segregation(df)
    for index, i in enumerate(num_cols):
        f = plot_gen_feat_analysis_label(idf=df, col=i, label=label, event_class=event_class, bin_type=bin_type,
                                         max_cat=max_cat)
        if output_path is None:
            f.write_json("fig_num_f2_" + str(index))
        else:
            f.write_json(ends_with(output_path) + "fig_num_f2_" + str(index))
        num_cols_int_chart.append(f)
    return num_cols_int_chart


def cat_cols_int_chart_list(df, id_col, label, event_class, output_path=None):
    cat_cols_int_chart = []
    num_cols, cat_cols, other_cols = attributeType_segregation(df)
    cat_cols = [x for x in cat_cols if label not in x]
    for index, i in enumerate(cat_cols):
        if i != id_col:
            f = plot_gen_feat_analysis_label(idf=df, col=i, label=label, event_class=event_class)
            if output_path is None:

                f.write_json("fig_cat_f2_" + str(index))
            else:
                f.write_json(ends_with(output_path) + "fig_cat_f2_" + str(index))
            cat_cols_int_chart.append(f)
        else:
            pass


def plot_comparative_drift_gen(df1, df2, col):
    num_cols, cat_cols, other_cols = attributeType_segregation(df1)

    if col in cat_cols:

        xx = outlier_categories(idf=df1, list_of_cols=col, coverage=0.9, max_category=10) \
            .groupBy(col).count() \
            .orderBy(col, ascending=True).withColumnRenamed("count", "count_source") \
            .join(outlier_categories(idf=df2, list_of_cols=col, coverage=0.9, max_category=10) \
                  .groupBy(col).count() \
                  .orderBy(col, ascending=True).withColumnRenamed("count", "count_target"), col, "left_outer") \
            .toPandas()
        xx.fillna({col: 'Missing', 'count_source': 0, 'count_target': 0}, inplace=True)

    elif col in num_cols:

        xx = feature_binning(df1, list_of_cols=col, method_type="equal_range", bin_size=10, output_type="number") \
            .groupBy(col).count() \
            .orderBy(col, ascending=True).withColumnRenamed("count", "count_source") \
            .join(feature_binning(df2, list_of_cols=col, method_type="equal_range", bin_size=10, output_type="number") \
                  .groupBy(col).count() \
                  .orderBy(col, ascending=True).withColumnRenamed("count", "count_target"), col, "left_outer") \
            .toPandas()
        xx.fillna({col: 'Missing', 'count_source': 0, 'count_target': 0}, inplace=True)

    else:
        pass

    xx['%_diff'] = (((xx['count_target'] / xx['count_source']) - 1) * 100)
    fig = go.Figure()
    fig.add_bar(y=list(xx.count_source.values), x=xx[col], name="source", marker=dict(color=global_theme))
    fig.update_traces(overwrite=True, marker={"opacity": 0.7})
    fig.add_bar(y=list(xx.count_target.values), x=xx[col], name="target",
                text=xx['%_diff'].apply(lambda x: '{0:0.2f}%'.format(x)), marker=dict(color=global_theme))
    fig.update_traces(textposition='outside')
    fig.update_layout(paper_bgcolor=global_paper_bg_color, plot_bgcolor=global_plot_bg_color, showlegend=False)
    fig.update_layout(title_text=str('Drift Comparison for ' + col + '<br><sup>(L->R : Source->Target)</sup>'))
    fig.update_traces(marker=dict(color=global_theme))
    fig.update_xaxes(type='category')
    fig.add_trace(go.Scatter(x=xx[col], y=xx.count_target.values, mode='lines+markers',
                             line=dict(color=px.colors.qualitative.Antique[10], width=3, dash='dot')))
    fig.update_layout(xaxis_tickfont_size=14, yaxis=dict(title='frequency', titlefont_size=16, tickfont_size=14))

    return fig


def violin_plot_gen(df, col, split_var=None, threshold=500000):
    count_df = df.count()

    if count_df > threshold:
        group_dist = dict(sub.values() for sub in \
                          df.groupBy(col).count().fillna("Missing", subset=cat_col) \
                          .withColumn("count_%", \
                                      int(threshold) * (F.col("count") / F.sum("count") \
                                                        .over(Window.partitionBy())) / F.col("count")) \
                          .select(col, "count_%").toPandas().to_dict('r'))

        df = df.dropna().sampleBy(col, fractions=group_dist, seed=common_seed).toPandas()

    else:
        df = df.dropna().toPandas()

    fig = px.violin(df, y=col, color=split_var, box=True, points="outliers",
                    color_discrete_sequence=[global_theme_r[8], global_theme_r[4]])
    fig.layout.plot_bgcolor = global_plot_bg_color
    fig.layout.paper_bgcolor = global_paper_bg_color
    fig.update_layout(legend=dict(orientation="h", x=0.5, yanchor="bottom", xanchor="center"))

    return fig


def num_cols_chart_list_outlier(idf, split_var=None, output_path=None):
    num_cols, cat_cols, other_cols = attributeType_segregation(idf)

    for index, i in enumerate(num_cols):
        f = violin_plot_gen(idf, i, split_var=split_var)
        if output_path is None:
            f.write_json("fig_num_f3_" + str(index))
        else:
            f.write_json(ends_with(output_path) + "fig_num_f3_" + str(index))


def charts_to_objects(idf, id_col=None, max_cat=10, label=None, event_class=None, chart_output_path=None):
    print("mapping chart objects")
    Path(chart_output_path).mkdir(parents=True, exist_ok=True)
    idf.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    num_cols_chart_list(idf, output_path=chart_output_path)
    cat_cols_chart_list(idf, id_col=id_col, output_path=chart_output_path)
    num_cols_chart_list_outlier(idf, split_var=label, output_path=chart_output_path)

    if label is not None:
        num_cols_int_chart_list(idf, label=label, event_class=event_class, output_path=chart_output_path)
        cat_cols_int_chart_list(idf, id_col=id_col, label=label, event_class=event_class, output_path=chart_output_path)


def output_pandas_df(idf, input_path, pandas_df_output_path, list_tabs, list_tab1, list_tab2, list_tab3, islabel=True):
    Path(pandas_df_output_path).mkdir(parents=True, exist_ok=True)
    idf.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    pd.DataFrame(idf.dtypes, columns=["Attributes", "Datatype"]).to_csv(
        ends_with(pandas_df_output_path) + "data_type_df.csv", index=False)

    list_tabs_arr = list_tabs.split(",")
    list_tab1_arr = list_tab1.split(",")
    list_tab2_arr = list_tab2.split(",")
    list_tab3_arr = list_tab3.split(",")

    remove_list = ['IV_calculation', 'IG_calculation']

    if islabel == False:
        list_tab3_arr = [x for x in list_tab3_arr if x not in remove_list]
    else:
        pass

    list_tabs_all = [list_tab1_arr, list_tab2_arr, list_tab3_arr]

    for index, i in enumerate(list_tabs_arr):
        for j in list_tabs_all[index]:
            if i == "stats_generator":
                spark.read.parquet(
                    ends_with(input_path) + ends_with("data_analyzer") + ends_with(i) + ends_with(j)).toPandas().to_csv(
                    ends_with(pandas_df_output_path) + i + "_" + j + ".csv", index=False)
            else:
                spark.read.parquet(
                    ends_with(input_path) + ends_with("data_analyzer") + ends_with(i) + ends_with(j) + ends_with(
                        "stats")).toPandas().to_csv(ends_with(pandas_df_output_path) + i + "_" + j + ".csv",
                                                    index=False)

    spark.read.parquet(ends_with(input_path) + ends_with("data_analyzer") + ends_with("stats_generator") + ends_with(
        "global_summary")).toPandas().to_csv(ends_with(pandas_df_output_path) + "global_summary_df.csv", index=False)


def data_drift(df2, df_source_path, drift_stats_path, chart_output_path=None, pandas_df_output_path=None,
               driftcheckrequired=False, drop_cols_viz=None):
    print("preparing drift charts")
    if bool(driftcheckrequired):
        Path(pandas_df_output_path).mkdir(parents=True, exist_ok=True)
        df1 = read_dataset(df_source_path.get("file_path"), df_source_path.get("file_type"),
                           df_source_path.get("file_configs"))
        stats_drift = read_dataset(drift_stats_path.get("file_path"), drift_stats_path.get("file_type"))
        # df1 = read_dataset(**df_source_path)
        # df2 = read_dataset(**df_target_path)
        # stats_drift = read_dataset(**drift_stats_path)
        stats_drift.toPandas().to_csv(ends_with(pandas_df_output_path) + "drift_statistics.csv", index=False)
        drifted_feats = stats_drift.where(F.col("flagged") == 1).select("attribute").rdd.flatMap(lambda x: x).collect()
        drifted_feats = [x for x in drifted_feats if x not in drop_cols_viz]
        num_cols, cat_cols, other_cols = attributeType_segregation(df1)
        for index, i in enumerate(drifted_feats):
            f = plot_comparative_drift_gen(df1, df2, i)
            if chart_output_path is None:
                f.write_json("fig_drift_feats_" + str(index))
            else:
                f.write_json(ends_with(chart_output_path) + "fig_drift_feats_" + str(index))
