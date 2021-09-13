from pyspark.sql import functions as F
from pyspark.sql import types as T
from spark import *
from com.mw.ds.shared.utils import *
from com.mw.ds.shared.transformers import *
from com.mw.ds.data_analyzer.quality_checker import *
from com.mw.ds.data_analyzer.stats_generator import *

def correlation_matrix(idf, list_of_cols='all', drop_cols=[], plot=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Correlation Dataframe <feature,<col_names>>
    """
     
    from phik.phik import spark_phik_matrix_from_hist2d_dict
    from popmon.analysis.hist_numpy import get_2dgrid
    import itertools
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    combis = [list(c) for c in itertools.combinations_with_replacement(list_of_cols, 2)]
    hists = idf.pm_make_histograms(combis)
    grids = {k:get_2dgrid(h) for k,h in hists.items()}
    odf_pd = spark_phik_matrix_from_hist2d_dict(sc, grids)
    
    if plot:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, (ax) = plt.subplots(1, 1, figsize=(10,6))
        hm = sns.heatmap(odf_pd,ax=ax,  cmap="Blues", linewidths=.05)
        fig.subplots_adjust(top=0.93)
        plt.figure()
        plt.show()
    
    odf_pd['feature'] = odf_pd.index
    odf = sqlContext.createDataFrame(odf_pd)
    
    return odf

def IV_calculation(idf, list_of_cols='all', drop_cols=[], label_col='label', event_label=1, 
                   encoding_configs={'bin_method':'equal_frequency', 'bin_size':10,'monotonicity_check':0}, plot=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params label_col: Label column
    :params event_label: Value of event (binary classfication)
    :params encoding_configs: dict format, {} empty dict for no encoding
                            bin_size: No. of bins, bin_method: equal_frequency, equal_range, 
                            monotonicity_check = 1 for monotonicity encoding else 0
    :return: Dataframe <feature, iv>
    """
    
    if label_col not in idf.columns:
        raise TypeError('Invalid input for Label Column')
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in (drop_cols+[label_col])]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if (idf.where(F.col(label_col) == event_label).count() == 0):
        raise TypeError('Invalid input for Event Label Value')
        
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols))
    
    if (len(num_cols) > 0) & bool(encoding_configs):
        bin_size = encoding_configs['bin_size']
        bin_method = encoding_configs['bin_method']
        monotonicity_check = encoding_configs['monotonicity_check']
        if monotonicity_check == 1:
            idf_encoded = monotonic_encoding(idf,num_cols,[],label_col,event_label,bin_method,bin_size)
        else:
            idf_encoded = feature_binning(idf,num_cols,[],bin_method,bin_size)
            
        idf_encoded.write.parquet("intermediate_data/IV_calculation",mode='overwrite')
        idf_encoded = spark.read.parquet("intermediate_data/IV_calculation")
    else:
        idf_encoded = idf
    
    output = []
    for col in list_of_cols:
        from pyspark.sql import Window
        df_iv = idf_encoded.groupBy(col,label_col).count()\
                    .withColumn(label_col, F.when(F.col(label_col) == event_label,1).otherwise(0))\
                    .groupBy(col).pivot(label_col).sum('count').fillna(0.5)\
                    .withColumn('event_pct', F.col("1")/F.sum("1").over(Window.partitionBy()))\
                    .withColumn('nonevent_pct', F.col("0")/F.sum("0").over(Window.partitionBy()))\
                    .withColumn('iv', (F.col('nonevent_pct') - F.col('event_pct'))*F.log(F.col('nonevent_pct')/F.col('event_pct')))
        iv_value = round(df_iv.select(F.sum('iv')).collect()[0][0],4)
        output.append([col,iv_value])
        
    odf = spark.createDataFrame(output, ["feature", "iv"]).orderBy(F.desc('iv'))
    
        
    if plot:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('ggplot')
        sns.set(font_scale = 1)
        data = odf.toPandas()
        sns.barplot(x="iv", y="feature", data=data, orient="h", color='steelblue')
        plt.figure()
        plt.show()
    return odf

def IG_calculation(idf, list_of_cols='all', drop_cols=[], label_col='label', event_label=1, 
                   encoding_configs={'bin_method':'equal_frequency', 'bin_size':10,'monotonicity_check':0}, plot=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params label_col: Label column
    :params event_label: Value of event (binary classfication)
    :params encoding_configs: dict format, {} empty dict for no encoding
                            bin_size: No. of bins, bin_method: equal_frequency, equal_range, 
                            monotonicity_check = 1 for monotonicity encoding else 0
    :return: Dataframe <feature, ig>
    """
    
    if label_col not in idf.columns:
        raise TypeError('Invalid input for Label Column')
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in (drop_cols+[label_col])]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if (idf.where(F.col(label_col) == event_label).count() == 0):
        raise TypeError('Invalid input for Event Label Value')
        
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols))
    
    if (len(num_cols) > 0) & bool(encoding_configs):
        bin_size = encoding_configs['bin_size']
        bin_method = encoding_configs['bin_method']
        monotonicity_check = encoding_configs['monotonicity_check']
        if monotonicity_check == 1:
            idf_encoded = monotonic_encoding(idf,num_cols,[],label_col,event_label,bin_method,bin_size)
        else:
            idf_encoded = feature_binning(idf,num_cols,[],bin_method,bin_size)
        idf_encoded.write.parquet("intermediate_data/IG_calculation",mode='overwrite')
        idf_encoded = spark.read.parquet("intermediate_data/IG_calculation")
    else:
        idf_encoded = idf
    
    import math
    from pyspark.sql import Window
    output = []
    for col in list_of_cols:
        idf_entropy = idf_encoded.withColumn(label_col, F.when(F.col(label_col) == event_label,1).otherwise(0))\
                    .groupBy(col).agg(F.sum(F.col(label_col)).alias('event_count'), 
                                      F.count(F.col(label_col)).alias('total_count')).dropna()\
                    .withColumn('event_pct', F.col('event_count')/F.col('total_count'))\
                    .withColumn('segment_pct', F.col('total_count')/F.sum('total_count').over(Window.partitionBy()))\
                    .withColumn('entropy', - F.col('segment_pct')*((F.col('event_pct')*F.log2(F.col('event_pct'))) + ((1-F.col('event_pct'))*F.log2((1-F.col('event_pct'))))))
        entropy = round(idf_entropy.groupBy().sum('entropy').rdd.flatMap(lambda x: x).collect()[0],4)
        total_event = idf.where(F.col(label_col) == event_label).count()/idf.count()
        total_entropy = - (total_event*math.log2(total_event) + ((1-total_event)*math.log2((1-total_event))))
        ig_value = total_entropy - entropy
        output.append([col,ig_value])
    
    odf = sqlContext.createDataFrame(output, ["feature", "ig"]).orderBy(F.desc('ig'))
    
    if plot:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('ggplot')
        sns.set(font_scale = 1)
        data = odf.toPandas()
        sns.barplot(x="ig", y="feature", data=data, orient="h", color='steelblue')
        plt.figure()
        plt.show()
    return odf

def features_association(idf, list_of_cols='all', drop_cols=[], label_col='label', event_label=1, 
                         encoding_configs={'bin_method':'equal_frequency', 'bin_size':10}, plot=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params label_col: Label column
    :params event_label: Value of event (binary classfication)
    :params encoding_configs: dict format, {} empty dict for no encoding
                            bin_size: No. of bins, bin_method: equal_frequency, equal_range
    :return: Dataframe <f1,f2,composite_score>
    """
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if label_col not in idf.columns:
        raise TypeError('Invalid input for Label Column')
    if (idf.where(F.col(label_col) == event_label).count() == 0):
        raise TypeError('Invalid input for Event Label Value')
    
    import itertools
    from pyspark.sql import Window
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols))
    if (len(num_cols) > 0) & bool(encoding_configs):
        bin_size = encoding_configs['bin_size']
        bin_method = encoding_configs['bin_method']
        idf_encoded = feature_binning (idf,num_cols,[],bin_method,bin_size)
        idf_encoded.write.parquet("intermediate_data/features_association",mode='overwrite')
        idf_encoded = spark.read.parquet("intermediate_data/features_association")
    else:
        idf_encoded = idf
    
    
    combis = itertools.combinations_with_replacement(list_of_cols, 2)
    combis = [list(c) for c in combis]
    idf_pairs = idf
    pair_cols = []
    for f1, f2 in combis:
        idf_pairs = idf_pairs.withColumn(f1 + ":" + f2, F.concat(F.col(f1), F.lit(":"), F.col(f2)))
        pair_cols.append(f1 + ":" + f2)
    
    idf_iv = IV_calculation(idf_encoded,list_of_cols,[], label_col, event_label,{})
    idf_ig = IG_calculation(idf_encoded,list_of_cols,[], label_col, event_label,{})
    idf_pairs_iv = IV_calculation(idf_pairs,pair_cols,[], label_col, event_label,{})
    idf_pairs_ig = IG_calculation(idf_pairs,pair_cols,[], label_col, event_label,{})
    
    odf = idf_pairs_iv.join(idf_pairs_ig,'feature','full_outer')\
            .withColumn('f1', F.split(F.col('feature'),':').getItem(0))\
            .withColumn('f2', F.split(F.col('feature'),':').getItem(1))\
            .join(idf_iv.select(F.col('feature').alias('f1'),F.col('iv').alias('f1_iv')),'f1','left_outer')\
            .join(idf_iv.select(F.col('feature').alias('f2'),F.col('iv').alias('f2_iv')),'f2','left_outer')\
            .join(idf_ig.select(F.col('feature').alias('f1'),F.col('ig').alias('f1_ig')),'f1','left_outer')\
            .join(idf_ig.select(F.col('feature').alias('f2'),F.col('ig').alias('f2_ig')),'f2','left_outer')\
            .withColumn('ivi', F.col('iv') - F.greatest(F.col('f1_iv'), F.col('f2_iv')))\
            .withColumn('ivl', F.col('iv')/F.greatest(F.col('f1_iv'), F.col('f2_iv')))\
            .withColumn('iv_gain', F.col('ivi') * F.col('ivl'))\
            .withColumn('igi', F.col('ig') - F.greatest(F.col('f1_ig'), F.col('f2_ig')))\
            .withColumn('igl', F.col('ig')/F.greatest(F.col('f1_ig'), F.col('f2_ig')))\
            .withColumn('ig_gain', F.col('igi') * F.col('igl'))\
            .withColumn('iv_score', F.rank().over(Window.partitionBy().orderBy('iv_gain')))\
            .withColumn('ig_score', F.rank().over(Window.partitionBy().orderBy('ig_gain')))\
            .withColumn('composite_score', (F.col('iv_score')+F.col('ig_score'))/2)
            
    odf = normalization(odf,['composite_score']).select('f1','f2','composite_score')
            
    if plot:
        corr = odf.select('f1','f2','composite_score')\
                .union(odf.select('f2','f1','composite_score')).drop_duplicates()\
                .groupBy('f1').pivot('f2').sum('composite_score').orderBy('f1').toPandas().set_index('f1')
        corr.index.name = None
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, (ax) = plt.subplots(1, 1, figsize=(10,6))
        hm = sns.heatmap(corr,ax=ax,  cmap="Blues", linewidths=.05)
        fig.subplots_adjust(top=0.93)
        plt.figure()
        plt.show()
        
    return odf

def SHAP_calculation(idf, list_of_cols='all', drop_cols=[], label_col='label', sample_size=100000, plot=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params label_col: Label column
    :params sample_size: maximum sample size for computation
    :return: Dataframe <feature, SHAP_value>
    """
    
    import shap
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    
    if label_col not in idf.columns:
        raise TypeError('Invalid input for Label Column')
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in (drop_cols+[label_col])]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
        
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols+[label_col]))
    idf_encoded = cat_to_num_unsupervised (idf, list_of_cols=cat_cols, method_type=1)
    idf_imputed = imputation_MMM(idf_encoded)
    idf_pd = idf_imputed.sample(False, min(1.0, float(sample_size)/idf.count()), 0).toPandas()
    X_train = idf_pd[list_of_cols]
    Y_train = idf_pd[label_col]
    mod = RandomForestClassifier(random_state=20)
    mod.fit(X=X_train, y=Y_train)
    explainer = shap.TreeExplainer(mod)
    shap_values = explainer.shap_values(X_train)
    output= pd.DataFrame(shap_values[0], columns=list_of_cols).abs().mean(axis=0).sort_values(ascending=False)
    odf = sqlContext.createDataFrame(pd.DataFrame(output.reset_index()), 
                                     schema=["feature", "SHAP_value"]).orderBy(F.desc('SHAP_value'))
    
    if plot:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('ggplot')
        sns.set(font_scale = 1)
        data = odf.toPandas()
        sns.barplot(x="SHAP_value", y="feature", data=data, orient="h", color='steelblue')
        plt.figure()
        plt.show()
    return odf

def variable_clustering(idf, list_of_cols='all', drop_cols=[], sample_size=100000, plot=False):
    
    """
    :params idf: Input Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params sample_size: maximum sample size for computation
    :return: Dataframe <Cluster, feature, RS_Ratio>
    """
    
    from varclushi import VarClusHi
    
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols))
    idf_encoded = cat_to_num_unsupervised (idf, list_of_cols=cat_cols, method_type=1)
    #missingCount_computation(idf_encoded).show()
    idf_imputed = imputation_MMM(idf_encoded.select(list_of_cols))
    #missingCount_computation(idf_imputed).show()
    idf_pd = idf_imputed.sample(False, min(1.0, float(sample_size)/idf.count()), 0).toPandas()    
    vc = VarClusHi(idf_pd,maxeigval2=1,maxclus=None)
    vc.varclus()
    odf_pd = vc.rsquare
    odf = sqlContext.createDataFrame(odf_pd).select('Cluster',F.col('Variable').alias('feature'),'RS_Ratio')
    
    if plot:
        odf.show(odf.count())
    return odf

