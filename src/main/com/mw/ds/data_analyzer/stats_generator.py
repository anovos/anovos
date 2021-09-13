import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from spark import *
from com.mw.ds.data_ingest.data_ingest import *
from com.mw.ds.shared.utils import *

def missingCount_computation(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: List of columns for missing stats computation (list or string of col names separated by |)
                          all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)             
    :return: Dataframe <feature,missing_count,missing_pct>
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
        
    idf_stats = idf.select(list_of_cols).summary("count")
    odf = transpose_dataframe(idf_stats, 'summary')\
                .withColumn('missing_count', F.lit(idf.count()) - F.col('count').cast(T.LongType()))\
                .withColumn('missing_pct', F.round(F.col('missing_count')/F.lit(idf.count()),4))\
                .select(F.col('key').alias('feature'),'missing_count','missing_pct')
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def uniqueCount_computation(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: List of column for cardinality computation. Ideally categorical features.
                         List or string of col names separated by |.
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature,unique_values>
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
    
    uniquevalue_count = idf.agg(*(F.countDistinct(F.col(i)).alias(i) for i in list_of_cols))
    odf = transpose_dataframe(uniquevalue_count.withColumn("tmp", F.lit("unique_values")),"tmp")\
            .select(F.col("key").alias("feature"), F.col("unique_values").cast(T.LongType()))
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def mode_computation(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: List of columns for mode (most frequently seen value) computation. Ideally categorical features.
                         List or string of col names separated by |. In case of tie, one value is randomly picked as mode.
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)                   
    :return: Dataframe <feature,mode, mode_pct>
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
    
    mode = [str(idf.select(i).dropna().groupby(i).count().orderBy("count", ascending=False).first()[0]) for i in list_of_cols]
    mode_pct = [round(idf.where(F.col(i) == m).count()/float(idf.select(i).dropna().count()),4) for i,m in zip(list_of_cols,mode)]
    odf = spark.createDataFrame(zip(list_of_cols,mode,mode_pct), schema=("feature", "mode","mode_pct"))
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def nonzeroCount_computation(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: List of columns for computing nonZero rows. Ideally numerical features. 
                         List or string of col names separated by |
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature, nonzero_count,nonzero_pct>
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
        
    from pyspark.mllib.stat import Statistics
    from pyspark.mllib.linalg import Vectors
    tmp = idf.select(list_of_cols).fillna("0").rdd.map(lambda row: Vectors.dense(row))
    nonzero_count = Statistics.colStats(tmp).numNonzeros()
    odf = spark.createDataFrame(zip(list_of_cols,[int(i) for i in nonzero_count]), schema=("feature","nonzero_count"))\
            .withColumn("nonzero_pct",F.round(F.col('nonzero_count')/F.lit(idf.count()),4))
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def measures_of_centralTendency(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature, mean, median, mode, mode_pct>
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
    
    odf = transpose_dataframe(idf.select(list_of_cols).summary("mean","50%"), 'summary')\
                    .withColumn('mean', F.round(F.col('mean').cast(T.DoubleType()),4))\
                    .withColumn('median', F.round(F.col('50%').cast(T.DoubleType()),4))\
                    .select(F.col('key').alias('feature'), 'mean', 'median')\
                    .join(mode_computation(idf, list_of_cols),'feature','full_outer')\
                    
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def measures_of_cardinality(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Ideally Categorical Columns (list or string of col names separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature, unique_values, IDness>
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
    
    odf = uniqueCount_computation(idf, list_of_cols)\
            .join(missingCount_computation(idf, list_of_cols),'feature','full_outer')\
            .withColumn('IDness', F.round(F.col('unique_values')/(F.lit(idf.count()) - F.col('missing_count')),4))\
            .select('feature', 'unique_values','IDness')                    
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def measures_of_dispersion(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical Columns (list or string of col names separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature, stddev, variance, cov, IQR, range>
    """
    if list_of_cols == 'all':
        list_of_cols = featureType_segregation(idf)[0]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]
    
    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
        
    odf = transpose_dataframe(idf.select(list_of_cols).summary("stddev","min","max","mean","25%","75%"), 'summary')\
                    .withColumn('stddev', F.round(F.col('stddev').cast(T.DoubleType()),4))\
                    .withColumn('variance', F.round(F.col('stddev')* F.col('stddev'),4))\
                    .withColumn('range', F.round(F.col('max') - F.col('min'),4))\
                    .withColumn('cov', F.round(F.col('stddev')/F.col('mean'),4))\
                    .withColumn('IQR', F.round(F.col('75%') - F.col('25%'),4))\
                    .select(F.col('key').alias('feature'),'stddev','variance','cov','IQR','range')
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def measures_of_percentiles(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical Columns (list or string of col names separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature,min,1%,5%,10%,25%,50%,75%,90%,95%,99%,max>
    """
    if list_of_cols == 'all':
        list_of_cols = featureType_segregation(idf)[0]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]
    
    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    stats = ["min", "1%","5%","10%","25%","50%","75%","90%","95%","99%","max"]
    odf = transpose_dataframe(idf.select(list_of_cols).summary(*stats), 'summary')\
            .withColumnRenamed("key","feature")
    for i in odf.columns:
        if i != "feature":
            odf = odf.withColumn(i, F.round(F.col(i).cast("Double"),4))
    odf = odf.select(['feature'] + stats)
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def measures_of_counts (idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature, fill_count,fill_pct,missing_count,missing_pct,nonzero_count,nonzero_pct>
    """
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]
    num_cols = featureType_segregation(idf)[0]
    
    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')

    odf = transpose_dataframe(idf.select(list_of_cols).summary("count"), 'summary')\
            .select(F.col("key").alias("feature"),F.col("count").cast(T.LongType()).alias("fill_count"))\
            .withColumn('fill_pct', F.round(F.col('fill_count')/F.lit(idf.count()),4))\
            .withColumn('missing_count', F.lit(idf.count()) - F.col('fill_count').cast(T.LongType()))\
            .withColumn('missing_pct', F.round(1 - F.col('fill_pct'),4))\
            .join(nonzeroCount_computation(idf,num_cols),"feature","full_outer")
        
    if print_impact:
        odf.show(len(list_of_cols))
    return odf

def measures_of_shape(idf, list_of_cols='all', drop_cols=[], print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical Columns (list or string of col names separated by |)
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Dataframe <feature,skewness,kurtosis>
    """
    if list_of_cols == 'all':
        list_of_cols = featureType_segregation(idf)[0]
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]
    
    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    shapes = []
    for i in list_of_cols:
        s, k = idf.select(F.skewness(i), F.kurtosis(i)).first()
        shapes.append([i,round(s,4),round(k,4)])
    odf = spark.createDataFrame(shapes, schema=("feature","skewness","kurtosis"))
    if print_impact:
        odf.show(len(list_of_cols))
    return odf


def global_summary(idf, list_of_cols='all', drop_cols=[], print_impact=True):
    '''
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Analysis Dataframe
    '''
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]
    
    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    row_count = idf.count()
    col_count = len(list_of_cols)
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols))
    numcol_count = len(num_cols)
    catcol_count = len(cat_cols)
    if print_impact:
        print("No. of Rows: %s" %"{0:,}".format(row_count))
        print("No. of Columns: %s" %"{0:,}".format(col_count))
        print("Numerical Columns: %s"%"{0:,}".format(numcol_count))
        if numcol_count > 0:
            print(num_cols)
        print("Categorical Columns: %s"%"{0:,}".format(catcol_count))
        if catcol_count > 0:
            print(cat_cols)
            
    odf = spark.createDataFrame([["rows_count",str(row_count)],["columns_count",str(col_count)],
                                ["numcols_count",str(numcol_count)],["numcols_name",', '.join(num_cols)],
                                ["catcols_count",str(catcol_count)],["catcols_name",', '.join(cat_cols)]], 
                                schema=['metric','value'])
    return odf
    
    
def descriptive_stats(idf, list_of_cols='all', drop_cols=[], print_impact=True):
    '''
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :return: Output Dataframe <feature, descriptive_stats>
    '''
    if list_of_cols == 'all':
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]
    
    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    row_count = idf.count()
    col_count = len(list_of_cols)
    num_cols, cat_cols, other_cols = featureType_segregation(idf.select(list_of_cols))
    numcol_count = len(num_cols)
    catcol_count = len(cat_cols)
    if print_impact:
        print("No. of Rows: %s" %"{0:,}".format(row_count))
        print("No. of Columns: %s" %"{0:,}".format(col_count))
        print("Numerical Columns: %s"%"{0:,}".format(numcol_count))
        if numcol_count > 0:
            print(num_cols)
        print("Categorical Columns: %s"%"{0:,}".format(catcol_count))
        if catcol_count > 0:
            print(cat_cols)
    
    if len(num_cols) > 0:
        stats = ["count", "min", "1%","5%","10%","25%","50%","75%","90%","95%","99%","max","mean","stddev"]
        idf_stats = transpose_dataframe(idf.select(num_cols).summary(*stats),'summary')\
                    .withColumnRenamed("key","feature")      
        odf_num = idf_stats.join(measures_of_shape(idf, num_cols),"feature","full_outer")\
                    .join(nonzeroCount_computation(idf,num_cols),"feature","full_outer")\
                    .join(mode_computation(idf,num_cols),"feature","full_outer")\
                    .withColumn('missing_count', F.lit(idf.count()) - F.col('count').cast(T.LongType()))\
                    .withColumn('missing_pct', F.col('missing_count')/F.lit(idf.count()))\
                    .withColumn('variance', F.col('stddev')*F.col('stddev'))\
                    .withColumn('range', F.col('max') - F.col('min'))\
                    .withColumn('cov', F.col('stddev')/F.col('mean'))\
                    .withColumn('IQR', F.col('75%') - F.col('25%'))\
                    .select('feature', 'missing_count', 'missing_pct', 
                            'nonzero_count', 'nonzero_pct',
                            'min' , 'max', 'mean', F.col('50%').alias('median'),'mode','mode_pct', 
                            'skewness', 'kurtosis', 
                            'stddev','variance', 'range', 'cov', 'IQR',
                            '1%','5%','10%', '25%', '75%', '90%', '95%', '99%')
                    
        for i in ['missing_pct','min','max','mean','median','mode',  
                   'stddev','variance', 'range', 'cov', 'IQR',
                    '1%','5%','10%', '25%', '75%', '90%', '95%', '99%']:
            odf_num = odf_num.withColumn(i, F.round(F.col(i).cast(T.DoubleType()),4))
                    
        if print_impact:
            odf_num.printSchema()
            odf_num.show(len(num_cols))
                
    if len(cat_cols) > 0:
        odf_cat = missingCount_computation(idf, cat_cols)\
                     .join(mode_computation(idf, cat_cols),"feature","full_outer")\
                     .join(uniqueCount_computation(idf, cat_cols),"feature","full_outer")\
                     .withColumn('IDness', F.round(F.col('unique_values')/(F.lit(row_count) - F.col('missing_count')),4))
                     
                     
        if print_impact:
            odf_cat.printSchema()
            odf_cat.show(len(cat_cols))
    
    if len(cat_cols) == 0:
        odf = odf_num.withColumn('feature_type', F.lit('numerical'))
    elif len(num_cols) == 0:
        odf = odf_num.withColumn('feature_type', F.lit('categorical'))
    else:
        all_cols = odf_num.columns + [e for e in odf_cat.columns if e not in odf_num.columns]
        for i in all_cols:
            if i not in odf_num.columns:
                    odf_num = odf_num.withColumn(i, F.lit(None))
            if i not in odf_cat.columns:
                    odf_cat = odf_cat.withColumn(i, F.lit(None))
        odf = concatenate_dataset(odf_num.withColumn('feature_type', F.lit('numerical')),
                                  odf_cat.withColumn('feature_type', F.lit('categorical')))
        #odf = odf_num.withColumn('feature_type', F.lit('numerical')\
        #   .unionByName(odf_cat.withColumn('feature_type', F.lit('categorical'), allowMissingColumns=True) #spark 3.X
    return odf
