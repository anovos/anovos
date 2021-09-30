import pyspark
import warnings
from pyspark.sql import functions as F
from pyspark.sql import types as T
from com.mw.ds.shared.spark import *
from com.mw.ds.shared.utils import attributeType_segregation, transpose_dataframe, get_dtype
from com.mw.ds.data_ingest.data_ingest import read_dataset
from com.mw.ds.data_analyzer.stats_generator import uniqueCount_computation, missingCount_computation, mode_computation, measures_of_cardinality
from com.mw.ds.data_transformer.transformers import imputation_MMM

def nullRows_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=0.8, print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all non-array columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params treatment: True if rows to be removed else False
    :params treatment_threshold: % of columns allowed to be Null per row, No row removal if treatment_threshold = 1
    :return: Output Dataframe (after row removal),
             Analysis Dataframe <null_cols_count,row_count,row_pct>
    """

    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if (treatment_threshold < 0) | (treatment_threshold > 1):
        raise TypeError('Invalid input for Treatment Threshold Value')
    if str(treatment).lower() == 'true':
        treatment= True 
    elif str(treatment).lower() == 'false':
        treatment= False 
    else: raise TypeError('Non-Boolean input for treatment')

    def null_count(*cols):
        return cols.count(None)
    f_null_count = F.udf(null_count, T.LongType())

    odf_tmp = idf.withColumn("null_cols_count", f_null_count(*list_of_cols))\
                .withColumn('flagged', F.when(F.col("null_cols_count") > (len(list_of_cols)*treatment_threshold), 1)\
                           .otherwise(0))
    
    if not(treatment) | (treatment_threshold == 1):
        odf = idf
    else:
        odf = odf_tmp.where(F.col("flagged") == 0).drop(*["null_cols_count","flagged"])

    odf_print = odf_tmp.groupBy("null_cols_count","flagged").agg(F.count(F.lit(1)).alias('row_count')) \
                        .withColumn('row_pct', F.round(F.col('row_count') / float(idf.count()), 4)) \
                        .select('null_cols_count','row_count','row_pct','flagged')
    if print_impact:
        odf_print.orderBy('null_cols_count').show(odf.count())
        
    return odf, odf_print


def duplicate_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all non-array columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params treatment: True if rows to be removed else False
    :return: Filtered Dataframe, Analysis Dataframe
    """
    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if str(treatment).lower() == 'true':
        treatment= True 
    elif str(treatment).lower() == 'false':
        treatment= False 
    else: raise TypeError('Non-Boolean input for treatment')

    odf_tmp = idf.drop_duplicates(subset=list_of_cols)
    odf = odf_tmp if treatment else idf
    
    odf_print = spark.createDataFrame([["rows_count",idf.count()],["unique_rows_count",odf_tmp.count()]],
                                      schema=['metric','value'])
    
    if print_impact:
        print("No. of Rows: " + str(idf.count()))
        print("No. of UNIQUE Rows: " + str(odf_tmp.count()))
        
    return odf, odf_print


def invalidEntries_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, 
                             output_mode='replace', print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: list or string of col names separated by |
                         all - to include all non-array columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params treatment: If True, replace invalid values by Null (false positives possible)
    :params output: replace or append
    :return: Output Dataframe (if treated) else Input Dataframe,
            Analysis Dataframe <attribute,invalid_entries,invalid_count,invalid_pct>
    """
    if list_of_cols == 'all':
        list_of_cols = []
        for i in idf.dtypes:
            if (i[1] in ('string', 'int', 'bigint', 'long')):
                list_of_cols.append(i[0])
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')
    if str(treatment).lower() == 'true':
        treatment= True 
    elif str(treatment).lower() == 'false':
        treatment= False 
    else: raise TypeError('Non-Boolean input for treatment')

    null_vocab = ['', ' ', 'nan', 'null', 'na', 'inf', 'n/a', 'not defined', 'none', 'undefined', 'blank']
    specialChars_vocab = ["&", "$", ";", ":", ".", ",", "*", "#", "@", "_", "?", "%", "!", "^", "(", ")", "-", "/", "'"]

    def detect(*v):
        output = []
        for idx,e in enumerate(v):
            if e is None:
                output.append(None)
                continue
            e = str(e).lower().strip()
            # Null & Special Chars Search
            if e in (null_vocab + specialChars_vocab):
                output.append(1)
                continue
            # Consecutive Identical Chars Search
            import re
            regex = "\\b([a-zA-Z0-9])\\1\\1+\\b"
            p = re.compile(regex)
            if (re.search(p, e)):
                output.append(1)
                continue
            # Ordered Chars Search
            l = len(e)
            check = 0
            if l >= 3:
                for i in range(1, l):
                    if ord(e[i]) - ord(e[i - 1]) != 1:
                        output.append(0)
                        check = 1
                        break
                if check == 1:
                    continue
                else:
                    output.append(1)
                    continue
            else:
                output.append(0)
                continue
        return output
    f_detect = F.udf(detect, T.ArrayType(T.LongType()))
    
    odf = idf.withColumn("invalid", f_detect(*list_of_cols))
    odf.persist()
    output_print = []
    for index, i in enumerate(list_of_cols):
        tmp = odf.withColumn(i + "_invalid", F.col('invalid')[index])
        invalid = tmp.where(F.col(i + "_invalid") == 1).select(i).distinct().rdd.flatMap(lambda x: x).collect()
        invalid = [str(x) for x in invalid]
        invalid_count = tmp.where(F.col(i + "_invalid") == 1).count()
        output_print.append([i, '|'.join(invalid), invalid_count, round(invalid_count / idf.count(), 4)])

    if treatment:
        for index, i in enumerate(list_of_cols):
            odf = odf.withColumn(i + "_invalid", F.when(F.col('invalid')[index] == 1, None).otherwise(F.col(i)))
            if output_mode == 'replace':
                odf = odf.drop(i).withColumnRenamed(i + "_invalid", i)
        odf = odf.drop("invalid")
    else:
        odf = idf  
    
    odf_print = spark.createDataFrame(output_print,
                                      schema=['attribute', 'invalid_entries', 'invalid_count', 'invalid_pct'])
    if print_impact:
        odf_print.show(len(list_of_cols))

    return odf, odf_print


def IDness_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=1.0, stats_unique={}, print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Categorical Columns (list or string of col names separated by |)
                         all - to include all categorical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)                  
    :params treatment: If True, delete columns based on treatment_threshold
    :params treatment_threshold: <0-1> Remove categorical column if no. of unique values is more than X% of total rows.
    :return: Filtered Dataframe (if treated), Analysis Dataframe <attribute, unique_values, IDness>
    """

    cat_cols = attributeType_segregation(idf)[1]
    if list_of_cols == 'all':
        list_of_cols = cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in cat_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if len(list_of_cols) == 0:
        warnings.warn("No IDness Check")
        odf = idf
        schema = T.StructType([T.StructField('attribute', T.StringType(), True),
                              T.StructField('unique_values', T.StringType(), True),
                              T.StructField('IDness', T.StringType(), True),
                              T.StructField('flagged', T.StringType(), True)])
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print
    if (treatment_threshold < 0) | (treatment_threshold > 1):
        raise TypeError('Invalid input for Treatment Threshold Value')
    if str(treatment).lower() == 'true':
        treatment= True 
    elif str(treatment).lower() == 'false':
        treatment= False 
    else: raise TypeError('Non-Boolean input for treatment')

    if stats_unique == {}:
        odf_print = measures_of_cardinality(idf, list_of_cols)\
                    .withColumn('flagged', F.lit("-"))
    else:
        odf_print = read_dataset(**stats_unique).withColumn('flagged', F.lit("-"))


    if treatment:
        remove_cols = odf_print.where(F.col('IDness') >= treatment_threshold) \
            .select('attribute').rdd.flatMap(lambda x: x).collect()
        odf = idf.drop(*remove_cols)
        odf_print = odf_print.withColumn('flagged', F.when(F.col('IDness') >= treatment_threshold, 1).otherwise(0))  
    else:
        odf = idf

    if print_impact:
        odf_print.show(len(list_of_cols))
        if treatment:
            print("Removed Columns: ", remove_cols)

    return odf, odf_print


def biasedness_detection(idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=1.0, stats_mode={}, print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Ideally categorical columns (in list format or string separated by |)
                             all - to include all non-array columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params treatment: If True, delete columns based on treatment_threshold
    :params treatment_threshold: <0-1> Remove categorical column if most freq value is in more than X% of total rows.
    :return: Filtered Dataframe (if treated), Analysis Dataframe <attribute,mode,mode_pct>
    """

    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if (treatment_threshold < 0) | (treatment_threshold > 1):
        raise TypeError('Invalid input for Treatment Threshold Value')
    if str(treatment).lower() == 'true':
        treatment= True 
    elif str(treatment).lower() == 'false':
        treatment= False 
    else: raise TypeError('Non-Boolean input for treatment')

    if stats_mode == {}:
        odf_print = transpose_dataframe(idf.select(list_of_cols).summary("count"), 'summary')\
                    .withColumnRenamed('key','attribute')\
                    .join(mode_computation(idf, list_of_cols),'attribute','full_outer')\
                    .withColumn('mode_pct', F.round(F.col('mode_rows')/F.col('count').cast(T.DoubleType()),4))\
                    .select('attribute','mode','mode_pct').withColumn('flagged', F.lit("-"))
    else:
        odf_print = read_dataset(**stats_mode).select('attribute','mode','mode_pct').withColumn('flagged', F.lit("-"))


    if treatment:
        remove_cols = odf_print.where((F.col('mode_pct') >= treatment_threshold)| (F.col('mode_pct').isNull())) \
            .select('attribute').rdd.flatMap(lambda x: x).collect()
        odf = idf.drop(*remove_cols)
        odf_print = odf_print.withColumn('flagged', 
                        F.when((F.col('mode_pct') >= treatment_threshold) | (F.col('mode_pct').isNull()), 1).otherwise(0)) 
    else:
        odf = idf

    if print_impact:
        odf_print.show(len(list_of_cols))
        if treatment:
            print("Removed Columns: ", remove_cols)

    return odf, odf_print

def outlier_detection(idf, list_of_cols='all', drop_cols=[], detection_side='upper', 
                      detection_configs={'pctile_lower': 0.05, 'pctile_upper': 0.95,
                                        'stdev_lower': 3.0, 'stdev_upper': 3.0,
                                        'IQR_lower': 1.5, 'IQR_upper': 1.5,
                                        'min_validation': 2},
                      treatment=False, treatment_type='value_replacement', pre_existing_model=False, 
                      model_path="NA", output_mode='replace', stats_unique={}, print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical Columns (list or string of col names separated by |)
                         all - to include all numerical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params detection_side: upper, lower, both
    :params detection_configs: dictionary format - upper & lower bound for each methodology. 
                        If a attribute value is less (more) than its derived lower (upper) bound value, 
                        it is considered as outlier by a methodology.
                        A attribute value is outliered if it is declared as oultlier by atleast 'min_validation' methodologies.
    :params treatment: True if rows to be removed else False
    :params treatment_type: null_replacement, row_removal, value_replacement
    :params pre_existing_model: outlier value for each attribute. True if model files exists already, False Otherwise
    :params model_path: If pre_existing_model is True, this argument is path for model file.
                  If pre_existing_model is False, this field can be used for saving the model file.
                  param NA means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    :return: Output Dataframe (after outlier treatment),
             Analysis Dataframe <attribute,lower_outliers,upper_outliers>
    """

    num_cols = attributeType_segregation(idf)[0]
    if len(num_cols) == 0:
        warnings.warn("No Outlier Check")
        odf = idf
        schema = T.StructType([T.StructField('attribute', T.StringType(), True),
                              T.StructField('lower_outliers', T.StringType(), True),
                              T.StructField('upper_outliers', T.StringType(), True)])
        odf_print = spark.sparkContext.emptyRDD().toDF(schema)
        return odf, odf_print
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    
    if stats_unique == {}:
        remove_cols = uniqueCount_computation(idf, list_of_cols).where(F.col('unique_values') < 2)\
                    .select('attribute').rdd.flatMap(lambda x:x).collect()
    else:
        remove_cols = read_dataset(**stats_unique).where(F.col('unique_values') < 2)\
                    .select('attribute').rdd.flatMap(lambda x:x).collect()

    list_of_cols = [e for e in list_of_cols if e not in (drop_cols + remove_cols)]

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if detection_side not in ('upper', 'lower', 'both'):
        raise TypeError('Invalid input for detection_side')
    if treatment_type not in ('null_replacement', 'row_removal', 'value_replacement'):
        raise TypeError('Invalid input for treatment_type')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')
    if str(treatment).lower() == 'true':
        treatment= True 
    elif str(treatment).lower() == 'false':
        treatment= False 
    else: raise TypeError('Non-Boolean input for treatment')
    if str(pre_existing_model).lower() == 'true':
        pre_existing_model = True 
    elif str(pre_existing_model).lower() == 'false':
        pre_existing_model = False 
    else: raise TypeError('Non-Boolean input for pre_existing_model')
    for arg in ['pctile_lower','pctile_upper']:
        if arg in detection_configs:
             if (detection_configs[arg] < 0) | (detection_configs[arg] > 1):
                raise TypeError('Invalid input for ' + arg)

    import numpy as np
    
    recast_cols = []
    recast_type = []
    for i in list_of_cols:
        if get_dtype(idf, i).startswith('decimal'):
            idf = idf.withColumn(i, F.col(i).cast(T.DoubleType()))
            recast_cols.append(i)
            recast_type.append(get_dtype(idf, i))
    
    if pre_existing_model:
        df_model = sqlContext.read.parquet(model_path + "/outlier_numcols")
        params = []
        for i in list_of_cols:
            mapped_value = df_model.where(F.col('attribute') == i).select('parameters')\
                                .rdd.flatMap(lambda x: x).collect()[0]
            params.append(mapped_value)
    else:
        detection_configs['pctile_lower'] = detection_configs['pctile_lower'] or 0.0
        detection_configs['pctile_upper'] = detection_configs['pctile_upper'] or 1.0
        pctile_params = idf.approxQuantile(list_of_cols, [detection_configs['pctile_lower'], 
                                                          detection_configs['pctile_upper']], 0.01)
        detection_configs['stdev_lower'] = detection_configs['stdev_lower'] or detection_configs['stdev_upper']
        detection_configs['stdev_upper'] = detection_configs['stdev_upper'] or detection_configs['stdev_lower']
        stdev_params = []
        for i in list_of_cols:
            mean, stdev = idf.select(F.mean(i), F.stddev(i)).first()
            stdev_params.append(
                [mean - detection_configs['stdev_lower'] * stdev, mean + detection_configs['stdev_upper'] * stdev])
            

        detection_configs['IQR_lower'] = detection_configs['IQR_lower'] or detection_configs['IQR_upper']
        detection_configs['IQR_upper'] = detection_configs['IQR_upper'] or detection_configs['IQR_lower']
        quantiles = idf.approxQuantile(list_of_cols, [0.25, 0.75], 0.01)
        IQR_params = [[e[0] - detection_configs['IQR_lower'] * (e[1] - e[0]),
                       e[1] + detection_configs['IQR_upper'] * (e[1] - e[0])] for e in quantiles]
        n = detection_configs['min_validation']
        params = [[sorted([x[0], y[0], z[0]], reverse=True)[n - 1], sorted([x[1], y[1], z[1]])[n - 1]] for x, y, z in
                  list(zip(pctile_params, stdev_params, IQR_params))]

        # Saving model File if required
        if model_path != "NA":
            df_model = spark.createDataFrame(zip(list_of_cols, params), schema=['attribute', 'parameters'])
            df_model.write.parquet(model_path + "/outlier_numcols", mode='overwrite')

    for i,j in zip(recast_cols,recast_type):
        idf = idf.withColumn(i,F.col(i).cast(j))
    
    def composite_outlier(*v):
        output = []
        for idx, e in enumerate(v):
            if e is None:
                output.append(None)
                continue
            if detection_side in ('upper', 'both'):
                if e > params[idx][1]:
                    output.append(1)
                    continue
            if detection_side in ('lower', 'both'):
                if e < params[idx][0]:
                    output.append(-1)
                    continue
            output.append(0)
        return output

    f_composite_outlier = F.udf(composite_outlier, T.ArrayType(T.IntegerType()))

    odf = idf.withColumn("outliered", f_composite_outlier(*list_of_cols))
    odf.persist()
    output_print = []
    for index, i in enumerate(list_of_cols):
        odf = odf.withColumn(i + "_outliered", F.col('outliered')[index])
        output_print.append(
            [i, odf.where(F.col(i + "_outliered") == -1).count(), odf.where(F.col(i + "_outliered") == 1).count()])

        if treatment & (treatment_type in ('value_replacement', 'null_replacement')):
            replace_vals = {'value_replacement': [params[index][0], params[index][1]], 'null_replacement': [None, None]}
            odf = odf.withColumn(i + "_outliered", F.when(F.col(i + "_outliered") == 1, replace_vals[treatment_type][1]) \
                                 .otherwise(F.when(F.col(i + "_outliered") == -1, replace_vals[treatment_type][0]) \
                                            .otherwise(F.col(i))))
            if output_mode == 'replace':
                odf = odf.drop(i).withColumnRenamed(i + "_outliered", i)

    odf = odf.drop("outliered")

    if treatment & (treatment_type == 'row_removal'):
        for index, i in enumerate(list_of_cols):
            odf = odf.where((F.col(i + "_outliered") == 0) | (F.col(i + "_outliered").isNull())).drop(i + "_outliered")

    if not (treatment):
        odf = idf
    
    odf_print = spark.createDataFrame(output_print, schema=['attribute', 'lower_outliers', 'upper_outliers'])
    if print_impact:
        odf_print.show(len(list_of_cols))

    return odf, odf_print


def nullColumns_detection(idf, list_of_cols='missing', drop_cols=[], treatment=False, treatment_method='row_removal', 
                   treatment_configs={}, stats_missing={}, stats_unique={}, stats_mode={}, print_impact=False):
    """
    :params idf: Pyspark Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                                 all - to include all non-array columns (excluding drop_cols)
                                 missing - all feautures with missing values (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params treatment: If True, Imputation/Dropna/Drop Column based on treatment_method
    :params treatment_method: MMM, row_removal or column_removal(more methods to be added soon)
    :params treatment_configs: All arguments of treatment_method/imputation functions in dictionary format
    :return: Imputed Dataframe
    """
    if stats_missing == {}:
        odf_print = missingCount_computation(idf)
    else:
        odf_print = read_dataset(**stats_missing).select('attribute','missing_count','missing_pct')
    
    missing_cols = odf_print.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()
    
    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        list_of_cols = num_cols + cat_cols
    if list_of_cols == "missing":
        list_of_cols = missing_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
        
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if len(list_of_cols) == 0:
        warnings.warn("No Action Performed - Imputation")
        return idf
    if any(x not in idf.columns for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if treatment_method not in ('MMM', 'row_removal','column_removal'):
        raise TypeError('Invalid input for method_type')
        
    odf_print = odf_print.where(F.col('attribute').isin(list_of_cols))

    if treatment:
        
        if treatment_method == 'column_removal':
            remove_cols = odf_print.where(F.col('attribute').isin(list_of_cols))\
                    .where(F.col('missing_pct') > treatment_configs['treatment_threshold'])\
                    .select('attribute').rdd.flatMap(lambda x: x).collect()
            odf = idf.drop(*remove_cols)
            if print_impact:
                print("Removed Columns: ", remove_cols)
        
        
        if treatment_method == 'row_removal':
            remove_cols = odf_print.where(F.col('attribute').isin(list_of_cols))\
                            .where(F.col('missing_pct') == 1.0)\
                            .select('attribute').rdd.flatMap(lambda x: x).collect()
            list_of_cols = [e for e in list_of_cols if e not in remove_cols]
            odf = idf.dropna(subset=list_of_cols)

            if print_impact:
                odf_print.show(len(list_of_cols))
                print("Before Count: " + str(idf.count()))
                print("After Count: " + str(odf.count()))
            
        if treatment_method == 'MMM':
            if stats_unique == {}:
                remove_cols = uniqueCount_computation(idf_sample, list_of_cols).where(F.col('unique_values') < 2)\
                            .select('attribute').rdd.flatMap(lambda x:x).collect()
            else:
                remove_cols = read_dataset(**stats_unique).where(F.col('unique_values') < 2)\
                            .select('attribute').rdd.flatMap(lambda x:x).collect()
            list_of_cols = [e for e in list_of_cols if e not in remove_cols]
            odf = imputation_MMM(idf, list_of_cols, **treatment_configs, stats_missing=stats_missing, stats_mode=stats_mode, print_impact=print_impact)
    else:
        odf = idf
        
    return odf, odf_print