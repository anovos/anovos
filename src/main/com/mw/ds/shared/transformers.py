import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
import warnings
from spark import *
from com.mw.ds.shared.utils import *
from com.mw.ds.data_analyzer.stats_generator import *

def feature_binning (idf, list_of_cols='all', drop_cols=[], method_type="equal_range", bin_size=10, 
                     pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical columns (in list format or string separated by |)
                         all - to include all numerical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params method_type: equal_frequency, equal_range
    :params bin_size: No of bins
    :params pre_existing_model: True if mapping values exists already, False Otherwise. 
    :params model_path: If pre_existing_model is True, this argument is path for the saved model. 
                  If pre_existing_model is False, this argument can be used for saving the model. 
                  Default "NA" means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    :return: Binned Dataframe
    """
    
    num_cols = featureType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')
    if len(list_of_cols) == 0:
        warnings.warn("No Transformation Performed")
        return idf
    
    if method_type not in ("equal_frequency", "equal_range"):
        raise TypeError('Invalid input for method_type')
    if bin_size < 2:
        raise TypeError('Invalid input for bin_size')
    if output_mode not in ('replace','append'):
        raise TypeError('Invalid input for output_mode')
    

    if pre_existing_model:
        df_model = sqlContext.read.parquet(model_path + "/feature_binning")
        bin_cutoffs = []
        for i in list_of_cols:
            mapped_value = df_model.where(F.col('feature') == i).select('parameters')\
                                .rdd.flatMap(lambda x: x).collect()[0]
            bin_cutoffs.append(mapped_value)
    else:
        if method_type == "equal_frequency":
            pctile_width = 1/bin_size
            pctile_cutoff = []
            for j in range(1,bin_size):
                pctile_cutoff.append(j*pctile_width)  
            bin_cutoffs = idf.approxQuantile(list_of_cols, pctile_cutoff , 0.01)

        else:
            bin_cutoffs = []
            for i in list_of_cols:
                max_val = idf.select(F.col(i)).groupBy().max().rdd.flatMap(lambda x: x).collect()[0]
                min_val = idf.select(F.col(i)).groupBy().min().rdd.flatMap(lambda x: x).collect()[0]
                bin_width = (max_val - min_val)/bin_size
                bin_cutoff = []
                for j in range(1,bin_size):
                    bin_cutoff.append(min_val+j*bin_width) 
                bin_cutoffs.append(bin_cutoff)
                
        if model_path != "NA":
            df_model = spark.createDataFrame(zip(list_of_cols, bin_cutoffs), schema=['feature', 'parameters'])
            df_model.write.parquet(model_path + "/feature_binning", mode='overwrite')
    
    def bucket_label (value, index):
        if value is None:
            return None
        for j in range (0,len(bin_cutoffs[0])):
            if value <= bin_cutoffs[index][j]:
                return round((j + 1.0)*0.1,2)
            else:
                next
        return 1.0
    f_bucket_label = F.udf(bucket_label, T.FloatType())

    odf = idf
    for idx, i in enumerate(list_of_cols):
        odf = odf.withColumn(i+"_binned",f_bucket_label(F.col(i), F.lit(idx)))
        
        if idx%5 == 0:
            odf.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
            
    if output_mode == 'replace':
        for col in list_of_cols:
            odf = odf.drop(col).withColumnRenamed(col+"_binned",col)
        
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_binned") for i in list_of_cols]
        uniqueCount_computation(odf, output_cols).show(len(output_cols))
    return odf

"""
def feature_binning (idf, list_of_cols='all', drop_cols=[], method_type="equal_range", bin_size=10, 
                     pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False):
    '''
    :params idf: Input Dataframe
    :params list_of_cols: Numerical columns (in list format or string separated by |)
                         all - to include all numerical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params method_type: equal_frequency, equal_range
    :params bin_size: No of bins
    :params pre_existing_model: True if mapping values exists already, False Otherwise. 
    :params model_path: If pre_existing_model is True, this argument is path for the saved model. 
                  If pre_existing_model is False, this argument can be used for saving the model. 
                  Default "NA" means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    :return: Binned Dataframe
    '''
    
    num_cols = featureType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in num_cols for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    if method_type not in ("equal_frequency", "equal_range"):
        raise TypeError('Invalid input for method_type')
    if bin_size < 2:
        raise TypeError('Invalid input for bin_size')
    if output_mode not in ('replace','append'):
        raise TypeError('Invalid input for output_mode')
    
    odf = idf
    for idx, col in enumerate(list_of_cols):
        
        #if (idx-1)%5 == 0:
         #   odf = spark.read.parquet("intermediate_data/feature_binning/"+str(idx-1))
        
        if method_type == "equal_frequency":
            from pyspark.ml.feature import QuantileDiscretizer
            if pre_existing_model == True:
                discretizerModel = QuantileDiscretizer.load(model_path + "/feature_binning/" + col)
            else:
                discretizer = QuantileDiscretizer(numBuckets=bin_size,inputCol=col, outputCol=col+"_binned")
                discretizerModel = discretizer.fit(odf)
            #print(discretizerModel.getSplits())
            odf = discretizerModel.transform(odf)
            
            if (pre_existing_model == False) & (model_path != "NA"):
                discretizerModel.write().overwrite().save(model_path + "/feature_binning/" + col)
        else:
            from pyspark.ml.feature import Bucketizer
            if pre_existing_model == True:
                bucketizer = Bucketizer.load(model_path + "/feature_binning/" + col)
            else:
                max_val = idf.select(F.col(col)).groupBy().max().rdd.flatMap(lambda x: x).collect()[0]
                min_val = idf.select(F.col(col)).groupBy().min().rdd.flatMap(lambda x: x).collect()[0]
                bin_width = (max_val - min_val)/bin_size
                bin_cutoff = [-float("inf")]
                for i in range(1,bin_size):
                    bin_cutoff.append(min_val+i*bin_width)            
                bin_cutoff.append(float("inf"))
                #print(col, bin_cutoff)
                bucketizer = Bucketizer(splits=bin_cutoff, inputCol=col, outputCol=col+"_binned")
                
                if (pre_existing_model == False) & (model_path != "NA"):
                    bucketizer.write().overwrite().save(model_path + "/feature_binning/" + col)    
            #print(bucketizer.getSplits())
            odf = bucketizer.transform(odf)
            
        if idx%5 == 0:
            odf.persist()
            print(odf.count())
         #   odf.write.parquet("intermediate_data/feature_binning/"+str(idx),mode='overwrite')
            
    if output_mode == 'replace':
        for col in list_of_cols:
            odf = odf.drop(col).withColumnRenamed(col+"_binned",col)
        
    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i+"_binned") for i in list_of_cols]
        uniqueCount_computation(odf, output_cols).show(len(output_cols))
    return odf
"""

def monotonic_encoding(idf,list_of_cols='all', drop_cols=[], label_col='label', 
                       event_label=1, bin_method="equal_frequency", bin_size=10):
    """
    :params idf: Input Dataframe
    :params list_of_cols: all or list of numerical columns (in list format or string separated by |)
    :params method_type: equal_frequency, equal_range
    :params bin_size: No of bins
    :return: Binned Dataframe
    """
    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = featureType_segregation(idf)
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    
    from scipy import stats
    idf_encoded = idf
    for col in list_of_cols:
        n = 20 #max_bin
        r = 0
        while n > 2:                
            tmp = feature_binning (idf,[col],bin_method, n,output_mode='append')\
                    .select(label_col,col,col+'_binned')\
                    .withColumn(label_col, F.when(F.col(label_col) == event_label,1).otherwise(0))\
                    .groupBy(col+'_binned').agg(F.avg(col).alias('mean_val'), F.avg(label_col).alias('mean_label')).dropna()
            #r = tmp.stat.corr('mean_age','mean_label')
            r,p = stats.spearmanr(tmp.toPandas()[['mean_val']], tmp.toPandas()[['mean_label']])
            if r == 1.0:
                idf_encoded = feature_binning (idf_encoded,[col],bin_method, n)
                print(col,n)
                break
            n = n-1
            r = 0
        if r < 1.0:
            idf_encoded = feature_binning (idf_encoded,[col],bin_method, bin_size)
            
    return idf_encoded


def cat_to_num_unsupervised (idf, list_of_cols, method_type, index_order='frequencyDesc', onehot_dropLast=False,
                             pre_existing_model=False, model_path="NA", output_mode='replace',print_impact=False):
    '''
    idf: Input Dataframe
    list_of_cols: List of categorical features
    method_type: 1 (Label Encoding) or 0 (One hot encoding)
    index_order: frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc
    onehot_dropLast= True or False (Dropping last column in one hot encoding)
    pre_existing_model: True if the models exist already, False Otherwise
    model_path: If pre_existing_model is True, this argument is path for the saved models. If pre_existing_model is False, 
                this argument can be used for saving the normalization model (value other than NA). 
                Default ("NA") means there is neither pre_existing_model nor there is a need to save one.
    output_mode: replace or append
    return: Dataframe with transformed categorical features
    '''
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, OneHotEncoderModel, StringIndexerModel, OneHotEncoderEstimator
    from pyspark.ml import Pipeline, PipelineModel
    
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|') if x.strip() in idf.columns]
    else:
        list_of_cols = [e for e in list_of_cols if e in idf.columns]
    
    if method_type not in (0,1):
        raise TypeError('Invalid input for method_type')
    if output_mode not in ('replace','append'):
        raise TypeError('Invalid input for output_mode')
    if len(list_of_cols) == 0:
        warnings.warn("No Action Performed")
        return idf
        
    if pre_existing_model == True:
        pipelineModel = PipelineModel.load(model_path + "/cat_to_num_unsupervised/indexer")
    else:
        stages = []
        # Multiple columns are allowed in StringIndexer from spark3.X
        for i in list_of_cols:
            stringIndexer = StringIndexer(inputCol=i, outputCol=i + '_index', 
                                          stringOrderType=index_order, handleInvalid='keep')
            stages += [stringIndexer]
        pipeline = Pipeline(stages = stages)
        pipelineModel = pipeline.fit(idf)
    
    odf_indexed = pipelineModel.transform(idf)
           
    if method_type == 0:
        list_of_cols_vec = []
        list_of_cols_idx = []
        for i in list_of_cols:
            list_of_cols_vec.append(i+"_vec")
            list_of_cols_idx.append(i+"_index")
        if pre_existing_model == True:
            encoder = OneHotEncoderEstimator.load(model_path + "/cat_to_num_unsupervised/encoder")
        else:
            encoder = OneHotEncoderEstimator(inputCols=list_of_cols_idx, outputCols=list_of_cols_vec, 
                                             dropLast=onehot_dropLast, handleInvalid='keep')
        
        odf_encoded = encoder.fit(odf_indexed).transform(odf_indexed)

        odf = odf_encoded
        selected_cols = odf_encoded.columns
        for i in list_of_cols:
            uniq_cats = idf.select(i).distinct().count()
            def vector_to_array(v):
                from pyspark.ml.linalg import DenseVector
                v = DenseVector(v)
                new_array = list([int(x) for x in v])
                return new_array
            f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.IntegerType()))

            odf = odf.withColumn("tmp", f_vector_to_array(i+'_vec')) \
                .select(selected_cols + [F.col("tmp")[j].alias(i + "_" + str(j)) for j in range(0, uniq_cats)])
            if output_mode =='replace':
                selected_cols = [e for e in odf.columns if e not in (i,i+'_vec',i + '_index')]
            else:
                selected_cols = [e for e in odf.columns if e not in (i+'_vec',i + '_index')]
            odf = odf.select(selected_cols)
    else:
        odf = odf_indexed
        for i in list_of_cols:
            odf = odf.withColumn(i + '_index', F.when(F.col(i).isNull(), None)
                                 .otherwise(F.col(i + '_index').cast(T.IntegerType())))
        if output_mode =='replace':
            for i in list_of_cols:
                odf = odf.drop(i).withColumnRenamed(i + '_index', i)
            odf = odf.select(idf.columns)

    if (pre_existing_model == False) & (model_path != "NA"):
        pipelineModel.write().overwrite().save(model_path + "/cat_to_num_unsupervised/indexer")
        if method_type == 0:
            encoder.write().overwrite().save(model_path + "/cat_to_num_unsupervised/encoder")

    if (print_impact == True) & (method_type==1):
        print("Before")
        idf.describe().where(F.col('summary').isin('count','min','max')).show()
        print("After")
        odf.describe().where(F.col('summary').isin('count','min','max')).show()
    if (print_impact == True) & (method_type==0):
        print("Before")
        idf.printSchema()
        print("After")
        odf.printSchema()

    return odf

