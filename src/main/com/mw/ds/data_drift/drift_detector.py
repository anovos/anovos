from __future__ import division,print_function
import pyspark
import warnings
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Iterable
from itertools import chain
from com.mw.ds.shared.spark import *
from com.mw.ds.shared.utils import *
from com.mw.ds.data_transformer.transformers import *
from com.mw.ds.data_analyzer.quality_checker import *


def drift_statistics(idf_target,idf_source, list_of_cols='all',drop_cols=[], method_type='PSI', bin_method='equal_range', 
                     bin_size=10,threshold=None, pre_existing_source=False, source_path="NA",print_impact=False):
    '''
    :params idf_target, idf_source: Input Dataframe
    :params list_of_cols: List of columns to check drift (list or string of col names separated by |)
                          all - to include all non-array columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)  
    :params method: PSI,JSD, HD,KS (list or string of methods separated by |)
                    all - to calculate all metrics 
    :params bin_method: equal_frequency, equal_range
    :params bin_size: 10 - 20 (recommended for PSI), >100 (other method types)
    :params threshold: To flag attributes meeting drift threshold
    :params pre_existing_source: True if binning model & frequency counts/attribute exists already, False Otherwise. 
    :params source_path: If pre_existing_source is True, this argument is path for the source dataset details - drift_statistics folder.
                  drift_statistics folder must contain attribute_binning & frequency_counts folders
                  If pre_existing_source is False, this argument can be used for saving the details. 
                  Default "NA" for temporarily saving source dataset attribute_binning folder 
    :return: Output Dataframe <attribute, <metric>>
    '''
    
    import numpy as np
    import pandas as pd
    import math
    
    if list_of_cols == 'all':
        num_cols, cat_cols, other_cols = attributeType_segregation(idf_target)
        list_of_cols = num_cols + cat_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]
    
    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf_target.columns for x in list_of_cols) | (len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')
    
    if method_type == 'all':
        method_type = ['PSI','JSD','HD','KS']
    if isinstance(method_type, str):
        method_type = [x.strip() for x in method_type.split('|')]
    if any(x not in ("PSI","JSD","HD","KS") for x in method_type):
        raise TypeError('Invalid input for method_type')
    
    num_cols = attributeType_segregation(idf_target.select(list_of_cols))[0]
    
    if not pre_existing_source:
        source_bin = attribute_binning(idf_source, list_of_cols=num_cols, method_type=bin_method, bin_size=bin_size, 
                               pre_existing_model=False, model_path=source_path+"/drift_statistics")
        source_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    
    target_bin = attribute_binning(idf_target, list_of_cols=num_cols, method_type=bin_method, bin_size=bin_size,  
                               pre_existing_model=True, model_path=source_path+"/drift_statistics")
    target_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    
    
    def hellinger_distance(p, q):
        hd = math.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2)/2)
        return hd

    def PSI(p,q):
        psi = np.sum((p-q)*np.log(p/q))
        return psi

    def JS_divergence(p,q):
        def KL_divergence(p,q):
            kl = np.sum(p*np.log(p/q))
            return kl
        m = (p + q)/2
        pm = KL_divergence(p,m)
        qm = KL_divergence(q,m)
        jsd = (pm+qm)/2
        return jsd
    
    def KS_distance(p,q):
        dstats = np.max(np.abs(np.cumsum(p)-np.cumsum(q)))
        return dstats
    
    output = {'attribute': []}
    output["flagged"] = []
    for method in method_type:
        output[method] = []
    
    for i in list_of_cols:
        if pre_existing_source:
            x = spark.read.csv(source_path+"/drift_statistics/frequency_counts" + i, header=True, inferSchema=True)
        else:
            x = source_bin.groupBy(i).agg((F.count(i)/idf_source.count()).alias('p')).fillna(-1)
            if source_path != "NA":
                x.repartition(1).write.csv(source_path+"/drift_statistics/frequency_counts" + i, header=True, mode='overwrite')
            
        y = target_bin.groupBy(i).agg((F.count(i)/idf_target.count()).alias('q')).fillna(-1)
        
        xy = x.join(y,i,'full_outer').fillna(0.0001, subset=['p','q']).replace(0, 0.0001).orderBy(i)
        p = np.array(xy.select('p').rdd.flatMap(lambda x:x).collect())
        q = np.array(xy.select('q').rdd.flatMap(lambda x:x).collect())
        
        output['attribute'].append(i)
        counter = 0
        for idx, method in enumerate(method_type):
            drift_function = {'PSI': PSI,'JSD':JS_divergence,'HD': hellinger_distance,'KS': KS_distance}
            metric = float(round(drift_function[method](p,q),4))
            output[method].append(metric)
            if (threshold is None) & (counter == 0):
                output["flagged"].append("-")
                counter = 1
            if counter == 0:
                if metric > threshold:
                    output["flagged"].append(1)
                    counter = 1
            if (idx == (len(method_type) - 1)) & (counter == 0):
                output["flagged"].append(0)
    
    odf = spark.createDataFrame(pd.DataFrame.from_dict(output,orient='index').transpose())\
            .select(['attribute'] + method_type + ['flagged']).orderBy(F.desc('flagged'))
    
    if print_impact:
        print("All Attributes:")
        odf.show(len(list_of_cols))
        if threshold != None:
            print("Attributes meeting Data Drift threshold:")
            drift = odf.where(F.col('flagged')  == 1)
            drift.show(drift.count())
    
    return odf