from __future__ import division,print_function
import pyspark
import warnings
from spark import *
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Iterable
from itertools import chain
from com.mw.ds.shared.transformers import *
from com.mw.ds.data_analyzer.quality_checker import *
from com.mw.ds.shared.utils import *


def drift_statistics(idf1, idf2, list_of_cols='all', drop_cols=[], method_type='PSI', bin_method='equal_range',
                     bin_size=10, threshold=None, print_impact=False):
    '''
    :params idf1, idf2: Input Dataframe
    :params list_of_cols: List of columns to check drift (list or string of col names separated by |)
                          all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params method: PSI,JSD, HD,KS (list or string of methods separated by |)
                    all - to calculate all metrics
    :params bin_method: equal_frequency, equal_range
    :params bin_size: 10 - 20 (recommended for PSI), >100 (other method types)
    :params threshold: To flag features meeting drift threshold
    :return: Output Dataframe <feature, <metric>>
    '''

    import numpy as np
    import pandas as pd

    if list_of_cols == 'all':
        list_of_cols = list(set(idf1.columns) & set(idf2.columns))
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf1.columns for x in list_of_cols) | any(x not in idf2.columns for x in list_of_cols) | (
            len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')

    if method_type == 'all':
        method_type = ['PSI', 'JSD', 'HD', 'KS']
    if isinstance(method_type, str):
        method_type = [x.strip() for x in method_type.split('|')]
    if any(x not in ("PSI", "JSD", "HD", "KS") for x in method_type):
        raise TypeError('Invalid input for method_type')

    num_cols, cat_cols, other_cols = featureType_segregation(idf1.select(list_of_cols))

    idf1_bin = feature_binning(idf1, list_of_cols=num_cols, method_type=bin_method, bin_size=bin_size,
                               pre_existing_model=False, model_path="intermediate_data/drift_statistics")
    idf2_bin = feature_binning(idf2, list_of_cols=num_cols, method_type=bin_method, bin_size=bin_size,
                               pre_existing_model=True, model_path="intermediate_data/drift_statistics")

    idf1_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    idf2_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    def hellinger_distance(p, q):
        hd = np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / len(p)
        return hd

    def PSI(p, q):
        psi = np.sum((p - q) * np.log(p / q))
        return psi

    def JS_divergence(p, q):
        def KL_divergence(p, q):
            kl = np.sum(p * np.log(p / q))
            return kl

        m = (p + q) / 2
        pm = KL_divergence(p, m)
        qm = KL_divergence(q, m)
        jsd = (pm + qm) / 2
        return jsd

    def KS_distance(p, q):
        dstats = np.max(np.abs(np.cumsum(p) - np.cumsum(q)))
        return dstats

    output = {'feature': []}
    output["flagged"] = []
    for method in method_type:
        output[method] = []

    for i in list_of_cols:
        x = idf1_bin.groupBy(i).agg((F.count(i) / idf1.count()).alias('p'))
        y = idf2_bin.groupBy(i).agg((F.count(i) / idf2.count()).alias('q'))
        xy = x.join(y, i, 'full_outer').fillna(0.0001, subset=['p', 'q']).replace(0, 0.0001).orderBy(i)
        p = np.array(xy.select('p').rdd.flatMap(lambda x: x).collect())
        q = np.array(xy.select('q').rdd.flatMap(lambda x: x).collect())

        output['feature'].append(i)
        counter = 0
        for idx, method in enumerate(method_type):
            drift_function = {'PSI': PSI, 'JSD': JS_divergence, 'HD': hellinger_distance, 'KS': KS_distance}
            metric = float(round(drift_function[method](p, q), 4))
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

    odf = spark.createDataFrame(pd.DataFrame.from_dict(output, orient='index').transpose()) \
        .select(['feature'] + method_type + ['flagged']).orderBy(F.desc('flagged'))

    if print_impact:
        print("All Features:")
        odf.show(len(list_of_cols))
        if threshold != None:
            print("Features meeting Data Drift threshold:")
            drift = odf.where(F.col('flagged') == 1)
            drift.show(drift.count())

    return odf


def drift_classifier(idf1, idf2, list_of_cols='all', drop_cols=[], threshold=0.1, sample_size=None, print_impact=False):
    '''
    :params idf1, idf2: Input Dataframe
    :params list_of_cols: List of columns to check drift (list or string of col names separated by |)
                          all - to include all columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params threshold: Excess accuracy/AUC over 0.5
    :params sample_size: sample size to be picked from each dataframe
    return: Print Results
    '''
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
    # from statsmodels.stats.proportion import proportions_ztest

    if list_of_cols == 'all':
        list_of_cols = list(set(idf1.columns) & set(idf2.columns))
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = [e for e in list_of_cols if e not in drop_cols]

    if any(x not in idf1.columns for x in list_of_cols) | any(x not in idf2.columns for x in list_of_cols) | (
            len(list_of_cols) == 0):
        raise TypeError('Invalid input for Column(s)')

    idf1 = idf1.select(list_of_cols)
    idf2 = idf2.select(list_of_cols)

    num_cols, cat_cols, other_cols = featureType_segregation(idf1)

    if sample_size:
        df = idf1.sample(False, min(1.0, float(sample_size) / idf1.count()), 123).withColumn('_label', F.lit(1)) \
            .union(idf2.sample(False, min(1.0, float(sample_size) / idf2.count()), 123).withColumn('_label', F.lit(0))) \
            .withColumn('_id', F.monotonically_increasing_id())
    else:
        df = idf1.withColumn('_label', F.lit(1)) \
            .union(idf2.withColumn('_label', F.lit(0))) \
            .withColumn('_id', F.monotonically_increasing_id())

    df_encoded = cat_to_num_unsupervised(df, list_of_cols=cat_cols, method_type=1)
    df_imputed = imputation_MMM(df_encoded)
    df_imputed.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    req_cols = [e for e in list_of_cols if e not in ('_label', '_id')]
    assembler = VectorAssembler(inputCols=req_cols, outputCol="features")
    modelling = assembler.transform(df_imputed)
    modelling.createOrReplaceTempView("data_table")
    sqlContext.cacheTable("data_table")
    min_count = modelling.groupBy('_label').count().agg({"count": "min"}).collect()[0][0]
    count_list = sqlContext.sql("select _label, count(*) from data_table group by _label order by _label") \
        .rdd.map(lambda x: x[1]).collect()
    frac_dict = {}

    for i in range(0, 2):
        frac_dict[i] = min(1.0, round((float(min_count) / count_list[i]), 4))
    modelling = modelling.sampleBy("_label", fractions=frac_dict, seed=20)
    # modelling.groupBy('_label').count().show()
    train = modelling.sample(False, 0.8, 0)
    test = modelling.join(train, '_id', 'left_anti')

    # Building Binary Classifier
    rf = RandomForestClassifier(labelCol="_label", featuresCol="features")
    model = rf.fit(train)

    def prob_predicted(prediction, probability):
        return probability.toArray().tolist()[int(prediction)]

    f_prob_predicted = F.udf(prob_predicted, T.FloatType())
    predictions = model.transform(test) \
        .withColumn('probability', f_prob_predicted(F.col('prediction'), F.col('probability')))
    predictionsAndLabels = predictions.rdd.map(lambda lp: (float(lp.prediction), float(lp._label)))
    metrics = MulticlassMetrics(predictionsAndLabels)
    accuracy = round(metrics.accuracy, 4)
    auc = round(BinaryClassificationMetrics(predictionsAndLabels).areaUnderROC, 4)

    if print_impact:
        print("Accuracy: %s" % accuracy)
        print("AUC: %s" % auc)
        print("Confusion Matrix:")
        print(metrics.confusionMatrix().toArray())

    if threshold:
        flag_accuracy = 1 if (abs(accuracy - 0.5) > threshold) else 0
    else:
        flag_accuracy = "-"

    if threshold:
        flag_auc = 1 if (abs(auc - 0.5) > threshold) else 0
    else:
        flag_auc = "-"

    odf_metric = spark.createDataFrame([["accuracy", accuracy, flag_accuracy], ["AUC", auc, flag_auc]],
                                       schema=['metric', 'value', 'flagged'])

    output = [(i, round(float(j), 4)) for i, j in list(zip(req_cols, model.featureImportances))]
    odf_imp = spark.createDataFrame(output, schema=['feature', 'importance']).orderBy(F.desc('importance'))

    return odf_metric, odf_imp
