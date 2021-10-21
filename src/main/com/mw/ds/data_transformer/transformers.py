import warnings

import pyspark
from com.mw.ds.data_analyzer.stats_generator import missingCount_computation, uniqueCount_computation
from com.mw.ds.data_ingest.data_ingest import read_dataset
from com.mw.ds.shared.spark import *
from com.mw.ds.shared.utils import attributeType_segregation, get_dtype
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, ImputerModel
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator
from pyspark.ml.linalg import DenseVector
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from scipy import stats


def attribute_binning(idf, list_of_cols='all', drop_cols=[], method_type="equal_range", bin_size=10,
                      pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical columns (in list format or string separated by |)
                         all - to include all numerical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params method_type: equal_frequency, equal_range
    :params bin_size: No of bins
    :params pre_existing_model: True if mapping values exists already, False Otherwise. 
    :params model_path: If pre_existing_model is True, this argument is path for the pre-saved model. 
                  If pre_existing_model is False, this argument can be used for saving the model. 
                  Default "NA" means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    :return: Binned Dataframe
    """

    num_cols = attributeType_segregation(idf)[0]
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
        warnings.warn("No Transformation Performed - Binning")
        return idf

    if method_type not in ("equal_frequency", "equal_range"):
        raise TypeError('Invalid input for method_type')
    if bin_size < 2:
        raise TypeError('Invalid input for bin_size')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model:
        df_model = sqlContext.read.parquet(model_path + "/attribute_binning")
        bin_cutoffs = []
        for i in list_of_cols:
            mapped_value = df_model.where(F.col('attribute') == i).select('parameters') \
                .rdd.flatMap(lambda x: x).collect()[0]
            bin_cutoffs.append(mapped_value)
    else:
        if method_type == "equal_frequency":
            pctile_width = 1 / bin_size
            pctile_cutoff = []
            for j in range(1, bin_size):
                pctile_cutoff.append(j * pctile_width)
            bin_cutoffs = idf.approxQuantile(list_of_cols, pctile_cutoff, 0.01)

        else:
            bin_cutoffs = []
            for i in list_of_cols:
                max_val = (idf.select(F.col(i)).groupBy().max().rdd.flatMap(lambda x: x).collect() + [None])[0]
                min_val = (idf.select(F.col(i)).groupBy().min().rdd.flatMap(lambda x: x).collect() + [None])[0]
                bin_cutoff = []
                if max_val:
                    bin_width = (max_val - min_val) / bin_size
                    for j in range(1, bin_size):
                        bin_cutoff.append(min_val + j * bin_width)
                bin_cutoffs.append(bin_cutoff)

        if model_path != "NA":
            df_model = spark.createDataFrame(zip(list_of_cols, bin_cutoffs), schema=['attribute', 'parameters'])
            df_model.write.parquet(model_path + "/attribute_binning", mode='overwrite')

    def bucket_label(value, index):
        if value is None:
            return None
        for j in range(0, len(bin_cutoffs[index])):
            if value <= bin_cutoffs[index][j]:
                return j + 1
            else:
                next
        return len(bin_cutoffs[0]) + 1

    f_bucket_label = F.udf(bucket_label, T.IntegerType())

    odf = idf
    for idx, i in enumerate(list_of_cols):
        odf = odf.withColumn(i + "_binned", f_bucket_label(F.col(i), F.lit(idx)))

        if idx % 5 == 0:
            odf.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    if output_mode == 'replace':
        for col in list_of_cols:
            odf = odf.drop(col).withColumnRenamed(col + "_binned", col)

    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_binned") for i in list_of_cols]
        uniqueCount_computation(odf, output_cols).show(len(output_cols))
    return odf


def monotonic_encoding(idf, list_of_cols='all', drop_cols=[], label_col='label',
                       event_label=1, bin_method="equal_frequency", bin_size=10):
    """
    :params idf: Input Dataframe
    :params list_of_cols: Numerical columns (in list format or string separated by |)
                         all - to include all numerical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params label_col: Label column
    :params event_label: Value of event (binary classfication)
    :params bin_method: equal_frequency, equal_range
    :params bin_size: No of bins
    :return: Binned Dataframe
    """
    num_cols = attributeType_segregation(idf)[0]
    if list_of_cols == 'all':
        list_of_cols = num_cols
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split('|')]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split('|')]

    list_of_cols = [e for e in list_of_cols if e not in (drop_cols + [label_col])]

    if any(x not in num_cols for x in list_of_cols):
        raise TypeError('Invalid input for Column(s)')

    attribute_binning(idf, list_of_cols='all', drop_cols=[], method_type="equal_range", bin_size=10,
                      pre_existing_model=False, model_path="NA", output_mode="replace", print_impact=False)

    idf_encoded = idf
    for col in list_of_cols:
        n = 20  # max_bin
        r = 0
        while n > 2:
            tmp = attribute_binning(idf, [col], drop_cols=[], method_type=bin_method, bin_size=n, output_mode='append') \
                .select(label_col, col, col + '_binned') \
                .withColumn(label_col, F.when(F.col(label_col) == event_label, 1).otherwise(0)) \
                .groupBy(col + '_binned').agg(F.avg(col).alias('mean_val'),
                                              F.avg(label_col).alias('mean_label')).dropna()
            # r = tmp.stat.corr('mean_age','mean_label')
            r, p = stats.spearmanr(tmp.toPandas()[['mean_val']], tmp.toPandas()[['mean_label']])
            if r == 1.0:
                idf_encoded = attribute_binning(idf_encoded, [col], drop_cols=[], method_type=bin_method, bin_size=n)
                break
            n = n - 1
            r = 0
        if r < 1.0:
            idf_encoded = attribute_binning(idf_encoded, [col], drop_cols=[], method_type=bin_method, bin_size=bin_size)

    return idf_encoded


def cat_to_num_unsupervised(idf, list_of_cols='all', drop_cols=[], method_type=1, index_order='frequencyDesc',
                            onehot_dropLast=False,
                            pre_existing_model=False, model_path="NA", output_mode='replace', print_impact=False):
    '''
    :params idf: Input Dataframe
    :params list_of_cols: Categorical columns (in list format or string separated by |)
                         all - to include all categorical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params method_type: 1 (Label Encoding) or 0 (One hot encoding)
    :params index_order: Valid only for Label Encoding - frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc
    :params onehot_dropLast= True or False (Dropping last column in one hot encoding)
    :params pre_existing_model: True if the models exist already, False Otherwise
    :params model_path: If pre_existing_model is True, this argument is path for the pre-saved models. If pre_existing_model is False, 
                this argument can be used for saving the normalization model (value other than NA). 
                Default ("NA") means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    :return: Dataframe with transformed categorical attributes
    '''

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
        warnings.warn("No Encoding Computation")
        return idf
    if method_type not in (0, 1):
        raise TypeError('Invalid input for method_type')
    if index_order not in ('frequencyDesc', 'frequencyAsc', 'alphabetDesc', 'alphabetAsc'):
        raise TypeError('Invalid input for Encoding Index Order')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model:
        pipelineModel = PipelineModel.load(model_path + "/cat_to_num_unsupervised/indexer")
    else:
        stages = []
        # Multiple columns are allowed in StringIndexer from spark3.X
        for i in list_of_cols:
            stringIndexer = StringIndexer(inputCol=i, outputCol=i + '_index',
                                          stringOrderType=index_order, handleInvalid='keep')
            stages += [stringIndexer]
        pipeline = Pipeline(stages=stages)
        pipelineModel = pipeline.fit(idf)

    odf_indexed = pipelineModel.transform(idf)

    if method_type == 0:
        list_of_cols_vec = []
        list_of_cols_idx = []
        for i in list_of_cols:
            list_of_cols_vec.append(i + "_vec")
            list_of_cols_idx.append(i + "_index")
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
                v = DenseVector(v)
                new_array = list([int(x) for x in v])
                return new_array

            f_vector_to_array = F.udf(vector_to_array, T.ArrayType(T.IntegerType()))

            odf = odf.withColumn("tmp", f_vector_to_array(i + '_vec')) \
                .select(selected_cols + [F.col("tmp")[j].alias(i + "_" + str(j)) for j in range(0, uniq_cats)])
            if output_mode == 'replace':
                selected_cols = [e for e in odf.columns if e not in (i, i + '_vec', i + '_index')]
            else:
                selected_cols = [e for e in odf.columns if e not in (i + '_vec', i + '_index')]
            odf = odf.select(selected_cols)
    else:
        odf = odf_indexed
        for i in list_of_cols:
            odf = odf.withColumn(i + '_index', F.when(F.col(i).isNull(), None)
                                 .otherwise(F.col(i + '_index').cast(T.IntegerType())))
        if output_mode == 'replace':
            for i in list_of_cols:
                odf = odf.drop(i).withColumnRenamed(i + '_index', i)
            odf = odf.select(idf.columns)

    if (pre_existing_model == False) & (model_path != "NA"):
        pipelineModel.write().overwrite().save(model_path + "/cat_to_num_unsupervised/indexer")
        if method_type == 0:
            encoder.write().overwrite().save(model_path + "/cat_to_num_unsupervised/encoder")

    if (print_impact == True) & (method_type == 1):
        print("Before")
        idf.describe().where(F.col('summary').isin('count', 'min', 'max')).show()
        print("After")
        odf.describe().where(F.col('summary').isin('count', 'min', 'max')).show()
    if (print_impact == True) & (method_type == 0):
        print("Before")
        idf.printSchema()
        print("After")
        odf.printSchema()

    return odf


def imputation_MMM(idf, list_of_cols="missing", drop_cols=[], method_type="median", pre_existing_model=False,
                   model_path="NA",
                   output_mode="replace", stats_missing={}, stats_mode={}, print_impact=False):
    """
    :params idf: Pyspark Dataframe
    :params list_of_cols: list of columns (in list format or string separated by |)
                                 all - to include all non-array columns (excluding drop_cols)
                                 missing - all feautures with missing values (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params method_type: median (default), mean for Numerical attributes. Mode is only option for categorical attributes
    :params pre_existing_model: True if median/mean/mode values exists already, False Otherwise.
    :params model_path: If pre_existing_model is True, this argument is path for pre-saved imputation model.
                  If pre_existing_model is False, this argument can be used for saving the imputation model.
                  Default "NA" means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    :params stats_missing: read_dataset arguments to read pre-saved statistics on missing count/pct (dictionary format)
    :params stats_mode: read_dataset arguments to read pre-saved statistics on mode (dictionary format)
    :return: Imputed Dataframe
    """
    if stats_missing == {}:
        missing_df = missingCount_computation(idf)
    else:
        missing_df = read_dataset(**stats_missing).select('attribute', 'missing_count', 'missing_pct')

    missing_cols = missing_df.where(F.col('missing_count') > 0).select('attribute').rdd.flatMap(lambda x: x).collect()

    if str(pre_existing_model).lower() == 'true':
        pre_existing_model = True
    elif str(pre_existing_model).lower() == 'false':
        pre_existing_model = False
    else:
        raise TypeError('Non-Boolean input for pre_existing_model')

    if (len(missing_cols) == 0) & (pre_existing_model == False) & (model_path == "NA"):
        return idf

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
    if method_type not in ('mean', 'median'):
        raise TypeError('Invalid input for method_type')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    num_cols, cat_cols, other_cols = attributeType_segregation(idf.select(list_of_cols))

    odf = idf
    if len(num_cols) > 0:

        # Checking for Integer/Decimal Type Columns & Converting them into Float/Double Type
        recast_cols = []
        recast_type = []
        for i in num_cols:
            if get_dtype(idf, i) not in ('float', 'double'):
                odf = odf.withColumn(i, F.col(i).cast(T.DoubleType()))
                recast_cols.append(i + "_imputed")
                recast_type.append(get_dtype(idf, i))

        # Building new imputer model or uploading the existing model
        if pre_existing_model == True:
            imputerModel = ImputerModel.load(model_path + "/imputation_MMM/num_imputer-model")
        else:
            imputer = Imputer(strategy=method_type, inputCols=num_cols,
                              outputCols=[(e + "_imputed") for e in num_cols])
            imputerModel = imputer.fit(odf)

        # Applying model
        # odf = recast_column(imputerModel.transform(odf), recast_cols, recast_type)
        odf = imputerModel.transform(odf)
        for i, j in zip(recast_cols, recast_type):
            odf = odf.withColumn(i, F.col(i).cast(j))

        # Saving model if required
        if (pre_existing_model == False) & (model_path != "NA"):
            imputerModel.write().overwrite().save(model_path + "/imputation_MMM/num_imputer-model")

    if len(cat_cols) > 0:
        if pre_existing_model:
            df_model = sqlContext.read.csv(model_path + "/imputation_MMM/cat_imputer", header=True, inferSchema=True)
            parameters = []
            for i in cat_cols:
                mapped_value = \
                    df_model.where(F.col('attribute') == i).select('parameters').rdd.flatMap(lambda x: x).collect()[0]
                parameters.append(mapped_value)
            # imputerModel = ImputerModel.load(model_path + "/imputation_MMM/cat_imputer-model") #spark 3.X
        else:
            if stats_mode == {}:
                parameters = [str((idf.select(i).dropna().groupby(i).count().orderBy("count", ascending=False).first()
                                   or [None])[0]) for i in cat_cols]
                # imputer = Imputer(strategy='mode', inputCols=cat_cols, outputCols=[(e + "_imputed") for e in cat_cols]) #spark 3.X
                # imputerModel = imputer.fit(odf) #spark 3.X
            else:
                mode_df = read_dataset(**stats_mode).replace('None', None)
                parameters = [mode_df.where(F.col('attribute') == i).select('mode').rdd.flatMap(list).collect()[0] for i
                              in cat_cols]

        for index, i in enumerate(cat_cols):
            odf = odf.withColumn(i + "_imputed", F.when(F.col(i).isNull(), parameters[index]).otherwise(F.col(i)))
        # odf = imputerModel.transform(odf) #spark 3.X

        # Saving model File if required
        if (pre_existing_model == False) & (model_path != "NA"):
            df_model = sqlContext.createDataFrame(zip(cat_cols, parameters), schema=['attribute', 'parameters'])
            df_model.repartition(1).write.csv(model_path + "/imputation_MMM/cat_imputer", header=True, mode='overwrite')
            # imputerModel.write().overwrite().save(model_path + "/imputation_MMM/num_imputer-model") #spark 3.X

    for i in (num_cols + cat_cols):
        if i not in missing_cols:
            odf = odf.drop(i + "_imputed")
        elif output_mode == 'replace':
            odf = odf.drop(i).withColumnRenamed(i + "_imputed", i)

    if print_impact:
        if output_mode == 'replace':
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(odf, list_of_cols) \
                      .select('attribute', F.col("missing_count").alias("missingCount_after")), 'attribute', 'inner')
        else:
            output_cols = [(i + "_imputed") for i in (num_cols + cat_cols)]
            odf_print = missing_df.select('attribute', F.col("missing_count").alias("missingCount_before")) \
                .join(missingCount_computation(odf, output_cols) \
                      .withColumnRename('attribute', 'attribute_after') \
                      .withColumn('attribute', F.expr("substring(attribute_after, 1, length(attribute_after)-8)")) \
                      .drop('missing_pct'), 'attribute', 'inner')
        odf_print.show(len(list_of_cols))
    return odf


def outlier_categories(idf, list_of_cols='all', drop_cols=[], coverage=1.0, max_category=50,
                       pre_existing_model=False, model_path="NA", output_mode='replace', print_impact=False):
    '''
    :params idf: Input Dataframe
    :params list_of_cols: Categorical columns (in list format or string separated by |)
                         all - to include all categorical columns (excluding drop_cols)
    :params drop_cols: List of columns to be dropped (list or string of col names separated by |)
    :params coverage: Minimum % of rows mapped to actual category name and rest will be mapped to others
    :params max_category: Even if coverage is less, only (max_category - 1) categories will be mapped to actual name and rest to others.
                            Caveat is when multiple categories have same rank. Then #categories can be more than max_category
    :params pre_existing_model: outlier value for each attribute. True if model files exists already, False Otherwise
    :params model_path: If pre_existing_model is True, this argument is path for pre-saved model file. 
                  If pre_existing_model is False, this field can be used for saving the model file. 
                  param NA means there is neither pre_existing_model nor there is a need to save one.
    :params output_mode: replace or append
    "return: Dataframe after outlier treatment
    '''

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
        warnings.warn("No Outlier Categories Computation")
        return idf
    if (coverage <= 0) | (coverage > 1):
        raise TypeError('Invalid input for Coverage Value')
    if max_category < 2:
        raise TypeError('Invalid input for Maximum No. of Categories Allowed')
    if output_mode not in ('replace', 'append'):
        raise TypeError('Invalid input for output_mode')

    if pre_existing_model == True:
        df_model = sqlContext.read.csv(model_path + "/outlier_categories", header=True, inferSchema=True)
    else:
        for index, i in enumerate(list_of_cols):
            window = Window.partitionBy().orderBy(F.desc('count_pct'))
            df_cats = idf.groupBy(i).count().dropna() \
                .withColumn('count_pct', F.col('count') / F.sum('count').over(Window.partitionBy())) \
                .withColumn('rank', F.rank().over(window)) \
                .withColumn('cumu', F.sum('count_pct').over(window.rowsBetween(Window.unboundedPreceding, 0))) \
                .withColumn('lag_cumu', F.lag('cumu').over(window)).fillna(0) \
                .where(~((F.col('cumu') >= coverage) & (F.col('lag_cumu') >= coverage))) \
                .where(F.col('rank') <= (max_category - 1)) \
                .select(F.lit(i).alias('attribute'), F.col(i).alias('parameters'))
            if index == 0:
                df_model = df_cats
            else:
                df_model = df_model.union(df_cats)

    odf = idf
    for i in list_of_cols:
        parameters = df_model.where(F.col('attribute') == i).select('parameters').rdd.flatMap(lambda x: x).collect()
        if output_mode == 'replace':
            odf = odf.withColumn(i, F.when((F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)).otherwise(
                "others"))
        else:
            odf = odf.withColumn(i + "_outliered",
                                 F.when((F.col(i).isin(parameters)) | (F.col(i).isNull()), F.col(i)).otherwise(
                                     "others"))

    # Saving model File if required
    if (pre_existing_model == False) & (model_path != "NA"):
        df_model.repartition(1).write.csv(model_path + "/outlier_categories", header=True, mode='overwrite')

    if print_impact:
        if output_mode == 'replace':
            output_cols = list_of_cols
        else:
            output_cols = [(i + "_outliered") for i in list_of_cols]
        uniqueCount_computation(idf, list_of_cols).select('attribute',
                                                          F.col("unique_values").alias("uniqueValues_before")).show(
            len(list_of_cols))
        uniqueCount_computation(odf, output_cols).select('attribute',
                                                         F.col("unique_values").alias("uniqueValues_after")).show(
            len(list_of_cols))

    return odf
