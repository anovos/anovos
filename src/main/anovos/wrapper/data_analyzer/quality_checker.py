# coding=utf-8
import re
import warnings

from anovos.data_analyzer import quality_checker

def duplicate_detection(spark, idf, list_of_cols='all', drop_cols=[], treatment=False, print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, duplicate rows are removed from the input dataframe.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is de-duplicated dataframe if treated, else original input dataframe.
              Metric Dataframe is of schema [metric, value] and contains metrics - number of rows, number of unique rows, 
              number of duplicate rows and percentage of duplicate rows in total.
    """

    odf, odf_print = quality_checker.duplicate_detection(spark, idf, list_of_cols, drop_cols, treatment, print_impact)
    return odf, odf_print


def nullRows_detection(spark, idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=0.8,
                       print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, rows with high no. of null columns (defined by
                      treatment_threshold argument) are removed from the input dataframe.
    :param treatment_threshold: Defines % of columns allowed to be Null per row and takes value between 0 to 1.
                                If % of null columns is above the threshold for a row, it is removed from the dataframe.
                                There is no row removal if the threshold is 1.0. And if the threshold is 0, all rows with
                                null value are removed.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after row removal if treated, else original input dataframe.
              Metric Dataframe is of schema [null_cols_count, row_count, row_pct, flagged]. null_cols_count is defined as
              no. of missing columns in a row. row_count is no. of rows with null_cols_count missing columns.
              row_pct is row_count divided by number of rows. flagged is 1 if null_cols_count is more than
              (threshold  X Number of Columns), else 0.
    """

    odf, odf_print = quality_checker.nullRows_detection(spark, idf, list_of_cols, drop_cols, treatment, treatment_threshold, print_impact)
    return odf, odf_print


def nullColumns_detection(spark, idf, list_of_cols='missing', drop_cols=[], treatment=False,
                          treatment_method='row_removal',
                          treatment_configs={}, stats_missing={}, stats_unique={}, stats_mode={}, print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all (non-array) columns for analysis.
                         "missing" (default) can be passed to include only those columns with missing values.
                         One of the usecases where "all" may be preferable over "missing" is when the user wants to save
                         the imputation model for the future use e.g. a column may not have missing value in the training
                         dataset but missing values may possibly appear in the prediction dataset.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, missing values are treated as per treatment_method argument.
    :param treatment_method: "MMM", "row_removal", "column_removal" (more methods to be added soon).
                             MMM (Mean Median Mode) replaces null value by the measure of central tendency (mode for
                             categorical features and mean or median for numerical features).
                             row_removal removes all rows with any missing value.
                             column_removal remove a column if % of rows with missing value is above a threshold (defined
                             by key "treatment_threshold" under treatment_configs argument).
    :param treatment_configs: Takes input in dictionary format.
                              For column_removal treatment, key ‘treatment_threshold’ is provided with a value between 0 to 1.
                              For MMM, arguments corresponding to imputation_MMM function (transformer module) are provided,
                              where each key is an argument from imputation_MMM function.
                              For row_removal, this argument can be skipped.
    :param stats_missing: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                          to read pre-saved statistics on missing count/pct i.e. if measures_of_counts or
                          missingCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :param stats_mode: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                       to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
                       mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the imputed dataframe if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, missing_count, missing_pct]. missing_count is number of rows
              with null values for an attribute and missing_pct is missing_count divided by number of rows.
    """
    odf, odf_print = quality_checker.nullColumns_detection(spark, idf, list_of_cols, drop_cols, treatment,
                                    treatment_method, treatment_configs, stats_missing,
                                        stats_unique, stats_mode, print_impact)
    metric_cols = odf_print.columns
    if 'attribute' in metric_cols:
        attribute_cols = list(odf_print.select('attribute').toPandas()['attribute'])
        if len(metric_cols) < len(attribute_cols):
            for col in idf.columns:
                if col not in attribute_cols:
                    newRow = spark.createDataFrame([(col, None, None)], schema=odf_print.schema)
                    odf_print = odf_print.union(newRow)
    return odf, odf_print


def outlier_detection(spark, idf, list_of_cols='all', drop_cols=[], detection_side='upper',
                      detection_configs={'pctile_lower': 0.05, 'pctile_upper': 0.95,
                                         'stdev_lower': 3.0, 'stdev_upper': 3.0,
                                         'IQR_lower': 1.5, 'IQR_upper': 1.5,
                                         'min_validation': 2},
                      treatment=False, treatment_method='value_replacement', pre_existing_model=False,
                      model_path="NA", output_mode='replace', stats_unique={}, print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of numerical columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all numerical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param detection_side: "upper", "lower", "both".
                           "lower" detects outliers in the lower spectrum of the column range, whereas "upper" detects
                           in the upper spectrum. "Both" detects in both upper and lower end of the spectrum.
    :param detection_configs: Takes input in dictionary format with keys representing upper & lower parameter for
                              three outlier detection methodologies.
                              a) Percentile Method: In this methodology, a value higher than a certain (default 0.95)
                              percentile value is considered as an outlier. Similarly, a value lower than a certain
                              (default 0.05) percentile value is considered as an outlier.
                              b) Standard Deviation Method: In this methodology, if a value is certain number of
                              standard deviations (default 3.0) away from the mean, then it is identified as an outlier.
                              c) Interquartile Range (IQR) Method: A value which is below Q1 – k * IQR or
                              above Q3 + k * IQR (default k is 1.5) are identified as outliers, where Q1 is first quartile/
                              25th percentile, Q3 is third quartile/75th percentile and IQR is difference between
                              third quartile & first quartile.
                              If an attribute value is less (more) than its derived lower (upper) bound value,
                              it is considered as outlier by a methodology. A attribute value is considered as outlier
                              if it is declared as outlier by atleast 'min_validation' methodologies (default 2).
    :param treatment: Boolean argument – True or False. If True, outliers are treated as per treatment_method argument.
    :param treatment_method: "null_replacement", "row_removal", "value_replacement".
                             In "null_replacement", outlier values are replaced by null so that it can be imputed by a
                             reliable imputation methodology. In "value_replacement", outlier values are replaced by
                             maximum or minimum permissible value by above methodologies. Lastly in "row_removal", rows
                             are removed if it is found with any outlier.
    :param pre_existing_model: Boolean argument – True or False. True if the model with upper/lower permissible values
                               for each attribute exists already to be used, False otherwise.
    :param model_path: If pre_existing_model is True, this argument is path for the pre-saved model.
                       If pre_existing_model is False, this field can be used for saving the model.
                       Default "NA" means there is neither pre-existing model nor there is a need to save one.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with treated column. “append” option append treated
                        column to the input dataset with a postfix "_outliered" e.g. column X is appended as X_outliered.
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the imputed dataframe if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, lower_outliers, upper_outliers]. lower_outliers is no. of outliers
              found in the lower spectrum of the attribute range and upper_outliers is outlier count in the upper spectrum.
    """

    odf, odf_print = quality_checker.outlier_detection(spark, idf, list_of_cols, drop_cols, detection_side,
                      detection_configs,treatment, treatment_method, pre_existing_model,
                      model_path, output_mode, stats_unique, print_impact)
    metric_cols = odf_print.columns
    if 'attribute' in metric_cols:
        attribute_cols = list(odf_print.select('attribute').toPandas()['attribute'])
        if len(metric_cols) < len(attribute_cols):
            for col in idf.columns:
                if col not in attribute_cols:
                    newRow = spark.createDataFrame([(col, None, None    )], schema=odf_print.schema)
                    odf_print = odf_print.union(newRow)
    return odf, odf_print


def IDness_detection(spark, idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=0.8,
                     stats_unique={},
                     print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all categorical columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, columns with high IDness (defined by
                      treatment_threshold argument) are removed from the input dataframe.
    :param treatment_threshold: Defines acceptable level of IDness (calculated as no. of unique values divided by no. of
                                non-null values) for a column and takes value between 0 to 1. Default threshold
                                of 0.8 can be interpreted as remove column if its unique values count is more than
                                80% of total rows (after excluding null values).
    :param stats_unique: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                         to read pre-saved statistics on unique value count i.e. if measures_of_cardinality or
                         uniqueCount_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after column removal if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, unique_values, IDness, flagged/treated]. unique_values is no. of distinct
              values in a column, IDness is unique_values divided by no. of non-null values. A column is flagged 1
              if IDness is above the threshold, else 0.
    """

    odf, odf_print = quality_checker.IDness_detection(spark, idf, list_of_cols, drop_cols, treatment, treatment_threshold,
                     stats_unique, print_impact)
    metric_cols = odf_print.columns
    if 'attribute' in metric_cols:
        attribute_cols = list(odf_print.select('attribute').toPandas()['attribute'])
        if len(metric_cols) < len(attribute_cols):
            for col in idf.columns:
                if col not in attribute_cols:
                    newRow = spark.createDataFrame([(col, None, None, -1)], schema=odf_print.schema)
                    odf_print = odf_print.union(newRow)
    return odf, odf_print

def biasedness_detection(spark, idf, list_of_cols='all', drop_cols=[], treatment=False, treatment_threshold=0.8,
                         stats_mode={},
                         print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, columns with high biasedness (defined by
                      treatment_threshold argument) are removed from the input dataframe.
    :param treatment_threshold: Defines acceptable level of biasedness (frequency of most-frequently seen value)for
                                a column and takes value between 0 to 1. Default threshold of 0.8 can be interpreted as
                                remove column if the number of rows with most-frequently seen value is more than 80%
                                of total rows (after excluding null values).
    :param stats_mode: Takes arguments for read_dataset (data_ingest module) function in a dictionary format
                       to read pre-saved statistics on most frequently seen values i.e. if measures_of_centralTendency or
                       mode_computation (data_analyzer.stats_generator module) has been computed & saved before.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after column removal if treated, else original input dataframe.
              Metric Dataframe is of schema [attribute, mode, mode_rows, mode_pct, flagged/treated]. mode is the most frequently seen value,
              mode_rows is number of rows with mode value and mode_pct is number of rows with mode value divided by non-null values. 
              A column is flagged 1 if mode_pct is above the threshold else 0.
    """

    odf, odf_print = quality_checker.biasedness_detection(spark, idf, list_of_cols, drop_cols, treatment, treatment_threshold,
                         stats_mode, print_impact)
    metric_cols = odf_print.columns
    if 'attribute' in metric_cols:
        attribute_cols = list(odf_print.select('attribute').toPandas()['attribute'])
        if len(metric_cols) < len(attribute_cols):
            for col in idf.columns:
                if col not in attribute_cols:
                    newRow = spark.createDataFrame([(col, None, None, None, -1)], schema=odf_print.schema)
                    odf_print = odf_print.union(newRow)
    return odf, odf_print


def invalidEntries_detection(spark, idf, list_of_cols='all', drop_cols=[], treatment=False,
                             output_mode='replace', print_impact=False):
    """
    :param spark: Spark Session
    :param idf: Input Dataframe
    :param list_of_cols: List of Discrete (Categorical + Integer) columns to inspect e.g., ["col1","col2"].
                         Alternatively, columns can be specified in a string format,
                         where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
                         "all" can be passed to include all discrete columns for analysis.
                         Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
                         drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols.
    :param drop_cols: List of columns to be dropped e.g., ["col1","col2"].
                      Alternatively, columns can be specified in a string format,
                      where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
    :param treatment: Boolean argument – True or False. If True, invalid values are replaced by Null.
    :param output_mode: "replace", "append".
                        “replace” option replaces original columns with treated column. “append” option append treated
                        column to the input dataset with a postfix "_invalid" e.g. column X is appended as X_invalid.
    :return: (Output Dataframe, Metric Dataframe)
              Output Dataframe is the dataframe after treatment if applicable, else original input dataframe.
              Metric Dataframe is of schema [attribute, invalid_entries, invalid_count, invalid_pct].
              invalid_entries are all potential invalid values (separated by delimiter pipe “|”), invalid_count	is no.
              of rows which are impacted by invalid entries and invalid_pct is invalid_count divided by no of rows.
    """

    odf, odf_print = quality_checker.invalidEntries_detection(spark, idf, list_of_cols, drop_cols, treatment,
                             output_mode, print_impact)
    metric_cols = odf_print.columns
    if 'attribute' in metric_cols:
        attribute_cols = list(odf_print.select('attribute').toPandas()['attribute'])
        if len(metric_cols) < len(attribute_cols):
            for col in idf.columns:
                if col not in attribute_cols:
                    newRow = spark.createDataFrame([(col, None, None, None)], schema=odf_print.schema)
                    odf_print = odf_print.union(newRow)
    return odf, odf_print
