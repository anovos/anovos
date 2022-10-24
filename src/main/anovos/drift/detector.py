import numpy as np
import pandas as pd
import pyspark
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from anovos.data_transformer.transformers import attribute_binning
from anovos.shared.utils import attributeType_segregation
from .distances import hellinger, js_divergence, ks, psi
from .validations import check_list_of_columns, check_distance_method


@check_distance_method
@check_list_of_columns
def statistics(
    spark: SparkSession,
    idf_target: DataFrame,
    idf_source: DataFrame,
    *,
    list_of_cols: list = "all",
    drop_cols: list = None,
    method_type: str = "PSI",
    bin_method: str = "equal_range",
    bin_size: int = 10,
    threshold: float = 0.1,
    pre_existing_source: bool = False,
    source_path: str = "NA",
    model_directory: str = "drift_statistics",
    run_type: str = "local",
    print_impact: bool = False,
):
    """
    When the performance of a deployed machine learning model degrades in production, one potential reason is that
    the data used in training and prediction are not following the same distribution.

    Data drift mainly includes the following manifestations:

    - Covariate shift: training and test data follow different distributions. For example, An algorithm predicting
    income that is trained on younger population but tested on older population.
    - Prior probability shift: change of prior probability. For example in a spam classification problem,
    the proportion of spam emails changes from 0.2
    in training data to 0.6 in testing data.
    - Concept shift: the distribution of the target variable changes given fixed input values. For example in
    the same spam classification problem, emails tagged as spam in training data are more likely to be tagged
    as non-spam in testing data.

    In our module, we mainly focus on covariate shift detection.

    In summary, given 2 datasets, source and target datasets, we would like to quantify the drift of some numerical
    attributes from source to target datasets. The whole process can be broken down into 2 steps: (1) convert each
    attribute of interest in source and target datasets into source and target probability distributions. (2)
    calculate the statistical distance between source and target distributions for each attribute.

    In the first step, attribute_binning is firstly performed to bin the numerical attributes of the source dataset,
    which requires two input variables: bin_method and bin_size. The same binning method is applied on the target
    dataset to align two results. The probability distributions are computed by dividing the frequency of each bin by
    the total frequency.

    In the second step, 4 choices of statistical metrics are provided to measure the data drift of an attribute from
    source to target distribution: Population Stability Index (PSI), Jensen-Shannon Divergence (JSD),
    Hellinger Distance (HD) and Kolmogorov-Smirnov Distance (KS).

    They are calculated as below:
    For two discrete probability distributions *P=(p_1,…,p_k)* and *Q=(q_1,…,q_k),*

    ![https://raw.githubusercontent.com/anovos/anovos-docs/main/docs/assets/drift_stats_formulae.png](https://raw.githubusercontent.com/anovos/anovos-docs/main/docs/assets/drift_stats_formulae.png)

    A threshold can be set to flag out drifted attributes. If multiple statistical metrics have been calculated,
    an attribute will be marked as drifted if any of its statistical metric is larger than the threshold.

    This function can be used in many scenarios. For example:

    1. Attribute level data drift can be analysed together with the attribute importance of a machine learning model.
    The more important an attribute is, the more attention it needs to be given if drift presents.
    2. To analyse data drift over time, one can treat one dataset as the source / baseline dataset and multiple
    datasets as the target datasets. Drift analysis can be performed between the source dataset and each of the
    target dataset to quantify the drift over time.

    Parameters
    ----------
    spark
        Spark Session
    idf_target
        Input Dataframe
    idf_source
        Baseline/Source Dataframe. This argument is ignored if pre_existing_source is True.
    list_of_cols
        List of columns to check drift e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all (non-array) columns for analysis.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2". (Default value = None)
    method_type
        "PSI", "JSD", "HD", "KS","all".
        "all" can be passed to calculate all drift metrics.
        One or more methods can be passed in a form of list or string where different metrics are separated
        by pipe delimiter “|” e.g. ["PSI", "JSD"] or "PSI|JSD". (Default value = "PSI")
    bin_method
        "equal_frequency", "equal_range".
        In "equal_range" method, each bin is of equal size/width and in "equal_frequency", each bin
        has equal no. of rows, though the width of bins may vary. (Default value = "equal_range")
    bin_size
        Number of bins for creating histogram. (Default value = 10)
    threshold
        A column is flagged if any drift metric is above the threshold. (Default value = 0.1)
    pre_existing_source
        Boolean argument – True or False. True if the drift_statistics folder (binning model &
        frequency counts for each attribute) exists already, False Otherwise. (Default value = False)
    source_path
        If pre_existing_source is False, this argument can be used for saving the drift_statistics folder.
        The drift_statistics folder will have attribute_binning (binning model) & frequency_counts sub-folders.
        If pre_existing_source is True, this argument is path for referring the drift_statistics folder.
        Default "NA" for temporarily saving data in "intermediate_data/" folder. (Default value = "NA")
    model_directory
        If pre_existing_source is False, this argument can be used for saving the drift stats to folder.
        The default drift statics directory is drift_statistics folder will have attribute_binning
        If pre_existing_source is True, this argument is model_directory for referring the drift statistics dir.
        Default "drift_statistics" for temporarily saving source dataset attribute_binning folder. (Default value = "drift_statistics")
    run_type
        "local", "emr", "databricks" (Default value = "local")
    print_impact
        True, False. (Default value = False)
        This argument is to print out the drift statistics of all attributes and attributes meeting the threshold.

    Returns
    -------
    DataFrame
        [attribute, *metric, flagged]
        Number of columns will be dependent on method argument. There will be one column for each drift method/metric.

    """
    drop_cols = drop_cols or []
    num_cols = attributeType_segregation(idf_target.select(list_of_cols))[0]

    if source_path == "NA":
        source_path = "intermediate_data"

    if not pre_existing_source:
        source_bin = attribute_binning(
            spark,
            idf_source,
            list_of_cols=num_cols,
            method_type=bin_method,
            bin_size=bin_size,
            pre_existing_model=False,
            model_path=source_path + "/" + model_directory,
        )
        source_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()

    target_bin = attribute_binning(
        spark,
        idf_target,
        list_of_cols=num_cols,
        method_type=bin_method,
        bin_size=bin_size,
        pre_existing_model=True,
        model_path=source_path + "/" + model_directory,
    )

    target_bin.persist(pyspark.StorageLevel.MEMORY_AND_DISK).count()
    result = {"attribute": [], "flagged": []}

    for method in method_type:
        result[method] = []

    for i in list_of_cols:
        if pre_existing_source:
            x = spark.read.csv(
                source_path + "/" + model_directory + "/frequency_counts/" + i,
                header=True,
                inferSchema=True,
            )
        else:
            x = (
                source_bin.groupBy(i)
                .agg((F.count(i) / idf_source.count()).alias("p"))
                .fillna(-1)
            )
            x.coalesce(1).write.csv(
                source_path + "/" + model_directory + "/frequency_counts/" + i,
                header=True,
                mode="overwrite",
            )

        y = (
            target_bin.groupBy(i)
            .agg((F.count(i) / idf_target.count()).alias("q"))
            .fillna(-1)
        )

        xy = (
            x.join(y, i, "full_outer")
            .fillna(0.0001, subset=["p", "q"])
            .replace(0, 0.0001)
            .orderBy(i)
        )
        p = np.array(xy.select("p").rdd.flatMap(lambda x: x).collect())
        q = np.array(xy.select("q").rdd.flatMap(lambda x: x).collect())

        result["attribute"].append(i)
        counter = 0

        for idx, method in enumerate(method_type):
            drift_function = {
                "PSI": psi,
                "JSD": js_divergence,
                "HD": hellinger,
                "KS": ks,
            }
            metric = float(round(drift_function[method](p, q), 4))
            result[method].append(metric)
            if counter == 0:
                if metric > threshold:
                    result["flagged"].append(1)
                    counter = 1
            if (idx == (len(method_type) - 1)) & (counter == 0):
                result["flagged"].append(0)

    odf = (
        spark.createDataFrame(
            pd.DataFrame.from_dict(result, orient="index").transpose()
        )
        .select(["attribute"] + method_type + ["flagged"])
        .orderBy(F.desc("flagged"))
    )

    if print_impact:
        logger.info("All Attributes:")
        odf.show(len(list_of_cols))
        logger.info("Attributes meeting Data Drift threshold:")
        drift = odf.where(F.col("flagged") == 1)
        drift.show(drift.count())

    return odf
