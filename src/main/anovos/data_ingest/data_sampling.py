import warnings

from pyspark.sql import functions as F

from anovos.shared.utils import attributeType_segregation


def data_sample(
    idf,
    strata_cols="all",
    drop_cols=[],
    fraction=0.1,
    method_type="random",
    stratified_type="population",
    seed_value=12,
    unique_threshold=0.5,
):
    """
    This is a method focus on under-sampling necessary data through multiple methods. It covers two popular sampling
    techniques - stratified sampling and random sampling

    In stratified sampling, we sample out data based on the presence of strata, determined by strata_cols. Inside
    stratified sampling, there are 2 sub-methods, called "population" and "balanced", determined by stratified_type.
    "Population" stratified sampling method is Proportionate Allocation sampling strategy, uses a sampling fraction in
    each of the strata that is proportional to that of the original dataframe. On the other hand, "Balanced" stratified
    sampling method is Optimum Allocation sampling strategy, meaning the sampling fraction of each stratum is
    not proportional to their occurrence in the original dataframe. Instead, the strata will have an equal number of
    all stratum available.

    In random sampling, we sample out data randomly, purely depends on the fraction, and seed_value that is being
    inputted

    Parameters
    ----------
    idf
        Input Dataframe
    strata_cols
        List of columns to be treated as strata e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        "all" can be passed to include all categorical columns for analysis. This is super useful instead of specifying all column names manually.
        Please note that this argument is used in conjunction with drop_cols i.e. a column mentioned in
        drop_cols argument is not considered for analysis even if it is mentioned in list_of_cols. (Default value = "all")
    drop_cols
        List of columns to be dropped e.g., ["col1","col2"].
        Alternatively, columns can be specified in a string format,
        where different column names are separated by pipe delimiter “|” e.g., "col1|col2".
        It is most useful when coupled with the “all” value of strata_cols, when we need to consider all columns except
        a few handful of them. (Default value = [])
    fraction : float
        Fraction of the data to be sampled out. (Default value = 0.1)
    method_type : str
        "stratified" for Stratified sampling, "random" for Random Sampling. (Default value = "random")
    stratified_type : str
        "population" for Proportionate Stratified Sampling, "balanced" for Optimum Stratified Sampling
    seed_value : int
        Seed value for sampling function. (Default value = 12)
    unique_threshold : float or int
        Defines threshold to skip columns with higher cardinality values from encoding.
        If unique_threshold < 1, meaning that if any column has unique records > unique_threshold * total records,
            it will be considered as high cardinality column, thus not fit to be in strata_cols
        If unique_threshold > 1, meaning that if any column has unique records > unique_threshold, it will
            be considered as high cardinality column, thus not fit to be in strata_cols. (Default value = 0.5)

    Returns
    -------
    DataFrame
        Sampled Dataframe

    """
    if type(fraction) != float and type(fraction) != int:
        raise TypeError("Invalid input for fraction")
    if fraction <= 0 or fraction > 1:
        raise TypeError("Invalid input for fraction: fraction value is between 0 and 1")
    if type(seed_value) != int:
        raise TypeError("Invalid input for seed_value")
    if method_type not in ["stratified", "random"]:
        raise TypeError("Invalid input for data_sample method_type")
    if method_type == "stratified":
        if type(unique_threshold) != float and type(unique_threshold) != int:
            raise TypeError("Invalid input for unique_threshold")
        if unique_threshold > 1 and type(unique_threshold) != int:
            raise TypeError(
                "Invalid input for unique_threshold: unique_threshold can only be integer if larger than 1"
            )
        if unique_threshold <= 0:
            raise TypeError(
                "Invalid input for unique_threshold: unique_threshold value is either between 0 and 1, or an integer > 1"
            )
        if stratified_type not in ["population", "balanced"]:
            raise TypeError("Invalid input for stratified_type")
        if strata_cols == "all":
            strata_cols = idf.columns
        if isinstance(strata_cols, str):
            strata_cols = [x.strip() for x in strata_cols.split("|")]
        if isinstance(drop_cols, str):
            drop_cols = [x.strip() for x in drop_cols.split("|")]
        strata_cols = list(set([e for e in strata_cols if e not in drop_cols]))
        if len(strata_cols) == 0:
            raise TypeError("Missing strata_cols value")
        skip_cols = []
        for col in strata_cols:
            if col not in idf.columns:
                raise TypeError(
                    "Invalid input for strata_cols: " + col + " does not exist"
                )
            if method_type == "stratified":
                if unique_threshold <= 1:
                    if float(
                        idf.select(col).distinct().count()
                    ) > unique_threshold * float(idf.select(col).count()):
                        skip_cols.append(col)
                else:
                    if float(idf.select(col).distinct().count()) > unique_threshold:
                        skip_cols.append(col)
        if skip_cols:
            warnings.warn(
                "Columns dropped from strata due to high cardinality: "
                + ",".join(skip_cols)
            )
        strata_cols = list(set([e for e in strata_cols if e not in skip_cols]))
        if len(strata_cols) == 0:
            warnings.warn(
                "No Stratified Sampling Computation - No strata column(s) to sample"
            )
            return idf
    if method_type == "stratified":
        sample_df = idf.na.drop(subset=strata_cols).withColumn(
            "merge", F.concat(*strata_cols)
        )
        fractions = (
            sample_df.select("merge")
            .distinct()
            .withColumn("fraction", F.lit(fraction))
            .rdd.collectAsMap()
        )
        if stratified_type == "population":
            odf = sample_df.stat.sampleBy("merge", fractions, seed_value).drop("merge")
        else:
            count_dict = (
                sample_df.groupby("merge").count().orderBy("count").rdd.collectAsMap()
            )
            smallest_count = int(count_dict[list(count_dict.keys())[0]])
            for key in fractions.keys():
                fractions[key] = float(fraction * smallest_count / int(count_dict[key]))
            odf = sample_df.stat.sampleBy("merge", fractions, seed_value).drop("merge")
    else:
        odf = idf.sample(withReplacement=False, fraction=fraction, seed=seed_value)
    return odf
