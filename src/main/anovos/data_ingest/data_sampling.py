from pyspark.sql.functions import concat, lit
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
    if type(fraction) != float and type(fraction) != int:
        raise TypeError("Invalid input for fraction")
    if fraction <= 0 or fraction > 1:
        raise TypeError("Invalid input for fraction: fraction value is between 0 and 1")
    if type(seed_value) != int:
        raise TypeError("Invalid input for seed_value")
    if type(unique_threshold) != float and type(unique_threshold) != int:
        raise TypeError("Invalid input for unique_threshold")
    if unique_threshold <= 0:
        raise TypeError(
            "Invalid input for unique_threshold: unique_threshold value is either between 0 and 1, or an integer > 1"
        )
    if method_type not in ["stratified", "random"]:
        raise TypeError("Invalid input for data_sample method_type")
    if method_type == "stratified":
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
    for col in strata_cols:
        if col not in idf.columns:
            raise TypeError("Invalid input for strata_cols: " + col + " does not exist")
        if col not in attributeType_segregation(idf)[1] and col != "":
            if method_type == "stratified":
                if unique_threshold <= 1:
                    if float(
                        idf.select(col).distinct().count()
                    ) > unique_threshold * float(idf.select(col).count()):
                        raise TypeError(
                            "Invalid input for strata_cols: "
                            + col
                            + " can only be a categorical column"
                        )
                else:
                    if float(idf.select(col).distinct().count()) > unique_threshold:
                        raise TypeError(
                            "Invalid input for strata_cols: "
                            + col
                            + " can only be a categorical column"
                        )
    if method_type == "stratified":
        sample_df = idf.withColumn("merge", concat(*strata_cols))
        fractions = (
            sample_df.select("merge")
            .distinct()
            .withColumn("fraction", lit(fraction))
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
        odf = idf.sample(withReplacement=False, fraction=fraction)
    return odf
