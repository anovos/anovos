from pyspark.sql.functions import concat, lit
from anovos.shared.utils import attributeType_segregation


def data_sample(
    idf,
    list_of_cols="all",
    drop_cols=[],
    fraction=0.1,
    method_type="random",
    stratified_type="proportionate",
    seed_value=12,
    unique_threshold=0.5,
):
    if type(fraction) != float and type(fraction) != int:
        raise TypeError("Invalid input for fraction")
    if fraction < 0 or fraction > 1:
        raise TypeError("Invalid input for fraction. Fraction value is between 0 and 1")
    if type(seed_value) != int:
        raise TypeError("Invalid input for seed_value")
    if method_type not in ["stratified", "random"]:
        raise TypeError("Invalid input for data_sample method_type")
    if method_type == "stratified":
        if stratified_type not in ["proportionate", "disproportionate"]:
            raise TypeError("Invalid input for stratified_type")
    if list_of_cols == "all":
        list_of_cols = idf.columns
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    if isinstance(drop_cols, str):
        drop_cols = [x.strip() for x in drop_cols.split("|")]
    list_of_cols = list(set([e for e in list_of_cols if e not in drop_cols]))
    if len(list_of_cols) == 0:
        raise TypeError("Missing list_of_cols value")
    for col in list_of_cols:
        if col not in idf.columns:
            raise TypeError("Invalid input for list_of_col: " + col + " does not exist")
        if col not in attributeType_segregation(idf)[1] and col != "":
            if method_type == "stratified":
                if float(idf.select(col).distinct().count()) > unique_threshold * float(
                    idf.select(col).count()
                ):
                    raise TypeError(
                        "Invalid input for list_of_col: "
                        + col
                        + " can only be a categorical column"
                    )
    if method_type == "stratified":
        sample_df = idf.withColumn("merge", concat(*list_of_cols))
        fractions = (
            sample_df.select("merge")
            .distinct()
            .withColumn("fraction", lit(fraction))
            .rdd.collectAsMap()
        )
        if stratified_type == "proportionate":
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
