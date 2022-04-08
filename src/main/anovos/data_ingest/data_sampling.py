from anovos.shared.utils import attributeType_segregation
from pyspark.sql.functions import concat, lit


def data_sample(
    idf,
    list_of_cols="all",
    label_col="",
    fraction=0.1,
    method_type="random",
    seed_value=0,
    unique_threshold=0.5,
):
    if type(fraction) != float:
        raise TypeError("Invalid input for fraction")
    if fraction < 0 or fraction > 1:
        raise TypeError("Invalid input for fraction. Fraction value is between 0 and 1")
    if type(seed_value) != int:
        raise TypeError("Invalid input for seed_value")
    if method_type not in ["stratified", "random"]:
        raise TypeError("Invalid input for data_sample method_type")
    if label_col not in idf.columns and label_col != "":
        raise TypeError("Invalid input for label_col: " + label_col + " does not exist")
    if label_col not in attributeType_segregation(idf)[1] and label_col != "":
        if float(idf.select(label_col).distinct().count()) > unique_threshold * float(
            idf.select(label_col).distinct().count()
        ):
            raise TypeError(
                "Invalid input for label_col: "
                + label_col
                + " can only be a categorical column"
            )
    if isinstance(list_of_cols, str):
        list_of_cols = [x.strip() for x in list_of_cols.split("|")]
    list_of_cols = list(set(list_of_cols))
    if list_of_cols == ["all"]:
        list_of_cols = idf.columns
    if method_type == "stratified":
        if len(list_of_cols) == 0:
            raise TypeError("Missing list_of_cols value for stratified method")
        for i, col in enumerate(list_of_cols):
            if col not in idf.columns:
                raise TypeError(
                    "Invalid input for list_of_col: " + col + " does not exist"
                )
            if col not in attributeType_segregation(idf)[1]:
                if float(idf.select(col).distinct().count()) > unique_threshold * float(
                    idf.select(col).distinct().count()
                ):
                    raise TypeError(
                        "Invalid input for list_of_col: "
                        + col
                        + " can only be a categorical column"
                    )
    if method_type == "stratified":
        if label_col != "":
            list_sample = []
            idf_distinct = idf.select(label_col).distinct()
            list_label_distinct = idf_distinct.rdd.flatMap(lambda x: x).collect()
            for i, value in enumerate(list_label_distinct):
                sample_df = idf.filter(idf[label_col] == value).withColumn(
                    "merge", concat(*list_of_cols)
                )
                fractions = (
                    sample_df.select("merge")
                    .distinct()
                    .withColumn("fraction", lit(fraction))
                    .rdd.collectAsMap()
                )
                output_df = sample_df.stat.sampleBy(
                    "merge", fractions, seed_value
                ).drop("merge")
                list_sample.append(output_df)
            if len(list_sample) == 1:
                odf = list_sample[0]
            else:
                start_out = list_sample[0]
                for j in range(1, len(list_sample)):
                    start_out = start_out.union(list_sample[j])
                odf = start_out
        else:
            sample_df = idf.withColumn("merge", concat(*list_of_cols))
            fractions = (
                sample_df.select("merge")
                .distinct()
                .withColumn("fraction", lit(fraction))
                .rdd.collectAsMap()
            )
            odf = sample_df.stat.sampleBy("merge", fractions, seed_value).drop("merge")
    else:
        odf = idf.sample(withReplacement=False, fraction=fraction)
    return odf
