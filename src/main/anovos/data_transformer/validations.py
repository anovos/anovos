from functools import wraps
from inspect import getcallargs
from pyspark.sql import functions as F
from pyspark.sql import types as T
from anovos.shared.utils import attributeType_segregation


def refactor_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_kwargs = getcallargs(func, *args, **kwargs)
        idf = all_kwargs.get("idf")

        if "list_of_cols" in all_kwargs.keys():
            list_of_cols = all_kwargs.get("list_of_cols")
            drop_cols = all_kwargs.get("drop_cols")

            num_cols, cat_cols, other_cols = attributeType_segregation(idf)
            if func.__name__ in ("imputation_MMM", "auto_imputation"):
                all_valid_cols = num_cols + cat_cols
            elif func.__name__ in (
                "attribute_binning",
                "monotonic_binning",
                "z_standardization",
                "IQR_standardization",
                "normalization",
                "imputation_sklearn",
                "imputation_matrixFactorization",
                "autoencoder_latentFeatures",
                "PCA_latentFeatures",
                "feature_transformation",
                "boxcox_transformation",
            ):
                all_valid_cols = num_cols
            elif func.__name__ in (
                "cat_to_num_supervised",
                "cat_to_num_unsupervised",
                "outlier_categories",
            ):
                all_valid_cols = cat_cols
            else:
                all_valid_cols = num_cols + cat_cols + other_cols

            if list_of_cols == "all":
                list_of_cols = all_valid_cols
            if list_of_cols == "missing":
                if isinstance(drop_cols, str):
                    drop_cols = [x.strip() for x in drop_cols.split("|")]
                all_kwargs["drop_cols"] = drop_cols
            else:
                if isinstance(list_of_cols, str):
                    list_of_cols = [
                        x.strip() for x in list_of_cols.split("|") if x.strip()
                    ]
                if isinstance(drop_cols, str):
                    drop_cols = [x.strip() for x in drop_cols.split("|")]
                list_of_cols = [e for e in list_of_cols if e not in drop_cols]
                if any(x not in all_valid_cols for x in list_of_cols):
                    raise TypeError(
                        f"Invalid input for column(s) in the function {func.__name__}. Invalid Column(s): {set(list_of_cols) - set(all_valid_cols)}."
                    )
                all_kwargs["list_of_cols"] = list_of_cols
                all_kwargs["drop_cols"] = []

        for boolarg in (
            "pre_existing_model",
            "print_impact",
            "standardization",
            "imputation",
        ):
            if boolarg in all_kwargs.keys():
                boolarg_val = str(all_kwargs.get(boolarg))
                if boolarg_val.lower() == "true":
                    boolarg_val = True
                elif boolarg_val.lower() == "false":
                    boolarg_val = False
                else:
                    raise TypeError(
                        f"Non-Boolean input for {boolarg} in the function {func.__name__}."
                    )
                all_kwargs[boolarg] = boolarg_val

        if "output_mode" in all_kwargs.keys():
            if all_kwargs.get("output_mode") not in ("replace", "append"):
                raise TypeError(
                    f"Invalid input for output_mode in the function {func.__name__}. output_mode should be replace or append - Received '{all_kwargs.get('output_mode')}'."
                )

        if "run_type" in all_kwargs.keys():
            if all_kwargs.get("run_type") not in ("local", "emr", "databricks"):
                raise TypeError(
                    f"Invalid input for run_type in the function {func.__name__}. run_type should be local, emr or databricks - Received '{all_kwargs.get('run_type')}'."
                )

        if func.__name__ in ("attribute_binning", "monotonic_binning"):
            method_mapping = {
                "attribute_binning": "method_type",
                "monotonic_binning": "bin_method",
            }
            if all_kwargs.get(method_mapping[func.__name__]) not in (
                "equal_frequency",
                "equal_range",
            ):
                raise TypeError(
                    f"Invalid input for {method_mapping[func.__name__]} in the function {func.__name__}. {method_mapping[func.__name__]} should be equal_frequency or equal_range - Received '{all_kwargs.get(method_mapping[func.__name__])}'."
                )

            bin_size = int(all_kwargs.get("bin_size"))
            if bin_size < 2:
                raise TypeError(
                    f"Invalid input for bin_size in the function {func.__name__}. bin_size should be atleast 2 - Received '{bin_size}'."
                )
            else:
                all_kwargs["bin_size"] = bin_size

            if all_kwargs.get("bin_dtype") not in ("numerical", "categorical"):
                raise TypeError(
                    f"Invalid input for bin_dtype in the function {func.__name__}. bin_dtype should be numerical or categorical - Received '{all_kwargs.get('bin_dtype')}'."
                )

            if func.__name__ == "monotonic_binning":
                label_col = all_kwargs.get("label_col")
                event_label = all_kwargs.get("event_label")
                if label_col not in idf.columns:
                    raise TypeError(
                        f"Invalid input for Label Column in the function {func.__name__}. {label_col} not found in the dataset."
                    )
                if idf.where(F.col(label_col) == event_label).count() == 0:
                    raise TypeError(
                        f"Invalid input for Event Label Value in the function {func.__name__}. {event_label} not found in the {label_col} column."
                    )
                list_of_cols = list(
                    set([e for e in list_of_cols if e not in [label_col]])
                )
                all_kwargs["list_of_cols"] = list_of_cols

        if func.__name__ == "cat_to_num_unsupervised":
            if all_kwargs.get("method_type") not in (0, 1):
                raise TypeError(
                    f"Invalid input for method_type in the function {func.__name__}. method_type should be 0 or 1 - Received '{all_kwargs.get('method_type')}'."
                )
            if all_kwargs.get("index_order") not in (
                "frequencyDesc",
                "frequencyAsc",
                "alphabetDesc",
                "alphabetAsc",
            ):
                raise TypeError(
                    f"Invalid input for Encoding Index Order in the function {func.__name__}. index_order should be frequencyDesc, frequencyAsc, alphabetDesc or alphabetAsc - Received '{all_kwargs.get('index_order')}'."
                )

            cardinality_threshold = int(all_kwargs.get("cardinality_threshold"))
            if cardinality_threshold < 2:
                raise TypeError(
                    f"Invalid input for Cardinality Threshold Value in the function {func.__name__}. cardinality_threshold should be greater than 2 - Received '{cardinality_threshold}'."
                )
            all_kwargs["cardinality_threshold"] = cardinality_threshold

        if func.__name__ == "cat_to_num_supervised":
            label_col = all_kwargs.get("label_col")
            event_label = all_kwargs.get("event_label")
            if label_col not in idf.columns:
                raise TypeError(
                    f"Invalid input for Label Column in the function {func.__name__}. {label_col} not found in the dataset."
                )
            if idf.where(F.col(label_col) == event_label).count() == 0:
                raise TypeError(
                    f"Invalid input for Event Label Value in the function {func.__name__}. {event_label} not found in the {label_col} column."
                )
            list_of_cols = list(set([e for e in list_of_cols if e not in [label_col]]))
            all_kwargs["list_of_cols"] = list_of_cols

        if func.__name__ == "imputation_MMM":
            if all_kwargs.get("method_type") not in ("mode", "mean", "median"):
                raise TypeError(
                    f"Invalid input for method_type in the function {func.__name__}. method_type should be mode, mean or median - Received '{all_kwargs.get('method_type')}'."
                )

        if func.__name__ == "imputation_sklearn":
            if all_kwargs.get("method_type") not in ("KNN", "regression"):
                raise TypeError(
                    f"Invalid input for method_type in the function {func.__name__}. method_type should be KNN or regression - Received '{all_kwargs.get('method_type')}'."
                )

            sample_size_val = int(all_kwargs.get("sample_size"))
            if sample_size_val < 1:
                raise TypeError(
                    f"Invalid input for sample_size in the function {func.__name__}. sample_size should be greater than 1 - Received '{sample_size_val}'."
                )
            all_kwargs["sample_size"] = sample_size_val

        if func.__name__ == "imputation_matrixFactorization":
            if all_kwargs.get("id_col"):
                if all_kwargs.get("id_col") not in idf.columns:
                    raise TypeError(
                        f"Invalid input for ID column in the function {func.__name__}. Received '{all_kwargs.get('id_col')}'."
                    )

        if func.__name__ == "auto_imputation":
            null_pct = float(all_kwargs.get("null_pct"))
            if (null_pct <= 0) | (null_pct >= 1):
                raise TypeError(
                    f"Invalid input for null_pct in the function {func.__name__}. null_pct should be between 0 & 1 - Received '{all_kwargs.get('null_pct')}'."
                )
            all_kwargs["null_pct"] = null_pct

        if func.__name__ == "autoencoder_latentFeatures":
            for intarg in ("sample_size", "epochs", "batch_size"):
                intarg_val = int(all_kwargs.get(intarg))
                if intarg_val < 1:
                    raise TypeError(
                        f"Invalid input for {intarg} Value in the function {func.__name__}. {intarg} should be greater than 1 - Received '{intarg_val}'."
                    )
                all_kwargs[intarg] = intarg_val

            reduction_params = float(all_kwargs.get("reduction_params"))
            if (reduction_params <= 0) | (reduction_params >= 1):
                raise TypeError(
                    f"Invalid input for reduction_params in the function {func.__name__}. reduction_params should be between 0 & 1 - Received '{all_kwargs.get('reduction_params')}'."
                )
            all_kwargs["reduction_params"] = reduction_params

        if func.__name__ == "PCA_latentFeatures":
            explained_variance_cutoff = float(
                all_kwargs.get("explained_variance_cutoff")
            )
            if (explained_variance_cutoff < 0) | (explained_variance_cutoff > 1):
                raise TypeError(
                    f"Invalid input for explained_variance_cutoff in the function {func.__name__}. explained_variance_cutoff should be between 0 & 1 - Received '{explained_variance_cutoff}'."
                )
            all_kwargs["explained_variance_cutoff"] = explained_variance_cutoff

        if func.__name__ == "feature_transformation":
            if all_kwargs.get("method_type") not in (
                "ln",
                "log10",
                "log2",
                "exp",
                "powOf2",
                "powOf10",
                "powOfN",
                "sqrt",
                "cbrt",
                "sq",
                "cb",
                "toPowerN",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "radians",
                "remainderDivByN",
                "factorial",
                "mul_inv",
                "floor",
                "ceil",
                "roundN",
            ):
                raise TypeError(
                    f"Invalid input for method_type ({all_kwargs.get('method_type')}) in the function {func.__name__}."
                )

            if all_kwargs.get("method_type") in (
                "powOfN",
                "toPowerN",
                "remainderDivByN",
                "roundN",
            ):
                if all_kwargs.get("N"):
                    all_kwargs["N"] = int(all_kwargs.get("N"))
                else:
                    raise TypeError(
                        f"Missing input for N in the function {func.__name__}."
                    )

        if func.__name__ == "boxcox_transformation":
            col_mins = idf.select([F.min(i) for i in list_of_cols])
            if any([i <= 0 for i in col_mins.rdd.flatMap(lambda x: x).collect()]):
                col_mins.show(1, False)
                raise ValueError(
                    f"Data must be positive for the function {func.__name__}"
                )

            boxcox_lambda = all_kwargs.get("boxcox_lambda")
            if boxcox_lambda is not None:
                if isinstance(boxcox_lambda, (list, tuple)) | isinstance(
                    boxcox_lambda, (float, int)
                ):
                    if isinstance(boxcox_lambda, (list, tuple)):
                        if (len(boxcox_lambda) != len(list_of_cols)) | (
                            not all(
                                [isinstance(l, (float, int)) for l in boxcox_lambda]
                            )
                        ):
                            raise TypeError(
                                f"Invalid input for boxcox_lambda in the function {func.__name__}"
                            )
                else:
                    raise TypeError(
                        "Invalid input for boxcox_lambda in the function {func.__name__}"
                    )

        if func.__name__ == "outlier_categories":
            coverage = float(all_kwargs.get("coverage"))
            if (coverage <= 0) | (coverage > 1):
                raise TypeError(
                    f"Invalid input for coverage in the function {func.__name__}. coverage should be between 0 & 1 - Received '{coverage}'."
                )
            all_kwargs["coverage"] = coverage

            max_category = int(all_kwargs.get("max_category"))
            if max_category < 2:
                raise TypeError(
                    f"Invalid input for Max Category Value in the function {func.__name__}. max_category should be greater than 2 - Received '{max_category}'."
                )
            all_kwargs["max_category"] = max_category

        return func(**all_kwargs)

    return wrapper
