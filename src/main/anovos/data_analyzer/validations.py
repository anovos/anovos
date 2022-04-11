
from functools import wraps
from inspect import getcallargs
from pyspark.sql import functions as F
from pyspark.sql import types as T
from anovos.shared.utils import attributeType_segregation, discrete_attributes

def refactor_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_kwargs = getcallargs(func,*args, **kwargs)
        idf = all_kwargs.get("idf")
        list_of_cols=all_kwargs.get("list_of_cols") 
        drop_cols=all_kwargs.get("drop_cols")
        
        num_cols, cat_cols, other_cols = attributeType_segregation(idf)
        if func.__name__ in ('missingCount_computation','measures_of_counts','measures_of_centralTendency',
                             'duplicate_detection','nullRows_detection','nullColumns_detection',
                             'correlation_matrix','variable_clustering','IV_calculation','IG_calculation'):
            all_valid_cols = num_cols + cat_cols
        elif func.__name__ in ('measures_of_dispersion','measures_of_counts',
                               'measures_of_percentiles','measures_of_shape','outlier_detection'):
            all_valid_cols = num_cols
        elif func.__name__ in ('uniqueCount_computation', 'mode_computation', 'measures_of_cardinality',
                                'IDness_detection','biasedness_detection','invalidEntries_detection'):
            all_valid_cols = discrete_attributes(idf)
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
                list_of_cols = [x.strip() for x in list_of_cols.split("|") if x.strip()]
            if isinstance(drop_cols, str):
                drop_cols = [x.strip() for x in drop_cols.split("|")]
            list_of_cols = [e for e in list_of_cols if e not in drop_cols]
            if any(x not in all_valid_cols for x in list_of_cols):
                raise TypeError(f"Invalid input for column(s) in the function {func.__name__}. Column(s) not found in the dataframe: {set(list_of_cols) - set(all_valid_cols)}.")
            all_kwargs["list_of_cols"] = list_of_cols
            all_kwargs["drop_cols"] = []


        for boolarg in ("treatment","pre_existing_model","print_impact"):
            if boolarg in all_kwargs.keys():
                boolarg_val = str(all_kwargs.get(boolarg))
                if boolarg_val.lower() == "true":
                    boolarg_val = True
                elif boolarg_val.lower() == "false":
                    boolarg_val = False
                else:
                    raise TypeError(f"Non-Boolean input for {boolarg} in the function {func.__name__}.")
                all_kwargs[boolarg] = boolarg_val

        
        if "treatment_threshold" in all_kwargs:
            treatment_threshold = float(all_kwargs.get("treatment_threshold"))
            if (treatment_threshold < 0) | (treatment_threshold > 1):
                raise TypeError(f"Invalid input for Treatment Threshold Value in the function {func.__name__}. treatment_threshold should be between 0 & 1 - Received '{treatment_threshold}'.")
            all_kwargs["treatment_threshold"] = treatment_threshold

        if "output_mode" in all_kwargs.keys():
            if all_kwargs.get("output_mode") not in ("replace", "append"):
                raise TypeError(f"Invalid input for output_mode in the function {func.__name__}. output_mode should replace or append - Received '{all_kwargs.get('output_mode')}'.")


        if func.__name__ in ("nullColumns_detection","invalidEntries_detection"):
            if all_kwargs.get("treatment_method") not in ("row_removal", "MMM", "null_replacement", "column_removal","KNN","regression","MF","auto"):
                raise TypeError(f"Invalid input for method_type in the function {func.__name__}. method_type should be MMM, row_removal, column_removal, KNN, regression, MF or auto - Received '{all_kwargs.get('method_type')}'.")

            treatment_threshold = all_kwargs.get("treatment_configs").get("treatment_threshold", None)
            if treatment_threshold:
                treatment_threshold = float(treatment_threshold)
                if (treatment_threshold < 0) | (treatment_threshold > 1):
                    raise TypeError(f"Invalid input for Treatment Threshold Value in the function {func.__name__}. treatment_threshold should be between 0 & 1 - Received '{treatment_threshold}'.")
                all_kwargs["treatment_configs"]["treatment_threshold"] = treatment_threshold
            else:
                if all_kwargs.get("treatment_method") == "column_removal":
                    raise TypeError(f"Missing input for Treatment Threshold Value in the function {func.__name__}. treatment_threshold should be between 0 & 1.")


        if func.__name__ == "outlier_detection":
            if all_kwargs.get("detection_side") not in ("upper", "lower", "both"):
                raise TypeError(f"Invalid input for detection_side in the function {func.__name__}. detection_side should be upper, lower or both - Received '{all_kwargs.get('detection_side')}'.")
            if all_kwargs.get("treatment_method") not in ("null_replacement", "row_removal", "value_replacement"):
                raise TypeError(f"Invalid input for treatment_method in the function {func.__name__}. treatment_method should be null_replacement, row_removal or value_replacement - Received '{all_kwargs.get('treatment_method')}'.")
           
            detection_configs = all_kwargs.get("detection_configs")
            for arg in ["pctile_lower", "pctile_upper"]:
                if arg in detection_configs:
                    if (detection_configs[arg] < 0) | (detection_configs[arg] > 1):
                        raise TypeError(f"Invalid input for {arg} in the function {func.__name__}. {arg} should be between 0 & 1 - Received '{detection_configs[arg]}'.")


        if func.__name__ in ("IV_calculation","IG_calculation"):
            label_col = all_kwargs.get("label_col")
            event_label = all_kwargs.get("event_label")
            if label_col not in idf.columns:
                raise TypeError(f"Invalid input for Label Column in the function {func.__name__}. {label_col} not found in the dataset.")
            if idf.where(F.col(label_col) == event_label).count() == 0:
                raise TypeError(f"Invalid input for Event Label Value in the function {func.__name__}. {event_label} not found in the {label_col} column.")
            list_of_cols = list(set([e for e in list_of_cols if e not in [label_col]]))
            all_kwargs["list_of_cols"] = list_of_cols

        return func(**all_kwargs)
    return wrapper