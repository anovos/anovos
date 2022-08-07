import timeit
from anovos.data_analyzer.association_evaluator import correlation_matrix, IV_calculation, IG_calculation, variable_clustering
from anovos.data_analyzer.quality_checker import outlier_detection
from anovos.data_transformer.transformers import cat_to_num_supervised, imputation_sklearn
from anovos.shared.spark import spark


def evaluate_functions(spark, all_anovos_configs, functions, idf, ncol="all", run_type="emr"):
    '''
    Evaluate the functions and log the time taken to execute
    '''

    execution_time_dict = {}
    execution_time_dict["ncol"] = len(idf.columns)
    function_time_dict_list = []

    for f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        execution_time = timelogger(idf, f_name, all_anovos_configs, run_type, function_time_dict_list)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")
    execution_time_dict["function_execution_data"] = function_time_dict_list
    print(f"Execution time data dictionary {execution_time_dict}")
    return execution_time_dict


def timelogger(idf, f_name, args, run_type, function_time_dict_list):
    start = timeit.default_timer()
    execute_function(idf, f_name, args, run_type)
    end = timeit.default_timer()
    execution_time = round(end - start, 4)
    function_time_dict = {"function": f_name, "execution_time": execution_time}
    function_time_dict_list.append(function_time_dict)
    return execution_time


def execute_function(idf, f_name, args, run_type):
    if f_name is "correlation_matrix":
        f_args = args.get("association_evaluator").get("correlation_matrix")
        odf = correlation_matrix(spark, idf=idf, list_of_cols=f_args.get(
            "list_of_cols"), drop_cols=f_args.get("drop_cols"))
        print(odf.head())

    if f_name is "IV_calculation":
        f_args = args.get("association_evaluator").get("IV_calculation")
        odf = IV_calculation(spark, idf, label_col=f_args.get("label_col"), event_label=f_args.get("event_label"))
        print(odf.head())

    if f_name is "IG_calculation":
        f_args = args.get("association_evaluator").get("IV_calculation")
        odf = IG_calculation(spark, idf, label_col=f_args.get("label_col"), event_label=f_args.get("event_label"))
        print(odf.head())

    if f_name is "variable_clustering":
        odf = variable_clustering(spark, idf)
        print(odf.head())

    if f_name is "outlier_detection":
        odf, _ = outlier_detection(spark, idf)
        print(odf.head())

    if f_name is "cat_to_num_supervised":
        f_args = args.get("transformers").get("categorical_encoding").get("cat_to_num_supervised")
        odf = cat_to_num_supervised(spark, idf=idf, list_of_cols=f_args.get("list_of_cols"), drop_cols=f_args.get("drop_cols"),
                                    label_col=f_args.get("label_col"), event_label=f_args.get("event_label"), run_type=run_type, print_impact=True)
        print(odf.head())

    if f_name is "imputation_sklearn":
        odf = imputation_sklearn(spark, idf=idf, run_type=run_type, print_impact=True)
        print(odf.head())
