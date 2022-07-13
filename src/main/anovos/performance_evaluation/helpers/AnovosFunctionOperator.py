import timeit
from anovos.data_analyzer.association_evaluator import correlation_matrix, IV_calculation, IG_calculation, variable_clustering
from anovos.data_analyzer.quality_checker import outlier_detection
from anovos.data_transformer.transformers import cat_to_num_supervised, imputation_sklearn


def evaluate_functions(spark, all_anovos_configs, functions, idf, ncol="all", run_type="emr"):
    '''
    Evaluate the functions and log the time taken to execute
    '''

    execution_time_dict = {}
    execution_time_dict["ncol"] = len(idf.columns)
    function_time_dict_list = []

    f_name = "correlation_matrix"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf = correlation_matrix(spark, idf=idf, list_of_cols=all_anovos_configs.get("association_evaluator").get("correlation_matrix").get(
            "list_of_cols"), drop_cols=all_anovos_configs.get("association_evaluator").get("correlation_matrix").get("drop_cols"))
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")

    f_name = "IV_calculation"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf = IV_calculation(spark, idf, label_col=all_anovos_configs.get("association_evaluator").get("IV_calculation").get(
            "label_col"), event_label=all_anovos_configs.get("association_evaluator").get("IV_calculation").get(
            "event_label"))
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")

    f_name = "IG_calculation"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf = IG_calculation(spark, idf, label_col=all_anovos_configs.get("association_evaluator").get("IV_calculation").get(
            "label_col"), event_label=all_anovos_configs.get("association_evaluator").get("IV_calculation").get(
            "event_label"))
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")

    f_name = "variable_clustering"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf = variable_clustering(spark, idf)
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")

    f_name = "outlier_detection"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf, _ = outlier_detection(spark, idf)
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")

    f_name = "cat_to_num_supervised"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf = cat_to_num_supervised(spark, idf=idf, list_of_cols=all_anovos_configs.get("transformers").get("categorical_encoding").get(
            "cat_to_num_supervised").get("list_of_cols"), drop_cols=all_anovos_configs.get("transformers").get("categorical_encoding").get(
            "cat_to_num_supervised").get("drop_cols"),
            label_col=all_anovos_configs.get("transformers").get("categorical_encoding").get(
            "cat_to_num_supervised").get("label_col"), event_label=all_anovos_configs.get("transformers").get("categorical_encoding").get(
            "cat_to_num_supervised").get("event_label"), run_type=run_type, print_impact=True)
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")

    f_name = "imputation_sklearn"
    if f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        start = timeit.default_timer()
        odf = imputation_sklearn(spark, idf=idf, run_type=run_type, print_impact=True)
        print(odf.head())
        end = timeit.default_timer()
        execution_time = round(end - start, 4)
        function_time_dict = { "function": f_name, "execution_time": execution_time}
        function_time_dict_list.append(function_time_dict)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")
    
    execution_time_dict["function_execution_data"] = function_time_dict_list
    print(f"Execution time data dictionary {execution_time_dict}")
    return execution_time_dict
