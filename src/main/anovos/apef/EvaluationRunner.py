from platform import machine
import yaml
import sys
import subprocess
import timeit
import math 
from anovos.data_ingest.data_ingest import read_dataset
from anovos.shared.spark import spark
from anovos.shared.utils import ends_with
import pandas as pd
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

    for f_name in functions:
        print(f"Start {f_name} execution time calculation(in secs) for {ncol} column(s)")
        execution_time = timelogger(idf, f_name, all_anovos_configs, run_type, function_time_dict_list)
        print(f"{f_name}: execution time (in secs) for {ncol} column(s) = {execution_time}")
    execution_time_dict["function_execution_data"] = function_time_dict_list
    print(f"Execution time data dictionary {execution_time_dict}")
    return execution_time_dict

def get_report(execution_time_list, dataset_name, column_ratio, machine_type, node_count, dataset_read_time):
    '''
    Build a csv report from the data_path uploaded by the AnovosOperator to the EMR
    '''
    report_item_list=[]
    
    for execution_time_data in execution_time_list:
        function_data = execution_time_data.get("function_execution_data")
        for function_execution_time_data in function_data:
            report_item_dict = {}
            report_item_dict["dataset_name"] = dataset_name
            report_item_dict["column_count_in_input_data"] = str(execution_time_data.get("ncol"))
            report_item_dict["column_type_distribution(cat/num)"] = str(column_ratio)
            report_item_dict["function_name"] = function_execution_time_data.get("function")
            report_item_dict["machine_type"] = machine_type
            report_item_dict["node_count"] = node_count
            report_item_dict["dataset_read_time"] = dataset_read_time
            report_item_dict["execution_time(s)"] = function_execution_time_data.get("execution_time")
            report_item_list.append(report_item_dict)

    report_df = pd.DataFrame(report_item_list)
    print(report_df)
    return report_df


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
        odf = IV_calculation(
            spark, 
            idf, 
            list_of_cols=f_args.get("list_of_cols"), 
            drop_cols=f_args.get("drop_cols"), 
            label_col=f_args.get("label_col"), 
            event_label=f_args.get("event_label"), 
            encoding_configs={
                             "bin_method": f_args.get("encoding_configs").get("bin_method"), 
                             "bin_size": f_args.get("encoding_configs").get("bin_size"), 
                             "monotonicity_check": f_args.get("encoding_configs").get("monotonicity_check")
                             },
            print_impact = False)
        print(odf.head())
   
    if f_name is "IG_calculation":
        f_args = args.get("association_evaluator").get("IG_calculation")
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


def build_dataset(spark, idf_path,  ncol, column_ratio):
    '''
    Build datasets of sizes mentioned in the config with the categorical and numerical columns in the ratio as mentioned in the config
    '''
    idf = read_dataset(spark, file_path=idf_path, file_type="csv",
                       file_configs={"header": "True", "delimiter": ",", "inferSchema": "True"})

    # TODO: Enable ncol based df building
    if(ncol == "all"):
        return idf
    else:
        return idf.select(idf.columns[:ncol])


def evaluation_helper(all_configs, all_anovos_configs, run_type, node_count, f_name):

    dataset_name = all_configs.get("dataset_configs").get("name").replace(" ", "_")
    idf_path = all_configs.get("dataset_configs").get("aws_path")
    ncols = all_configs.get("dataset_configs").get("dataset_column_coverage_pct")
    output_parent_path = all_configs.get("dataset_configs").get("aws_output_parent_path")
    column_ratio = all_configs.get("dataset_configs").get("categorical_column_count_factor")
    functions = f_name.split(",")
    machine_type = all_configs.get("emr_configs").get("executor_instance_type")

    execution_time_list = []
    if len(ncols) == 0 or ncols is None:
        start = timeit.default_timer()
        main_df = read_dataset(spark, file_path=idf_path, file_type="csv",
                               file_configs={"header": "True", "delimiter": ",", "inferSchema": "True"})
        column_count = len(main_df.columns)
        end = timeit.default_timer()
        data_read_time = round(end - start, 4)
        ncol = "all"
        print(f"Read Dataset: execution time (in secs) for {ncol} column(s) = {data_read_time}")
        execution_time_dict = evaluate_functions(spark, all_anovos_configs, functions, main_df, ncol, run_type)
        execution_time_list.append(execution_time_dict)
        print(execution_time_list)
        column_ratio = "as_is"
    else:
        for ncol in ncols:
            ncol_count = math.ceil(column_count * (ncol/100))
            idf = build_dataset(spark, idf_path,  ncol_count, column_ratio)
            # returns a dict containing functions,ncols and their respective execution times
            execution_time_dict = evaluate_functions(spark, all_anovos_configs, functions, idf, ncol)
            execution_time_list.append(execution_time_dict)
            print(execution_time_list)

    report_df = get_report(execution_time_list, dataset_name, column_ratio, machine_type, node_count, data_read_time)

    print(report_df)
    for function in functions:
        report_path_name = ends_with(output_parent_path) + "execution_time_reports/"
        report_name = f"{dataset_name}_{str(function)}_{str(node_count)}.csv"
        report_df.to_csv(report_name, index=False)
        bash_cmd = "aws s3 cp " + report_name + " " + report_path_name
        _ = subprocess.check_output(["bash", "-c", bash_cmd])

    # viz_path_name = ends_with(output_parent_path)+ "viz/" + str(node_count) + ".csv"
    # generate_visualisations(report_df, viz_path_name)


def evaluate(eval_config_path, run_type, node_count, f_name):

    bash_cmd_2 = "aws s3 cp " + eval_config_path + " eval_config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd_2])
    eval_config_file = "eval_config.yaml"
    with open(eval_config_file, "r") as f1:
        all_configs = yaml.load(f1, yaml.SafeLoader)

    anovos_config_path = all_configs.get("emr_configs").get("anovos_function_config")
    bash_cmd_1 = "aws s3 cp " + anovos_config_path + " config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd_1])
    config_file = "config.yaml"

    with open(config_file, "r") as f2:
        all_anovos_configs = yaml.load(f2, yaml.SafeLoader)


    evaluation_helper(all_configs, all_anovos_configs, run_type, node_count, f_name)

evaluate(eval_config_path=sys.argv[1], run_type=sys.argv[2], node_count=sys.argv[3], f_name=sys.argv[4])