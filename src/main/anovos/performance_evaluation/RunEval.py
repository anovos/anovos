from platform import machine
import yaml
import copy
import subprocess
import timeit

from anovos.workflow import ETL
from anovos.performance_evaluation.helpers.DatasetBuilder import build_dataset
from anovos.performance_evaluation.helpers.AnovosFunctionOperator import evaluate_functions
from anovos.performance_evaluation.reports.ReportBuilder import get_report
from anovos.data_ingest.data_ingest import read_dataset
from anovos.shared.spark import spark
from anovos.shared.utils import ends_with
import pandas as pd

###########
# TODO: One function, one step approach
###########


def main(all_configs, all_anovos_configs, run_type, node_count, f_name):

    dataset_name = all_configs.get("dataset_name").replace(" ", "_")
    idf_path = all_configs.get("dataset_path")
    ncols = all_configs.get("dataframe_size_list")
    output_parent_path = all_configs.get("output_parent_path")
    column_ratio = all_configs.get("column_type_ratio")
    functions = f_name.split(",")
    machine_type = all_configs.get("machine_type")

    execution_time_list = []
    if len(ncols) == 0 or ncols is None:
        start = timeit.default_timer()
        main_df = read_dataset(spark, file_path=idf_path, file_type="csv",
                               file_configs={"header": "True", "delimiter": ",", "inferSchema": "True"})

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
            idf = build_dataset(spark, idf_path,  ncol, column_ratio)
            # returns a dict containing functions,ncols and their respective execution times
            execution_time_dict = evaluate_functions(spark, all_anovos_configs, functions, idf, ncol)
            execution_time_list.append(execution_time_dict)
            print(execution_time_list)

    report_df = get_report(spark, execution_time_list, dataset_name, column_ratio, machine_type, node_count, data_read_time)

    print(report_df)
    for function in functions:
        report_path_name = ends_with(output_parent_path) + "execution_time_reports/"
        report_name = f"{dataset_name}_{str(function)}_{str(node_count)}.csv"
        report_df.to_csv(report_name, index=False)
        bash_cmd = "aws s3 cp " + report_name + " " + report_path_name
        _ = subprocess.check_output(["bash", "-c", bash_cmd])

    # viz_path_name = ends_with(output_parent_path)+ "viz/" + str(node_count) + ".csv"
    # generate_visualisations(report_df, viz_path_name)


def evaluate(eval_config_path, node_count, f_name):

    bash_cmd_2 = "aws s3 cp " + eval_config_path + " eval_config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd_2])
    eval_config_file = "eval_config.yaml"
    with open(eval_config_file, "r") as f1:
        all_configs = yaml.load(f1, yaml.SafeLoader)

    anovos_config_path = all_configs.get("anovos_function_config")
    bash_cmd_1 = "aws s3 cp " + anovos_config_path + " config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd_1])
    config_file = "config.yaml"

    with open(config_file, "r") as f2:
        all_anovos_configs = yaml.load(f2, yaml.SafeLoader)

    run_type = all_configs.get("run_type")

    main(all_configs, all_anovos_configs, run_type, node_count, f_name)
