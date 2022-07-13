from platform import machine
import yaml
import copy
import subprocess
import timeit

from anovos.workflow import ETL
from anovos.performance_evaluation.helpers.DatasetBuilder import build_dataset
from anovos.performance_evaluation.helpers.AnovosFunctionOperator import evaluate_functions
from anovos.performance_evaluation.reports.ReportBuilder import get_report
from anovos.shared.spark import spark
from anovos.shared.utils import ends_with


def main(all_configs, all_anovos_configs, run_type, node_count):

    dataset_name = all_configs.get("dataset_name")
    idf_path = all_configs.get("input_dataset")
    ncols = all_configs.get("dataframe_size_list")
    output_parent_path = all_configs.get("output_parent_path")
    column_ratio = all_configs.get("column_type_ratio")
    functions = all_configs.get("functions")
    machine_type = all_configs.get("machine_type")

    execution_time_list = []
    if len(ncols) == 0 or ncols is None:
        main_df = ETL(idf_path)
        execution_time_dict = evaluate_functions(spark, all_anovos_configs, functions, main_df, ncol, run_type)
        execution_time_list.append(execution_time_dict)
        column_ratio = "all"
    else:
        for ncol in ncols:
            idf = build_dataset(spark, idf_path,  ncol, column_ratio)
            # returns a dict containing functions,ncols and their respective execution times
            execution_time_dict = evaluate_functions(spark, all_anovos_configs, functions, idf, ncol)
            execution_time_list.append(execution_time_dict)

    report_df = get_report(spark, execution_time_list, dataset_name, column_ratio, machine_type, node_count)

    print(report_df)
    report_path_name = ends_with(output_parent_path) + "execution_time_reports/" + str(node_count) + ".csv"
    report_df.coalesce(1).write.format('com.databricks.spark.csv').save(report_path_name, index=False, mode='overwrite', header = 'true')

    # viz_path_name = ends_with(output_parent_path)+ "viz/" + str(node_count) + ".csv"
    # generate_visualisations(report_df, viz_path_name)


def evaluate(eval_config_path, anovos_config_path, run_type, node_count):

    bash_cmd_1 = "aws s3 cp " + anovos_config_path + " config.yaml"
    bash_cmd_2 = "aws s3 cp " + eval_config_path + " eval_config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd_1])
    _ = subprocess.check_output(["bash", "-c", bash_cmd_2])
    config_file = "config.yaml"
    eval_config_file = "eval_config.yaml"

    with open(eval_config_file, "r") as f1:
        all_configs = yaml.load(f1, yaml.SafeLoader)
    with open(config_file, "r") as f2:
        all_anovos_configs = yaml.load(f2, yaml.SafeLoader)

    main(all_configs, all_anovos_configs, run_type, node_count)
