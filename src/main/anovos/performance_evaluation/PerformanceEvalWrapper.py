from platform import machine
import yaml
import copy
import subprocess
import timeit

from anovos.workflow import ETL
from helpers.DatasetBuilder import build_dataset
from performance_evaluation.ReportBuilder import get_report
from performance_evaluation.ReportViz import visualize

def main(all_configs, all_anovos_configs):
    start_main = timeit.default_timer()
    node_count_list = all_configs.get("node_counts")
    machine_type = all_configs.get("machine_type")
    for node in node_count_list:

        main_df = ETL(all_configs.get("input_dataset"))
        df_list = build_dataset(main_df,all_configs.get("dataframe_size_list"),all_configs.get("column_type_ratio"))

        create_cluster_bash_cmd = "./src/main/anovos/performance_evaluation/scripts/creater_cluster.sh "+ node +" "+machine_type

        clusterID = subprocess.check_output(["bash", "-c", create_cluster_bash_cmd])

        for df in df_list:

            run_file_path = "src/main/anovos/performance_evaluation/helpers/AnovosFunctionOperator.py"
            output_path = "" #s3 path where the stats will be written
            add_emr_step_bash = "./src/main/anovos/performance_evaluation/scripts/add_steps.sh "+ run_file_path +" "+output_path

            report = get_report(output_path)

            visualize(report)





def evaluate(eval_config_path, anovos_config_path): 

    bash_cmd = "aws s3 cp " + anovos_config_path + " config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd])
    config_file = "config.yaml"

    with open(eval_config_path, "r") as f1:
        all_configs = yaml.load(f1, yaml.SafeLoader)
    with open(config_file, "r") as f2:
        all_anovos_configs = yaml.load(f2, yaml.SafeLoader)

    main(all_configs, all_anovos_configs)

