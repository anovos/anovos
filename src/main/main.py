"""Anovos modules reflect the key components of the Machine Learning (ML) pipeline and are scalable using python API
of Spark (PySpark) - the distributed computing framework.

The key modules included in the alpha release are:

1. **Data Ingest**: This module is an ETL (Extract, transform, load) component of Anovos and helps load dataset(s) as
Spark Dataframe. It also allows performing some basic pre-processing, like selecting, deleting, renaming,
and recasting columns to ensure cleaner data is used in downstream data analysis.

2. **Data Analyzer**: This data analysis module gives a 360º view of the ingested data. It helps provide a better
understanding of the data quality and the transformations required for the modeling purpose. There are three
submodules of this module targeting specific needs of the data analysis process.

    a. *Statistics Generator*: This submodule generates all descriptive statistics related to the ingested data. The
    descriptive statistics are further broken down into different metric types such as Measures of Counts,
    Measures of Central Tendency, Measures of Cardinality, Measures of Dispersion (aka Measures of Spread in
    Statistics), Measures of Percentiles (aka Measures of Position), and Measures of Shape (aka Measures of Moments).

    b. *Quality Checker*: This submodule focuses on assessing the data quality at both row and column levels. It
    includes an option to fix identified issues with the correct treatment method. The row-level quality checks
    include duplicate detection and null detection (% columns that are missing for a row). The column level quality
    checks include outlier detection, null detection (% rows which are missing for a column), biasedness detection (
    checking if a column is biased towards one specific value), cardinality detection (checking if a
    categorical/discrete column have very high no. of unique values) and invalid entries detection which checks for
    suspicious patterns in the column values.

    c. *Association Evaluator*: This submodule focuses on understanding the interaction between different attributes
    (correlation, variable clustering) and/or the relationship between an attribute & the binary target variable (
    Information Gain, Information Value).

3. **Data Drift & Data Stability Computation**: In an ML context, data drift is the change in the distribution of the
baseline dataset that trained the model (source distribution) and the ingested data (target distribution) that makes
the prediction. Data drift is one of the primary causes of poor performance of ML models over time. This module
ensures the stability of the ingested dataset over time by analyzing it with the baseline dataset (via computing
drift statistics) and/or with historically ingested datasets (via computing stability index – currently supports only
numerical features), if available. Identifying the data drift at an early stage enables data scientists to be
proactive and fix the root cause.

4. **Data Transformer**: In the alpha release, the data transformer module only includes some basic pre-processing
functions like binning, encoding, to name a few. These functions were required to support computations of the above
key modules.  A more exhaustive set of transformations can be expected in future releases.

5. **Data Report**: This module is a visualization component of Anovos. All the analysis on the key modules is
visualized via an HTML report to get a well-rounded understanding of the ingested dataset. The report contains an
executive summary, wiki for data dictionary & metric dictionary, a tab corresponding to key modules demonstrating the
output.

Note: Upcoming Modules - Feature Wiki, Feature store, Auto ML, ML Flow Integration
"""
import copy
import subprocess
import sys
import timeit

import yaml
from loguru import logger

from anovos.data_analyzer import association_evaluator
from anovos.data_analyzer import quality_checker
from anovos.data_analyzer import stats_generator
from anovos.data_ingest import data_ingest
from anovos.data_transformer import transformers
from anovos.data_report import report_preprocessing
from anovos.data_report.basic_report_generation import anovos_basic_report
from anovos.data_report.report_generation import anovos_report
from anovos.data_report.report_preprocessing import save_stats
from anovos.drift import detector as ddetector
from anovos.shared.spark import spark


def ETL(args):
    """

    Parameters
    ----------
    args :


    Returns
    -------

    """
    f = getattr(data_ingest, "read_dataset")
    read_args = args.get("read_dataset", None)
    if read_args:
        df = f(spark, **read_args)
    else:
        raise TypeError("Invalid input for reading dataset")

    for key, value in args.items():
        if key != "read_dataset":
            if value is not None:
                f = getattr(data_ingest, key)
                if isinstance(value, dict):
                    df = f(df, **value)
                else:
                    df = f(df, value)
    return df


def save(data, write_configs, folder_name, reread=False):
    """

    Parameters
    ----------
    data :
        param write_configs:
    folder_name :
        param reread: (Default value = False)
    write_configs :

    reread :
         (Default value = False)

    Returns
    -------

    """
    if write_configs:
        if "file_path" not in write_configs:
            raise TypeError("file path missing for writing data")

        write = copy.deepcopy(write_configs)
        write["file_path"] = write["file_path"] + "/" + folder_name
        data_ingest.write_dataset(data, **write)

        if reread:
            read = copy.deepcopy(write)
            if "file_configs" in read:
                read["file_configs"].pop("repartition", None)
                read["file_configs"].pop("mode", None)
            data = data_ingest.read_dataset(spark, **read)
            return data


def stats_args(all_configs, func):
    """

    Parameters
    ----------
    all_configs :
        param func:
    func :


    Returns
    -------

    """
    stats_configs = all_configs.get("stats_generator", None)
    write_configs = all_configs.get("write_stats", None)
    report_input_path = ""
    report_configs = all_configs.get("report_preprocessing", None)
    if report_configs is not None:
        if "master_path" not in report_configs:
            raise TypeError("Master path missing for saving report statistics")
        else:
            report_input_path = report_configs.get("master_path")

    result = {}
    if stats_configs:
        mainfunc_to_args = {
            "biasedness_detection": ["stats_mode"],
            "IDness_detection": ["stats_unique"],
            "outlier_detection": ["stats_unique"],
            "correlation_matrix": ["stats_unique"],
            "nullColumns_detection": ["stats_unique", "stats_mode", "stats_missing"],
            "variable_clustering": ["stats_unique", "stats_mode"],
            "charts_to_objects": ["stats_unique"],
            "cat_to_num_unsupervised": ["stats_unique"],
            "PCA_latentFeatures": ["stats_missing"],
            "autoencoder_latentFeatures": ["stats_missing"],
        }
        args_to_statsfunc = {
            "stats_unique": "measures_of_cardinality",
            "stats_mode": "measures_of_centralTendency",
            "stats_missing": "measures_of_counts",
        }

        for arg in mainfunc_to_args.get(func, []):
            if not report_input_path:
                if write_configs:
                    read = copy.deepcopy(write_configs)
                    if "file_configs" in read:
                        read["file_configs"].pop("repartition", None)
                        read["file_configs"].pop("mode", None)

                    if read["file_type"] == "csv":
                        read["file_configs"]["inferSchema"] = True

                    read["file_path"] = (
                        read["file_path"]
                        + "/data_analyzer/stats_generator/"
                        + args_to_statsfunc[arg]
                    )
                    result[arg] = read
            else:
                result[arg] = {
                    "file_path": (
                        report_input_path + "/" + args_to_statsfunc[arg] + ".csv"
                    ),
                    "file_type": "csv",
                    "file_configs": {"header": True, "inferSchema": True},
                }

    return result


def main(all_configs, run_type):
    """

    Parameters
    ----------
    all_configs :
        param run_type:
    run_type :


    Returns
    -------

    """
    start_main = timeit.default_timer()
    df = ETL(all_configs.get("input_dataset"))

    write_main = all_configs.get("write_main", None)
    write_intermediate = all_configs.get("write_intermediate", None)
    write_stats = all_configs.get("write_stats", None)

    report_input_path = ""
    report_configs = all_configs.get("report_preprocessing", None)
    if report_configs is not None:
        if "master_path" not in report_configs:
            raise TypeError("Master path missing for saving report statistics")
        else:
            report_input_path = report_configs.get("master_path")

    for key, args in all_configs.items():

        if (key == "concatenate_dataset") & (args is not None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ("method")]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.concatenate_dataset(*idfs, method_type=args.get("method"))
            df = save(
                df,
                write_intermediate,
                folder_name="data_ingest/concatenate_dataset",
                reread=True,
            )
            end = timeit.default_timer()
            logger.info(f"{key}, execution time (in secs) = {round(end - start, 4)}")
            continue

        if (key == "join_dataset") & (args is not None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ("join_type", "join_cols")]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.join_dataset(
                *idfs, join_cols=args.get("join_cols"), join_type=args.get("join_type")
            )
            df = save(
                df,
                write_intermediate,
                folder_name="data_ingest/join_dataset",
                reread=True,
            )
            end = timeit.default_timer()
            logger.info(f"{key}, execution time (in secs) = {round(end - start, 4)}")
            continue

        if (
            (key == "anovos_basic_report")
            & (args is not None)
            & args.get("basic_report", False)
        ):
            start = timeit.default_timer()
            anovos_basic_report(
                spark, df, **args.get("report_args", {}), run_type=run_type
            )
            end = timeit.default_timer()
            logger.info(
                f"Basic Report, execution time (in secs) ={round(end - start, 4)}"
            )
            continue

        if not all_configs.get("anovos_basic_report", {}).get("basic_report", False):
            if (key == "stats_generator") & (args is not None):
                for m in args["metric"]:
                    start = timeit.default_timer()
                    logger.debug("\n" + m + ": \n")
                    f = getattr(stats_generator, m)
                    df_stats = f(spark, df, **args["metric_args"], print_impact=False)
                    if report_input_path:
                        save_stats(
                            spark,
                            df_stats,
                            report_input_path,
                            m,
                            reread=True,
                            run_type=run_type,
                        ).show(100)
                    else:
                        save(
                            df_stats,
                            write_stats,
                            folder_name="data_analyzer/stats_generator/" + m,
                            reread=True,
                        ).show(100)

                    end = timeit.default_timer()
                    logger.info(
                        f"{key}, metric:{m}, execution time (in secs) ={round(end - start, 4)}"
                    )

            if (key == "quality_checker") & (args is not None):
                for subkey, value in args.items():
                    if value is not None:
                        start = timeit.default_timer()
                        logger.debug("\n" + subkey + ": \n")
                        f = getattr(quality_checker, subkey)
                        extra_args = stats_args(all_configs, subkey)
                        df, df_stats = f(
                            spark, df, **value, **extra_args, print_impact=False
                        )
                        df = save(
                            df,
                            write_intermediate,
                            folder_name="data_analyzer/quality_checker/"
                            + subkey
                            + "/dataset",
                            reread=True,
                        )
                        if report_input_path:
                            save_stats(
                                spark,
                                df_stats,
                                report_input_path,
                                subkey,
                                reread=True,
                                run_type=run_type,
                            ).show(100)
                        else:
                            save(
                                df_stats,
                                write_stats,
                                folder_name="data_analyzer/quality_checker/" + subkey,
                                reread=True,
                            ).show(100)
                        end = timeit.default_timer()
                        logger.info(
                            f"{key} and subkey:{subkey}, execution time (in secs) ={round(end - start, 4)}"
                        )

            if (key == "association_evaluator") & (args is not None):
                for subkey, value in args.items():
                    if value is not None:
                        start = timeit.default_timer()
                        logger.debug("\n" + subkey + ": \n")
                        f = getattr(association_evaluator, subkey)
                        extra_args = stats_args(all_configs, subkey)
                        df_stats = f(
                            spark, df, **value, **extra_args, print_impact=False
                        )
                        if report_input_path:
                            save_stats(
                                spark,
                                df_stats,
                                report_input_path,
                                subkey,
                                reread=True,
                                run_type=run_type,
                            ).show(100)
                        else:
                            save(
                                df_stats,
                                write_stats,
                                folder_name="data_analyzer/association_evaluator/"
                                + subkey,
                                reread=True,
                            ).show(100)
                        end = timeit.default_timer()
                        logger.info(
                            f"{key} and subkey:{subkey}, execution time (in secs) ={round(end - start, 4)}"
                        )

            if (key == "drift_detector") & (args is not None):
                for subkey, value in args.items():

                    if (subkey == "drift_statistics") & (value is not None):
                        start = timeit.default_timer()
                        if not value["configs"]["pre_existing_source"]:
                            source = ETL(value.get("source_dataset"))
                        else:
                            source = None

                        logger.info(
                            f"running drift statistics detector using {value['configs']}"
                        )
                        df_stats = ddetector.statistics(
                            spark, df, source, **value["configs"], print_impact=False
                        )
                        if report_input_path:
                            save_stats(
                                spark,
                                df_stats,
                                report_input_path,
                                subkey,
                                reread=True,
                                run_type=run_type,
                            ).show(100)
                        else:
                            save(
                                df_stats,
                                write_stats,
                                folder_name="drift_detector/drift_statistics",
                                reread=True,
                            ).show(100)
                        end = timeit.default_timer()
                        logger.info(
                            f"{key} and subkey:{subkey}, execution time (in secs) ={round(end - start, 4)}"
                        )

                    if (subkey == "stability_index") & (value is not None):
                        start = timeit.default_timer()
                        idfs = []
                        for k in [e for e in value.keys() if e not in ("configs")]:
                            tmp = ETL(value.get(k))
                            idfs.append(tmp)
                        df_stats = ddetector.stability_index_computation(
                            spark, *idfs, **value["configs"], print_impact=False
                        )
                        if report_input_path:
                            save_stats(
                                spark,
                                df_stats,
                                report_input_path,
                                subkey,
                                reread=True,
                                run_type=run_type,
                            ).show(100)
                            appended_metric_path = value["configs"].get(
                                "appended_metric_path", ""
                            )
                            if appended_metric_path:
                                df_metrics = data_ingest.read_dataset(
                                    spark,
                                    file_path=appended_metric_path,
                                    file_type="csv",
                                    file_configs={"header": True, "mode": "overwrite"},
                                )
                                save_stats(
                                    spark,
                                    df_metrics,
                                    report_input_path,
                                    "stabilityIndex_metrics",
                                    reread=True,
                                    run_type=run_type,
                                ).show(100)
                        else:
                            save(
                                df_stats,
                                write_stats,
                                folder_name="drift_detector/stability_index",
                                reread=True,
                            ).show(100)
                        end = timeit.default_timer()
                        logger.info(
                            f"{key} and subkey:{subkey}, execution time (in secs) ={round(end - start, 4)}"
                        )

                logger.info(
                    f"execution time w/o report (in sec) ={round(end - start_main, 4)}"
                )

            if (key == "transformers") & (args != None):
                for subkey, value in args.items():
                    if value != None:
                        for subkey2, value2 in value.items():
                            if value2 != None:
                                start = timeit.default_timer()
                                print("\n" + subkey2 + ": \n")
                                f = getattr(transformers, subkey2)
                                extra_args = stats_args(all_configs, subkey2)
                                if subkey2 in (
                                    "normalization",
                                    "feature_transformation",
                                    "boxcox_transformation",
                                    "expression_parser",
                                ):
                                    df_transformed = f(df, **value2, print_impact=True)
                                else:
                                    df_transformed = f(
                                        spark, df, **value2, print_impact=True
                                    )
                                df = save(
                                    df_transformed,
                                    write_intermediate,
                                    folder_name="data_transformer/transformers/"
                                    + subkey2,
                                    reread=True,
                                )
                                end = timeit.default_timer()
                                print(
                                    key,
                                    subkey,
                                    subkey2,
                                    ", execution time (in secs) =",
                                    round(end - start, 4),
                                )

            if (key == "report_preprocessing") & (args is not None):
                for subkey, value in args.items():
                    if (subkey == "charts_to_objects") & (value is not None):
                        start = timeit.default_timer()
                        f = getattr(report_preprocessing, subkey)
                        extra_args = stats_args(all_configs, subkey)
                        f(
                            spark,
                            df,
                            **value,
                            **extra_args,
                            master_path=report_input_path,
                            run_type=run_type,
                        )
                        end = timeit.default_timer()
                        logger.info(
                            f"{key} and subkey:{subkey}, execution time (in secs) ={round(end - start, 4)}"
                        )

            if (key == "report_generation") & (args is not None):
                anovos_report(**args, run_type=run_type)

    save(df, write_main, folder_name="final_dataset", reread=False)


if __name__ == "__main__":
    config_path = sys.argv[1]
    run_type = sys.argv[2]

    if run_type in ("local", "databricks"):
        config_file = open(config_path, "r")
    elif run_type == "emr":
        bash_cmd = "aws s3 cp " + config_path + " config.yaml"
        output = subprocess.check_output(["bash", "-c", bash_cmd])
        config_file = open("config.yaml", "r")
    else:
        raise ValueError("Invalid run_type")

    all_configs = yaml.load(config_file, yaml.SafeLoader)
    main(all_configs, run_type)
