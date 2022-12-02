import contextlib
import copy
import glob
import os
import subprocess
import timeit

import mlflow
import yaml
from loguru import logger

from anovos.data_analyzer import association_evaluator, quality_checker, stats_generator
from anovos.data_analyzer.geospatial_analyzer import geospatial_autodetection
from anovos.data_analyzer.ts_analyzer import ts_analyzer
from anovos.data_ingest import data_ingest
from anovos.data_ingest.ts_auto_detection import ts_preprocess
from anovos.data_report import report_preprocessing
from anovos.data_report.basic_report_generation import anovos_basic_report
from anovos.data_report.report_generation import anovos_report
from anovos.data_report.report_preprocessing import save_stats
from anovos.data_transformer import transformers
from anovos.data_transformer.geospatial import (
    centroid,
    geo_format_geohash,
    geo_format_latlon,
    location_in_country,
    rog_calculation,
)
from anovos.drift_stability import drift_detector as ddetector
from anovos.drift_stability import stability as dstability
from anovos.feature_store import feast_exporter
from anovos.shared.spark import spark

mapbox_list = [
    "open-street-map",
    "white-bg",
    "carto-positron",
    "carto-darkmatter",
    "stamen-terrain",
    "stamen-toner",
    "stamen-watercolor",
]


def ETL(args):
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
    if write_configs:
        if "file_path" not in write_configs:
            raise TypeError("file path missing for writing data")

        write = copy.deepcopy(write_configs)

        run_id = write.pop("mlflow_run_id", "")
        log_mlflow = write.pop("log_mlflow", False)

        write["file_path"] = write["file_path"] + "/" + folder_name + "/" + str(run_id)
        data_ingest.write_dataset(data, **write)

        if log_mlflow:
            mlflow.log_artifacts(
                local_dir=write["file_path"], artifact_path=folder_name
            )

        if reread:
            read = copy.deepcopy(write)
            if "file_configs" in read:
                read["file_configs"].pop("repartition", None)
                read["file_configs"].pop("mode", None)
            data = data_ingest.read_dataset(spark, **read)
            return data


def stats_args(all_configs, func):
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
            "nullColumns_detection": ["stats_unique", "stats_mode", "stats_missing"],
            "variable_clustering": ["stats_mode"],
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


def main(all_configs, run_type, auth_key_val={}):
    if run_type == "ak8s":
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        conf.set("fs.wasbs.impl", "org.apache.hadoop.fs.azure.NativeAzureFileSystem")

        # Set credentials using auth_key_val
        for key, value in auth_key_val.items():
            spark.conf.set(key, value)
            auth_key = value
    else:
        auth_key = "NA"

    start_main = timeit.default_timer()
    df = ETL(all_configs.get("input_dataset"))

    write_main = all_configs.get("write_main", None)
    write_intermediate = all_configs.get("write_intermediate", None)
    write_stats = all_configs.get("write_stats", None)
    write_feast_features = all_configs.get("write_feast_features", None)

    if write_intermediate and run_type == "ak8s":
        default_root_path = write_intermediate.get("file_path", None)
    else:
        default_root_path = None

    if write_feast_features is not None:
        repartition_count = (
            write_main["file_configs"]["repartition"]
            if "file_configs" in write_main
            and "repartition" in write_main["file_configs"]
            else -1
        )
        feast_exporter.check_feast_configuration(
            write_feast_features, repartition_count
        )

    mlflow_config = all_configs.get("mlflow", None)

    if mlflow_config is not None:
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        mlflow.set_experiment(mlflow_config["experiment"])

    mlflow_run = (
        mlflow.start_run() if mlflow_config is not None else contextlib.nullcontext()
    )

    with mlflow_run:
        if mlflow_config is not None:
            mlflow_config["run_id"] = mlflow_run.info.run_id

        start_main = timeit.default_timer()
        df = ETL(all_configs.get("input_dataset"))

        write_main = all_configs.get("write_main", None)
        write_intermediate = all_configs.get("write_intermediate", None)
        write_stats = all_configs.get("write_stats", None)

        if mlflow_config:
            if write_main:
                write_main["mlflow_run_id"] = mlflow_run.info.run_id
                write_main["log_mlflow"] = mlflow_config["track_output"]
            if write_intermediate:
                write_intermediate["mlflow_run_id"] = mlflow_run.info.run_id
                write_intermediate["log_mlflow"] = mlflow_config["track_intermediates"]
            if write_stats:
                write_stats["mlflow_run_id"] = mlflow_run.info.run_id
                write_stats["log_mlflow"] = mlflow_config["track_reports"]

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
                df = data_ingest.concatenate_dataset(
                    *idfs, method_type=args.get("method")
                )
                df = save(
                    df,
                    write_intermediate,
                    folder_name="data_ingest/concatenate_dataset",
                    reread=True,
                )
                end = timeit.default_timer()
                logger.info(
                    f"{key}: execution time (in secs) = {round(end - start, 4)}"
                )
                continue

            if (key == "join_dataset") & (args is not None):
                start = timeit.default_timer()
                idfs = [df]
                for k in [
                    e for e in args.keys() if e not in ("join_type", "join_cols")
                ]:
                    tmp = ETL(args.get(k))
                    idfs.append(tmp)
                df = data_ingest.join_dataset(
                    *idfs,
                    join_cols=args.get("join_cols"),
                    join_type=args.get("join_type"),
                )
                df = save(
                    df,
                    write_intermediate,
                    folder_name="data_ingest/join_dataset",
                    reread=True,
                )
                end = timeit.default_timer()
                logger.info(
                    f"{key}: execution time (in secs) = {round(end - start, 4)}"
                )
                continue

            if (key == "geospatial_controller") & (args is not None):

                start = timeit.default_timer()

                auto_detection_analyzer_flag = args.get("geospatial_analyzer").get(
                    "auto_detection_analyzer", False
                )
                geo_transformations = args.get("geo_transformations", False)
                id_col = args.get("geospatial_analyzer").get("id_col", None)
                max_analysis_records = args.get("geospatial_analyzer").get(
                    "max_analysis_records", None
                )
                top_geo_records = args.get("geospatial_analyzer").get(
                    "top_geo_records", None
                )
                max_cluster = args.get("geospatial_analyzer").get("max_cluster", None)
                eps = args.get("geospatial_analyzer").get("eps", None)
                min_samples = args.get("geospatial_analyzer").get("min_samples", None)

                try:
                    global_map_box_val = mapbox_list.index(
                        args.get("geospatial_analyzer", None).get(
                            "global_map_box_val", None
                        )
                    )
                except:
                    global_map_box_val = 0

                if auto_detection_analyzer_flag:

                    start = timeit.default_timer()

                    lat_cols, long_cols, gh_cols = geospatial_autodetection(
                        df,
                        id_col,
                        report_input_path,
                        max_analysis_records,
                        top_geo_records,
                        max_cluster,
                        eps,
                        min_samples,
                        global_map_box_val,
                        run_type,
                        auth_key,
                    )

                    end = timeit.default_timer()
                    logger.info(
                        f"{key}, auto_detection_geospatial: execution time (in secs) ={round(end - start, 4)}"
                    )

                if geo_transformations:

                    country_val = args.get("geo_transformations").get("country", None)
                    country_shapefile_path = args.get("geo_transformations").get(
                        "country_shapefile_path", None
                    )
                    method_type = args.get("geo_transformations").get(
                        "method_type", None
                    )
                    result_prefix = args.get("geo_transformations").get(
                        "result_prefix", None
                    )
                    loc_input_format = args.get("geo_transformations").get(
                        "loc_input_format", None
                    )
                    loc_output_format = args.get("geo_transformations").get(
                        "loc_output_format", None
                    )
                    result_prefix_lat_lon = args.get("geo_transformations").get(
                        "result_prefix_lat_lon", None
                    )
                    result_prefix_geo = args.get("geo_transformations").get(
                        "result_prefix_geo", None
                    )
                    id_col = args.get("geo_transformations").get("id_col", None)

                    if ((lat_cols == []) & (long_cols == [])) or (gh_cols == []):
                        lat_cols = args.get("geo_transformations").get(
                            "list_of_lat", None
                        )
                        long_cols = args.get("geo_transformations").get(
                            "list_of_lon", None
                        )
                        gh_cols = args.get("geo_transformations").get(
                            "list_of_geohash", None
                        )

                    if args.get("geo_transformations").get(
                        "location_in_country_detection"
                    ):

                        df = location_in_country(
                            spark,
                            df,
                            lat_cols,
                            long_cols,
                            country_val,
                            country_shapefile_path,
                            method_type,
                            result_prefix_lat_lon,
                        )

                    if args.get("geo_transformations").get("geo_format_conversion"):

                        if len(lat_cols) >= 1:
                            df = geo_format_latlon(
                                df,
                                lat_cols,
                                long_cols,
                                loc_input_format,
                                loc_output_format,
                                result_prefix_geo,
                            )

                        if (len(lat_cols) >= 1) & (len(gh_cols) >= 1):
                            logger.info(
                                f"Transformation of Latitude and Longitude columns have been done. Transformation of Geohash columns will be skipped."
                            )

                        elif len(gh_cols) >= 1:
                            df = geo_format_geohash(
                                df, gh_cols, loc_output_format, result_prefix_lat_lon
                            )

                    if args.get("geo_transformations").get("centroid_calculation"):

                        for idx, i in enumerate(lat_cols):
                            df_ = centroid(df, lat_cols[idx], long_cols[idx], id_col)
                            df = df.join(df_, id_col, "inner")

                    if args.get("geo_transformations").get("rog_calculation"):

                        for idx, i in enumerate(lat_cols):
                            cols_drop = [
                                lat_cols[idx] + "_centroid",
                                long_cols[idx] + "_centroid",
                            ]
                            df_ = rog_calculation(
                                df.drop(*cols_drop),
                                lat_cols[idx],
                                long_cols[idx],
                                id_col,
                            )
                            df = df.join(df_, id_col, "inner")

                if (not auto_detection_analyzer_flag) & (not geo_transformations):
                    lat_cols, long_cols, gh_cols = [], [], []

                continue

            if (key == "timeseries_analyzer") & (args is not None):

                auto_detection_flag = args.get("auto_detection", False)
                id_col = args.get("id_col", None)
                tz_val = args.get("tz_offset", None)
                inspection_flag = args.get("inspection", False)
                analysis_level = args.get("analysis_level", None)
                max_days_limit = args.get("max_days", None)

                if auto_detection_flag:
                    start = timeit.default_timer()
                    df = ts_preprocess(
                        spark,
                        df,
                        id_col,
                        output_path=report_input_path,
                        tz_offset=tz_val,
                        run_type=run_type,
                        mlflow_config=mlflow_config,
                        auth_key=auth_key,
                    )
                    end = timeit.default_timer()
                    logger.info(
                        f"{key}, auto_detection: execution time (in secs) ={round(end - start, 4)}"
                    )

                if inspection_flag:
                    start = timeit.default_timer()
                    ts_analyzer(
                        spark,
                        df,
                        id_col,
                        max_days=max_days_limit,
                        output_path=report_input_path,
                        output_type=analysis_level,
                        tz_offset=tz_val,
                        run_type=run_type,
                        auth_key=auth_key,
                    )
                    end = timeit.default_timer()
                    logger.info(
                        f"{key}, inspection: execution time (in secs) ={round(end - start, 4)}"
                    )
                continue

            if (
                (key == "anovos_basic_report")
                & (args is not None)
                & args.get("basic_report", False)
            ):
                start = timeit.default_timer()
                anovos_basic_report(
                    spark,
                    df,
                    **args.get("report_args", {}),
                    run_type=run_type,
                    auth_key=auth_key,
                    mlflow_config=mlflow_config,
                )
                end = timeit.default_timer()
                logger.info(
                    f"Basic Report: execution time (in secs) ={round(end - start, 4)}"
                )
                continue

            if not all_configs.get("anovos_basic_report", {}).get(
                "basic_report", False
            ):
                if (key == "stats_generator") & (args is not None):
                    for m in args["metric"]:
                        start = timeit.default_timer()
                        print("\n" + m + ": \n")
                        f = getattr(stats_generator, m)
                        df_stats = f(
                            spark, df, **args["metric_args"], print_impact=False
                        )
                        if report_input_path:
                            save_stats(
                                spark,
                                df_stats,
                                report_input_path,
                                m,
                                reread=True,
                                run_type=run_type,
                                auth_key=auth_key,
                                mlflow_config=mlflow_config,
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
                            f"{key}, {m}: execution time (in secs) ={round(end - start, 4)}"
                        )

                if (key == "quality_checker") & (args is not None):
                    for subkey, value in args.items():
                        if value is not None:
                            start = timeit.default_timer()
                            print("\n" + subkey + ": \n")
                            f = getattr(quality_checker, subkey)
                            extra_args = stats_args(all_configs, subkey)
                            if subkey == "nullColumns_detection":
                                if "invalidEntries_detection" in args.keys():
                                    if args.get("invalidEntries_detection").get(
                                        "treatment", None
                                    ):
                                        extra_args["stats_missing"] = {}
                                if "outlier_detection" in args.keys():
                                    if args.get("outlier_detection").get(
                                        "treatment", None
                                    ):
                                        if (
                                            args.get("outlier_detection").get(
                                                "treatment_method", None
                                            )
                                            == "null_replacement"
                                        ):
                                            extra_args["stats_missing"] = {}

                            if subkey in ["outlier_detection", "duplicate_detection"]:
                                extra_args["print_impact"] = True
                            else:
                                extra_args["print_impact"] = False

                            df, df_stats = f(spark, df, **value, **extra_args)
                            df = save(
                                df,
                                write_intermediate,
                                folder_name="data_analyzer/quality_checker/"
                                + subkey
                                + "/dataset",
                                reread=True,
                            )
                            if report_input_path:
                                df_stats = save_stats(
                                    spark,
                                    df_stats,
                                    report_input_path,
                                    subkey,
                                    reread=True,
                                    run_type=run_type,
                                    auth_key=auth_key,
                                )
                            else:
                                df_stats = save(
                                    df_stats,
                                    write_stats,
                                    folder_name="data_analyzer/quality_checker/"
                                    + subkey,
                                    reread=True,
                                )

                            if subkey != "outlier_detection":
                                df_stats.show(100)

                            end = timeit.default_timer()
                            logger.info(
                                f"{key}, {subkey}: execution time (in secs) ={round(end - start, 4)}"
                            )

                if (key == "association_evaluator") & (args is not None):
                    for subkey, value in args.items():
                        if value is not None:
                            start = timeit.default_timer()
                            print("\n" + subkey + ": \n")
                            if subkey == "correlation_matrix":
                                f = getattr(association_evaluator, subkey)
                                extra_args = stats_args(all_configs, subkey)
                                cat_to_num_trans_params = all_configs.get(
                                    "cat_to_num_transformer", None
                                )
                                df_trans_corr = transformers.cat_to_num_transformer(
                                    spark, df, **cat_to_num_trans_params
                                )
                                df_stats = f(
                                    spark,
                                    df_trans_corr,
                                    **value,
                                    **extra_args,
                                    print_impact=False,
                                )
                            else:
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
                                    auth_key=auth_key,
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
                                f"{key}, {subkey}: execution time (in secs) ={round(end - start, 4)}"
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
                                spark,
                                df,
                                source,
                                **value["configs"],
                                print_impact=False,
                            )
                            if report_input_path:
                                save_stats(
                                    spark,
                                    df_stats,
                                    report_input_path,
                                    subkey,
                                    reread=True,
                                    run_type=run_type,
                                    auth_key=auth_key,
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
                                f"{key}, {subkey}: execution time (in secs) ={round(end - start, 4)}"
                            )

                        if (subkey == "stability_index") & (value is not None):
                            start = timeit.default_timer()
                            idfs = []
                            for k in [e for e in value.keys() if e not in ("configs")]:
                                tmp = ETL(value.get(k))
                                idfs.append(tmp)
                            df_stats = dstability.stability_index_computation(
                                spark, idfs, **value["configs"], print_impact=False
                            )
                            if report_input_path:
                                save_stats(
                                    spark,
                                    df_stats,
                                    report_input_path,
                                    subkey,
                                    reread=True,
                                    run_type=run_type,
                                    auth_key=auth_key,
                                ).show(100)
                                appended_metric_path = value["configs"].get(
                                    "appended_metric_path", ""
                                )
                                if appended_metric_path:
                                    df_metrics = data_ingest.read_dataset(
                                        spark,
                                        file_path=appended_metric_path,
                                        file_type="csv",
                                        file_configs={
                                            "header": True,
                                            "mode": "overwrite",
                                        },
                                    )
                                    save_stats(
                                        spark,
                                        df_metrics,
                                        report_input_path,
                                        "stabilityIndex_metrics",
                                        reread=True,
                                        run_type=run_type,
                                        auth_key=auth_key,
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
                                f"{key}, {subkey}: execution time (in secs) ={round(end - start, 4)}"
                            )

                    logger.info(
                        f"execution time w/o report (in sec) ={round(end - start_main, 4)}"
                    )

                if (key == "transformers") & (args is not None):
                    for subkey, value in args.items():
                        if value is not None:
                            for subkey2, value2 in value.items():
                                if value2 is not None:
                                    start = timeit.default_timer()
                                    print("\n" + subkey2 + ": \n")
                                    f = getattr(transformers, subkey2)
                                    extra_args = stats_args(all_configs, subkey2)
                                    if subkey2 in (
                                        "imputation_sklearn",
                                        "autoencoder_latentFeatures",
                                        "auto_imputation",
                                        "PCA_latentFeatures",
                                    ):
                                        extra_args["run_type"] = run_type
                                        extra_args["auth_key"] = auth_key
                                    if subkey2 == "cat_to_num_supervised":
                                        if (
                                            "model_path" not in value2.keys()
                                            and default_root_path
                                        ):
                                            extra_args["model_path"] = (
                                                default_root_path
                                                + "/intermediate_model"
                                            )
                                    if subkey2 in (
                                        "normalization",
                                        "feature_transformation",
                                        "boxcox_transformation",
                                        "expression_parser",
                                    ):
                                        df_transformed = f(
                                            df,
                                            **value2,
                                            **extra_args,
                                            print_impact=True,
                                        )
                                    elif subkey2 in "imputation_sklearn":
                                        df_transformed = f(
                                            spark,
                                            df,
                                            **value2,
                                            **extra_args,
                                            print_impact=False,
                                        )
                                    else:
                                        df_transformed = f(
                                            spark,
                                            df,
                                            **value2,
                                            **extra_args,
                                            print_impact=True,
                                        )
                                    df = save(
                                        df_transformed,
                                        write_intermediate,
                                        folder_name="data_transformer/transformers/"
                                        + subkey2,
                                        reread=True,
                                    )
                                    end = timeit.default_timer()
                                    logger.info(
                                        f"{key}, {subkey2}: execution time (in secs) ={round(end - start, 4)}"
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
                                auth_key=auth_key,
                            )
                            end = timeit.default_timer()
                            logger.info(
                                f"{key}, {subkey}: execution time (in secs) ={round(end - start, 4)}"
                            )

                if (key == "report_generation") & (args is not None):
                    start = timeit.default_timer()
                    timeseries_analyzer = all_configs.get("timeseries_analyzer", None)
                    if timeseries_analyzer:
                        analysis_level = timeseries_analyzer.get("analysis_level", None)
                    else:
                        analysis_level = None
                    geospatial_controller = all_configs.get(
                        "geospatial_controller", None
                    )
                    if not geospatial_controller:
                        lat_cols, long_cols, gh_cols = [], [], []
                        max_analysis_records, top_geo_records = None, None

                    anovos_report(
                        **args,
                        run_type=run_type,
                        output_type=analysis_level,
                        lat_cols=lat_cols,
                        long_cols=long_cols,
                        gh_cols=gh_cols,
                        max_records=max_analysis_records,
                        top_geo_records=top_geo_records,
                        auth_key=auth_key,
                        mlflow_config=mlflow_config,
                    )
                    end = timeit.default_timer()
                    logger.info(
                        f"{key}, full_report: execution time (in secs) ={round(end - start, 4)}"
                    )
        if write_feast_features is not None:
            file_source_config = write_feast_features["file_source"]
            df = feast_exporter.add_timestamp_columns(df, file_source_config)

        save(df, write_main, folder_name="final_dataset", reread=False)

        if write_feast_features is not None:
            if "file_path" not in write_feast_features:
                raise ValueError(
                    "File path missing for saving feature_store feature descriptions"
                )
            else:
                path = os.path.join(write_main["file_path"], "final_dataset", "part*")
                filename = glob.glob(path)[0]
                feast_exporter.generate_feature_description(
                    df.dtypes, write_feast_features, filename
                )


def run(config_path, run_type, auth_key_val={}):
    if run_type in ("local", "databricks", "ak8s"):
        config_file = config_path
    elif run_type == "emr":
        bash_cmd = "aws s3 cp " + config_path + " config.yaml"
        _ = subprocess.check_output(["bash", "-c", bash_cmd])
        config_file = "config.yaml"
    else:
        raise ValueError("Invalid run_type")

    if run_type == "ak8s" and auth_key_val == {}:
        raise ValueError("Invalid auth key for run_type")

    with open(config_file, "r") as f:
        all_configs = yaml.load(f, yaml.SafeLoader)

    main(all_configs, run_type, auth_key_val)
