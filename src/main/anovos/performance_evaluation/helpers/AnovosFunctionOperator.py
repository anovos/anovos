import copy
import subprocess
import timeit

import yaml
from loguru import logger

from anovos.data_analyzer import association_evaluator, quality_checker, stats_generator
from anovos.data_analyzer.ts_analyzer import ts_analyzer
from anovos.data_ingest import data_ingest
from anovos.data_ingest.ts_auto_detection import ts_preprocess
from anovos.data_report import report_preprocessing
from anovos.data_report.basic_report_generation import anovos_basic_report
from anovos.data_report.report_generation import anovos_report
from anovos.data_report.report_preprocessing import save_stats
from anovos.data_transformer import transformers
from anovos.drift import detector as ddetector
from anovos.workflow import ETL
from anovos.shared.spark import spark


def stats_args(anovos_configs, func):
	stats_configs = anovos_configs.get("stats_generator", None)
	write_configs = anovos_configs.get("write_stats", None)
	report_input_path = ""
	report_configs = anovos_configs.get("report_preprocessing", None)
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


report_logger = []


def run_functions(eval_configs, anovos_configs, df, functions_list, report_logger):
	for function in functions_list:
		for key, args in anovos_configs.items():

			if (function == "concatenate_dataset") & (args is not None):
				start = timeit.default_timer()
				idfs = [df]
				for k in [e for e in args.functions() if e not in ("method")]:
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
				logger.info(f"{function}: execution time (in secs) = {round(end - start, 4)}")
				report_logger.append(round(end - start, 4))
				continue

			if (function == "join_dataset") & (args is not None):
				start = timeit.default_timer()
				idfs = [df]
				for k in [e for e in args.functions() if e not in ("join_type", "join_cols")]:
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
				logger.info(f"{function}: execution time (in secs) = {round(end - start, 4)}")
				report_logger.append(round(end - start, 4))

				continue

			if (function == "timeseries_analyzer") & (args is not None):

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
					)
					end = timeit.default_timer()
					logger.info(
						f"{function}, auto_detection: execution time (in secs) ={round(end - start, 4)}")
					report_logger.append(round(end - start, 4))

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
					)
					end = timeit.default_timer()
					logger.info(
						f"{function}, inspection: execution time (in secs) ={round(end - start, 4)}")
					report_logger.append(round(end - start, 4))

				continue

			if (
				(function == "anovos_basic_report")
				& (args is not None)
				& args.get("basic_report", False)
			):
				start = timeit.default_timer()
				anovos_basic_report(
					spark,
					df,
					**args.get("report_args", {}),
					run_type=run_type,
				)
				end = timeit.default_timer()
				logger.info(
					f"Basic Report: execution time (in secs) ={round(end - start, 4)}")
				report_logger.append(round(end - start, 4))

				continue

			if not all_configs.get("anovos_basic_report", {}).get("basic_report", False):
				if (function == "stats_generator") & (args is not None):
					for m in args["metric"]:
						start = timeit.default_timer()
						print("\n" + m + ": \n")
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
							f"{function}, {m}: execution time (in secs) ={round(end - start, 4)}")
						report_logger.append(round(end - start, 4))


				if (function == "quality_checker") & (args is not None):
					for subfunction, value in args.items():
						if value is not None:
							start = timeit.default_timer()
							print("\n" + subfunction + ": \n")
							f = getattr(quality_checker, subfunction)
							extra_args = stats_args(all_configs, subfunction)
							if subfunction == "nullColumns_detection":
								if "invalidEntries_detection" in args.functions():
									if args.get("invalidEntries_detection").get(
										"treatment", None
									):
										extra_args["stats_missing"] = {}
								if "outlier_detection" in args.functions():
									if args.get("outlier_detection").get("treatment", None):
										if (
											args.get("outlier_detection").get(
												"treatment_method", None
											)
											== "null_replacement"
										):
											extra_args["stats_missing"] = {}
							df, df_stats = f(
								spark, df, **value, **extra_args, print_impact=False
							)
							df = save(
								df,
								write_intermediate,
								folder_name="data_analyzer/quality_checker/"
								+ subfunction
								+ "/dataset",
								reread=True,
							)
							if report_input_path:
								save_stats(
									spark,
									df_stats,
									report_input_path,
									subfunction,
									reread=True,
									run_type=run_type,
								).show(100)
							else:
								save(
									df_stats,
									write_stats,
									folder_name="data_analyzer/quality_checker/" + subfunction,
									reread=True,
								).show(100)
							end = timeit.default_timer()
							logger.info(
								f"{function}, {subfunction}: execution time (in secs) ={round(end - start, 4)}")
							report_logger.append(round(end - start, 4))


				if (function == "association_evaluator") & (args is not None):
					for subfunction, value in args.items():
						if value is not None:
							start = timeit.default_timer()
							print("\n" + subfunction + ": \n")
							f = getattr(association_evaluator, subfunction)
							extra_args = stats_args(all_configs, subfunction)
							df_stats = f(
								spark, df, **value, **extra_args, print_impact=False
							)
							if report_input_path:
								save_stats(
									spark,
									df_stats,
									report_input_path,
									subfunction,
									reread=True,
									run_type=run_type,
								).show(100)
							else:
								save(
									df_stats,
									write_stats,
									folder_name="data_analyzer/association_evaluator/"
									+ subfunction,
									reread=True,
								).show(100)
							end = timeit.default_timer()
							logger.info(
								f"{function}, {subfunction}: execution time (in secs) ={round(end - start, 4)}")
							report_logger.append(round(end - start, 4))


				if (function == "drift_detector") & (args is not None):
					for subfunction, value in args.items():

						if (subfunction == "drift_statistics") & (value is not None):
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
								run_type=run_type,
								print_impact=False,
							)
							if report_input_path:
								save_stats(
									spark,
									df_stats,
									report_input_path,
									subfunction,
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
								f"{function}, {subfunction}: execution time (in secs) ={round(end - start, 4)}")
							report_logger.append(round(end - start, 4))


						if (subfunction == "stability_index") & (value is not None):
							start = timeit.default_timer()
							idfs = []
							for k in [e for e in value.functions() if e not in ("configs")]:
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
									subfunction,
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
								f"{function}, {subfunction}: execution time (in secs) ={round(end - start, 4)}")
							report_logger.append(round(end - start, 4))


					logger.info(
						f"execution time w/o report (in sec) ={round(end - start_main, 4)}"
					)

				if (function == "transformers") & (args is not None):
					for subfunction, value in args.items():
						if value is not None:
							for subfunction2, value2 in value.items():
								if value2 is not None:
									start = timeit.default_timer()
									print("\n" + subfunction2 + ": \n")
									f = getattr(transformers, subfunction2)
									extra_args = stats_args(all_configs, subfunction2)
									if subfunction2 in (
										"cat_to_num_supervised",
										"imputation_sklearn",
										"autoencoder_latentFeatures",
										"auto_imputation",
										"PCA_latentFeatures",
									):
										extra_args["run_type"] = run_type
									if subfunction2 in (
										"normalization",
										"feature_transformation",
										"boxcox_transformation",
										"expression_parser",
									):
										df_transformed = f(
											df, **value2, **extra_args, print_impact=True
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
										+ subfunction2,
										reread=True,
									)
									end = timeit.default_timer()
									logger.info(
										f"{function}, {subfunction2}: execution time (in secs) ={round(end - start, 4)}")
									report_logger.append(round(end - start, 4))


				if (function == "report_preprocessing") & (args is not None):
					for subfunction, value in args.items():
						if (subfunction == "charts_to_objects") & (value is not None):
							start = timeit.default_timer()
							f = getattr(report_preprocessing, subfunction)
							extra_args = stats_args(all_configs, subfunction)
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
								f"{function}, {subfunction}: execution time (in secs) ={round(end - start, 4)}")
							report_logger.append(round(end - start, 4))

				if (function == "report_generation") & (args is not None):
					start = timeit.default_timer()
					timeseries_analyzer = all_configs.get("timeseries_analyzer", None)
					if timeseries_analyzer:
						analysis_level = timeseries_analyzer.get("analysis_level", None)
					else:
						analysis_level = None
					anovos_report(**args, run_type=run_type, output_type=analysis_level)
					end = timeit.default_timer()
					logger.info(
						f"{function}, full_report: execution time (in secs) ={round(end - start, 4)}")
					report_logger.append(round(end - start, 4))		

		save(df, write_main, folder_name="final_dataset", reread=False)
			