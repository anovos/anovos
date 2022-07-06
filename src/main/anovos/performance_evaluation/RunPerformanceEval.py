import sys

from anovos.performance_evaluation import PerformanceEvalWrapper

PerformanceEvalWrapper.evaluate(eval_config_path=sys.argv[1], anovos_config_path=sys.argv[2])