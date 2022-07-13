import sys

from anovos.performance_evaluation import RunEval

RunEval.evaluate(eval_config_path=sys.argv[1], anovos_config_path=sys.argv[2], run_type=sys.argv[3])