import sys

from anovos.performance_evaluation import RunEval

RunEval.evaluate(eval_config_path=sys.argv[1], run_type=sys.argv[2], node_count=sys.argv[3], f_name=sys.argv[4])