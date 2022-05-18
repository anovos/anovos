import sys

from anovos import workflow

workflow.run(config_path=sys.argv[1], run_type=sys.argv[2])
