import sys

from .workflow import run

run(config_path=sys.argv[1], run_type=sys.argv[2])
