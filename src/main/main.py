import sys
import json

from anovos import workflow

if len(sys.argv) == 4:
    workflow.run(
        config_path=sys.argv[1],
        run_type=sys.argv[2],
        auth_key_val=json.loads(sys.argv[3]),
    )
else:
    workflow.run(config_path=sys.argv[1], run_type=sys.argv[2])
