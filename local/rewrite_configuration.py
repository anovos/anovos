import os
import pathlib
import sys

import yaml

UNSUPPORTED_FEATURES = ["write_feast_features"]

_NEW_CONFIG_NAME = "config.yaml.tmp"
_DATA_DIR = pathlib.Path("/data")
_OUTPUT_DIR = pathlib.Path("/output")

config_file = sys.argv[1]

with open(config_file, "rt") as f:
    full_config = yaml.load(f, yaml.SafeLoader)


def validate_config(sub_config):
    try:
        for k, v in sub_config.items():
            if k in UNSUPPORTED_FEATURES:
                raise ValueError(f"{k} is not supported in Docker execution mode.")
            if isinstance(v, dict):
                validate_config(v)
    except AttributeError:
        print(sub_config)


validate_config(full_config)

input_data_paths = []


def add_path(path: str):
    if path.startswith("dbfs:") or path.startswith("s3:"):
        raise ValueError(f"Only local paths are supported: {path}")
    input_data_paths.append(path)


def find_input_data_paths(sub_config):
    for k, v in sub_config.items():
        if k == "read_dataset":
            file_path = v["file_path"]
            add_path(file_path)
        elif k in ("metricDict_path", "dataDict_path"):
            add_path(v)
        elif k in ("source_path", "model_path"):
            if v != "NA":
                add_path(v)
        else:
            if isinstance(v, dict):
                find_input_data_paths(v)


find_input_data_paths(full_config)

data_directory = pathlib.Path(
    os.path.commonpath(
        [str(pathlib.Path(path).absolute()) for path in input_data_paths]
    )
).absolute()


def rewrite_path(path: str, old_parent: pathlib.Path, new_parent: pathlib.Path) -> str:
    old_path = pathlib.Path(path).absolute()
    relative_path = old_path.relative_to(old_parent)
    new_path = new_parent / relative_path
    return str(new_path)


def rewrite_input_data_paths(sub_config):
    for k, v in sub_config.items():
        if k == "read_dataset":
            v["file_path"] = rewrite_path(v["file_path"], data_directory, _DATA_DIR)
        elif k in ("metricDict_path", "dataDict_path"):
            sub_config[k] = rewrite_path(v, data_directory, _DATA_DIR)
        elif k in ("source_path", "model_path"):
            if v != "NA":
                sub_config[k] = rewrite_path(v, data_directory, _DATA_DIR)
        else:
            if isinstance(v, dict):
                rewrite_input_data_paths(v)


rewrite_input_data_paths(full_config)


def rewrite_output_path(sub_config, path_key):
    try:
        old_path = sub_config[path_key]
    except KeyError:
        pass
    else:
        sub_config[path_key] = str(_OUTPUT_DIR / pathlib.Path(old_path))


def rewrite_paths_in_block(sub_config, block_key, path_keys):
    for path_key in path_keys:
        try:
            rewrite_output_path(sub_config[block_key], path_key)
        except KeyError:
            pass


for write_key in ["write_intermediate", "write_main", "write_stats"]:
    rewrite_paths_in_block(full_config, write_key, ["file_path"])

if "anovos_basic_report" in full_config:
    rewrite_paths_in_block(
        full_config["anovos_basic_report"], "report_args", ["output_path"]
    )
rewrite_paths_in_block(full_config, "report_preprocessing", ["master_path"])
rewrite_paths_in_block(
    full_config, "report_generation", ["master_path", "final_report_path"]
)


class OrderPreservingDumper(yaml.Dumper):
    """cf. https://stackoverflow.com/a/52621703"""

    def represent_dict_preserve_order(self, data):
        return self.represent_dict(data.items())


OrderPreservingDumper.add_representer(
    dict, OrderPreservingDumper.represent_dict_preserve_order
)

with open(_NEW_CONFIG_NAME, "wt") as f:
    yaml.dump(full_config, f, Dumper=OrderPreservingDumper)

with open("data_directory.tmp", "wt") as f:
    f.write(str(data_directory.absolute()))


def diff_config(a, b, tree=None):
    tree = tree or []
    for k, v in a.items():
        if isinstance(v, dict):
            diff_config(a[k], b[k], tree + [k])
        else:
            if v != b[k]:
                print(f"{'.'.join(tree + [k])}: {v} -> {b[k]}")


with open(config_file, "rt") as f:
    old_config = yaml.load(f, yaml.SafeLoader)

with open(_NEW_CONFIG_NAME, "rt") as f:
    new_config = yaml.load(f, yaml.SafeLoader)

print("Adapted configuration for execution inside an anovos-worker container:")
diff_config(old_config, new_config)
