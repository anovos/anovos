import os
from datetime import datetime

from black import FileMode, format_str
from jinja2 import Template
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit

ANOVOS_SOURCE = "anovos_source"

dataframe_to_feast_type_mapping = {
    "string": "String",
    "int": "Int64",
    "float": "Float32",
    "timestamp": "String"
    # TODO: default type
}


def generate_entity_definition(config: dict) -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "entity.txt"
    )

    with open(source_template_path, "r") as f:
        template_string = f.read()
        entity_template = Template(template_string)
        data = {
            "entity_name": config["entity"],
            "join_keys": config["id_col"],
            "value_type": "STRING",
            "description": config["entity_description"],
        }

        return entity_template.render(data)


def generate_feature_view(types: list[(str, str)], config: dict) -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "feature_view.txt"
    )

    with open(source_template_path, "r") as f:
        template_string = f.read()

        # TODO: remove id_columns and dedicated timestamp columns from columns list
        fields = generate_fields(types)

        feature_view_template = Template(template_string)
        data = {
            "feature_view_name": config["view_name"],
            "source": ANOVOS_SOURCE,
            "view_name": config["view_name"],
            "entity": config["entity"],
            "fields": fields,
            "ttl_in_seconds": config["view_ttl_in_seconds"],
            "owner": config["view_owner"],
        }

        return feature_view_template.render(data)


def generate_fields(types: list[(str, str)]) -> str:
    fields = ""
    for (field_name, field_type) in types:
        fields += generate_field(
            field_name, dataframe_to_feast_type_mapping[field_type]
        )

    return fields


def generate_field(field_name: str, field_type: str) -> str:
    template_string = """ Field(name="{{name}}", dtype={{type}}),\n"""
    field_template = Template(template_string)

    return field_template.render({"name": field_name, "type": field_type})


def generate_file_source(config: dict, file_name="Test") -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "file_source.txt"
    )

    with open(source_template_path, "r") as f:
        template_string = f.read()

        file_source_template = Template(template_string)
        data = {
            "source_name": ANOVOS_SOURCE,
            "filename": file_name,
            "ts_column": config["timestamp_col"],
            "create_ts_column": config["create_timestamp_col"],
            "source_description": config["source_description"],
            "owner": config["owner"],
        }

    return file_source_template.render(data)


def generate_prefix():
    prefix_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "prefix.txt"
    )

    with open(prefix_path, "r") as f:
        prefix = f.read()
        return prefix


def generate_feature_description(
    types: list[(str, str)], feast_config: dict, file_name: str
):
    print("Building feature definitions for feature_store")
    prefix = generate_prefix()
    file_source_definition = generate_file_source(feast_config, file_name)
    entity_definition = generate_entity_definition(feast_config)
    feature_view = generate_feature_view(types, feast_config)

    complete_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "complete_file.txt"
    )

    with open(complete_file_path, "r") as f:
        template_string = f.read()

        complete_file_template = Template(template_string)
        data = {
            "prefix": prefix,
            "file_source": file_source_definition,
            "entity": entity_definition,
            "feature_view": feature_view,
        }

        file_content = complete_file_template.render(data)
        # TODO: make black optimize imports
        file_content = format_str(file_content, mode=FileMode())

        feature_file = os.path.join(feast_config["file_path"], "feature_demo.py")
        with open(feature_file, "w") as of:
            of.write(file_content)


def add_timestamp_columns(idf: DataFrame, feast_config: dict):
    print("Adding timestamp columns")
    return idf.withColumn(
        feast_config["timestamp_col"], lit(datetime.now())
    ).withColumn(feast_config["create_timestamp_col"], lit(datetime.now()))
