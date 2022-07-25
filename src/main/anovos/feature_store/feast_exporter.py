import os
from datetime import datetime

import isort
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


def check_feast_configuration(feast_config: dict, repartition_count: int):
    if repartition_count != 1:
        raise ValueError(
            "Please, set repartition parameter to 1 in write_main block in your config yml!"
        )
    if "file_path" not in feast_config:
        raise ValueError(
            "Please, provide a path to the anovos feature_store repository!"
        )
    if "entity" not in feast_config:
        raise ValueError("Please, provide an entity definition in your config yml!")
    if "file_source" not in feast_config:
        raise ValueError("Please, provide a file source definition in your config yml!")
    if "feature_view" not in feast_config:
        raise ValueError(
            "Please, provide a feature view definition in your config yml!"
        )


def generate_entity_definition(config: dict) -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "entity.txt"
    )

    with open(source_template_path, "r") as f:
        template_string = f.read()
        entity_template = Template(template_string)
        data = {
            "entity_name": config["name"],
            "join_keys": config["id_col"],
            "value_type": "STRING",
            "description": config["description"],
        }

        return entity_template.render(data)


def generate_feature_view(
    types: list, exclude_list: list, config: dict, entity_name: str
) -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "feature_view.txt"
    )

    with open(source_template_path, "r") as f:
        template_string = f.read()

        fields = generate_fields(types, exclude_list)

        feature_view_template = Template(template_string)
        data = {
            "feature_view_name": config["name"],
            "source": ANOVOS_SOURCE,
            "view_name": config["name"],
            "entity": entity_name,
            "fields": fields,
            "ttl_in_seconds": config["ttl_in_seconds"],
            "owner": config["owner"],
        }

        return feature_view_template.render(data)


def generate_fields(types: list, exclude_list: list) -> str:
    fields = ""
    for (field_name, field_type) in types:
        if field_name not in exclude_list:
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
            "source_description": config["description"],
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


def generate_feature_service(service_name: str, view_name: str):
    service_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "feature_service.txt"
    )

    with open(service_template_path, "r") as f:
        template_string = f.read()
        service_template = Template(template_string)
        data = {
            "feature_service_name": service_name,
            "view_name": view_name,
        }

        return service_template.render(data)


def generate_feature_description(types: list, feast_config: dict, file_name: str):
    print("Building feature definitions for feature_store")
    prefix = generate_prefix()

    file_source_config = feast_config["file_source"]
    file_source_definition = generate_file_source(file_source_config, file_name)

    entity_config = feast_config["entity"]
    entity_definition = generate_entity_definition(entity_config)

    feature_view_config = feast_config["feature_view"]
    columns_to_exclude = [
        feast_config["entity"]["id_col"],
        feast_config["file_source"]["timestamp_col"],
        feast_config["file_source"]["create_timestamp_col"],
    ]
    feature_view = generate_feature_view(
        types, columns_to_exclude, feature_view_config, entity_config["name"]
    )

    feature_service = (
        generate_feature_service(
            feast_config["service_name"], feature_view_config["name"]
        )
        if "service_name" in feast_config
        else ""
    )

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
            "feature_service": feature_service,
        }

        file_content = complete_file_template.render(data)
        file_content = format_str(file_content, mode=FileMode())
        file_content = isort.code(file_content)

        feature_file = os.path.join(feast_config["file_path"], "anovos.py")
        with open(feature_file, "w") as of:
            of.write(file_content)


def add_timestamp_columns(idf: DataFrame, feast_file_source__config: dict):
    print("Adding timestamp columns")
    return idf.withColumn(
        feast_file_source__config["timestamp_col"], lit(datetime.now())
    ).withColumn(feast_file_source__config["create_timestamp_col"], lit(datetime.now()))
