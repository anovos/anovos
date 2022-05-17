import os
from jinja2 import Template
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from datetime import datetime

ANOVOS_SOURCE = 'anovos_source'

dataframe_to_feast_type_mapping = {
    'string': 'String',
    'int': 'Int64',
    'float': 'Float32',
    'timestamp': 'String'
    # TODO: type conversion
}


def generate_entity_definition(config: dict) -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "entity.txt"
    )

    with open(source_template_path, 'r') as f:
        template_string = f.read()
        entity_template = Template(template_string)
        data = {
            "entity_name": config['entity'],
            "join_keys": config['id_col'],
            "value_type": "STRING",
            "description": config["entity_description"]
        }

        return entity_template.render(data)


def generate_feature_view(types: list[(str, str)], config: dict) -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "feature_view.txt"
    )

    with open(source_template_path, 'r') as f:
        template_string = f.read()

        fields = generate_fields(types)

        feature_view_template = Template(template_string)
        data = {
            'source': ANOVOS_SOURCE,
            'view_name': config['view_name'],
            'entity': config['entity'],
            'fields': fields,
            'ttl_in_seconds': config['view_ttl_in_seconds'],
            'owner': config['view_owner']
        }

        return feature_view_template.render(data)


def generate_fields(types: list[(str, str)]) -> str:
    fields = ""
    for (field_name, field_type) in types:
        fields += generate_field(field_name, dataframe_to_feast_type_mapping[field_type])

    return fields


def generate_field(field_name: str, field_type: str) -> str:
    template_string = """ Field(name="{{name}}", dtype={{type}}),\n"""
    field_template = Template(template_string)

    return field_template.render({'name': field_name, 'type': field_type})


def generate_file_source(config: dict, file_name="Test") -> str:
    source_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "file_source.txt"
    )

    with open(source_template_path, 'r') as f:
        template_string = f.read()

        file_source_template = Template(template_string)
        data = {
            'source_name': ANOVOS_SOURCE,
            'filename': file_name,
            'ts_column': config['timestamp_col'],
            'create_ts_column': config['create_timestamp_col'],
            'source_description': config['source_description'],
            'owner': config['owner']
        }

    return file_source_template.render(data)


def generate_prefix():
    prefix_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "prefix.txt"
    )

    with open(prefix_path, 'r') as f:
        prefix = f.read()
        return prefix


def generate_feature_description(types: list[(str, str)], feast_config: dict, file_name: str):
    print("Building feature definitions for feature_store:")
    prefix = generate_prefix()
    print(prefix)
    file_source_definition = generate_file_source(feast_config, file_name)
    print(file_source_definition)
    entity_definition = generate_entity_definition(feast_config)
    print(entity_definition)
    feature_view = generate_feature_view(types, feast_config)
    print(feature_view)

    import os

    feature_file = os.path.join(feast_config['file_path'], "feature_demo.py")
    with open(feature_file, 'w') as f:
        f.write(prefix + "\n")
        f.write(file_source_definition + "\n")
        f.write(entity_definition + "\n")
        f.write(feature_view + "\n")


def add_timestamp_columns(idf: DataFrame, feast_config: dict):
    print("Adding timestamp columns")
    return idf.withColumn(feast_config['timestamp_col'], lit(datetime.now()))\
        .withColumn(feast_config['create_timestamp_col'], lit(datetime.now()))

