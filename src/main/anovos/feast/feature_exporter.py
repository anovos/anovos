from pyspark.pandas import DataFrame
from jinja2 import Template


def generate_entity_definition(config: dict) -> str:
    # TODO: move this into template files
    template_string = """
        Entity(
        name="{{entity_name}}",
        join_keys=["{{join_keys}}"],
        value_type=ValueType.{{value_type}},
        description="{{description}}",
        )
    """
    entity_template = Template(template_string)
    data = {
        "entity_name": config['entity'],
        "join_keys": config['id_col'],
        "value_type": "STRING",
        "description": config["entity_description"]
    }

    return entity_template.render(data)


def generate_feature_view(idf: DataFrame, config: dict) -> str:
    template_string = """
        FeatureView(
            name="{{view_name}}",
            entities=["{{entity}}"],
            ttl=timedelta(seconds={{ttl_in_seconds}}),
            schema=[
                {{fields}}
            ],
            online=True,
            source=anovos_source,
            tags={"production": "True"},
            owner="{{owner}}",
        )
    """

    feature_view_template = Template(template_string)
    data = {
        'view_name': config['view_name'],
        'entity': config['entity'],
        'ttl_in_seconds': config['view_ttl_in_seconds'],
        'owner': config['view_owner']
    }

    return feature_view_template.render(data)


def generate_fields(df: DataFrame) -> str:
    types = df.dtypes


def generate_field(field_name: str, field_type: str) -> str:
    template_string = """ Field(name="{{name}}", dtype={{type}}),"""
    field_template = Template(template_string)

    return field_template.render({'name': field_name, 'type': field_type})


def generate_feature_description(idf: DataFrame, config: dict):
    print("Building feature definitions for feast:")
    entity_definition = generate_entity_definition(config)
    print(entity_definition)
    feature_view = generate_feature_view(idf, config)
    print(feature_view)
