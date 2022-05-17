import pytest
from unittest.mock import Mock


# TODO: add edge case tests
def test_generate_entity_definition():
    config = {
        'entity': 'entity',
        'id_col': 'id column',
        'entity_description': 'test_description'
    }

    from anovos.feature_store.feast_exporter import generate_entity_definition
    result = generate_entity_definition(config)
    assert 'name="entity"' in result
    assert 'description="test_description"' in result
    assert 'join_keys=["id column"]' in result


def test_generate_feature_view(mocker):
    config = {
        'entity': 'test_entity',
        'view_name': 'test_view',
        'view_ttl_in_seconds': 1,
        'view_owner': 'pytest@case'
    }

    from anovos.feature_store.feast_exporter import generate_feature_view
    result = generate_feature_view([('field1', 'type1')], config)
    assert 'name="test_view"' in result
    assert 'entities=["test_entity"]' in result
    assert 'Field(name="field1", dtype=type1)' in result
    assert 'ttl=timedelta(seconds=1)' in result
    assert 'owner="pytest@case"'


def test_generate_field():
    field_name = "field"
    type_name = "type"

    from anovos.feature_store.feast_exporter import generate_field
    result = generate_field(field_name, type_name)

    assert result.strip() == 'Field(name="field", dtype=type),'


def test_generate_file_source():
    config = {
        'owner': 'test@owner.com',
        'source_description': 'testcase description',
        'timestamp_col': 'eventtime',
        'create_timestamp_col': 'test_create_column'
    }

    from anovos.feature_store.feast_exporter import generate_file_source
    result = generate_file_source(config, 'testfile')

    assert 'path="testfile"' in result
    assert 'timestamp_field="eventtime"' in result
    assert 'created_timestamp_column="test_create_column"' in result
    assert 'description="testcase description"' in result
    assert 'owner="test@owner.com"' in result


def test_integration():
    config = {
        'owner': 'test@owner.com',
        'source_description': 'testcase description',
        'timestamp_col': 'eventtime',
        'create_timestamp_col': 'test_create_column',
        'entity': 'test_entity',
        'view_name': 'test_view',
        'view_ttl_in_seconds': 1,
        'view_owner': 'pytest@case',
        'id_col': 'id column',
        'entity_description': 'test_description',
        'file_path': '/Users/matzep/Workspaces/inlinity/mobilewalla/feast-demo/anovos_repo/'
    }
    from anovos.feature_store.feast_exporter import generate_feature_description
    generate_feature_description([('field1', 'string')], config, file_name="/output/result.csv")