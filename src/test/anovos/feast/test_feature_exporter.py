import pytest


def test_generate_entity_definition():
    config = {
        'entity': 'entity',
        'id_col': 'id column',
        'entity_description': 'test_description'
    }

    from anovos.feast.feature_exporter import generate_entity_definition
    result = generate_entity_definition(config)
    assert 'name="entity"' in result
    assert 'description="test_description"' in result
    assert 'join_keys=["id column"]' in result


# TODO: add edge case tests


def test_generate_feature_view():
    config = {
        'entity': 'test_entity',
        'view_name': 'test_view',
        'view_ttl_in_seconds': 1,
        'view_owner': 'pytest@case'
    }
    from anovos.feast.feature_exporter import generate_feature_view
    result = generate_feature_view(None, config)
    assert 'name="test_view"' in result
    assert 'entities=["test_entity"]' in result
    assert 'ttl=timedelta(seconds=1)' in result
    assert 'owner="pytest@case"'
