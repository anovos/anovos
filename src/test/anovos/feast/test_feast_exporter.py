import os
from copy import deepcopy
from unittest.mock import Mock

import pytest


def _build_config() -> dict:
    entity_config = {
        "name": "test_entity",
        "id_col": "id column",
        "description": "test_description",
    }

    file_source_config = {
        "owner": "test@owner.com",
        "description": "testcase description",
        "timestamp_col": "eventtime",
        "create_timestamp_col": "test_create_column",
    }

    feature_view_config = {
        "name": "test_view",
        "ttl_in_seconds": 1,
        "owner": "pytest@case",
    }

    config = {
        "entity": entity_config,
        "file_source": file_source_config,
        "feature_view": feature_view_config,
        "file_path": f"{os.path.dirname(os.path.abspath(__file__))}",
    }
    return config


# TODO: add edge case tests
def test_generate_entity_definition():
    config = {
        "name": "entity",
        "id_col": "id column",
        "description": "test_description",
    }

    from anovos.feature_store.feast_exporter import generate_entity_definition

    result = generate_entity_definition(config)
    assert 'name="entity"' in result
    assert 'description="test_description"' in result
    assert 'join_keys=["id column"]' in result


def test_generate_feature_view():
    config = {"name": "test_view", "ttl_in_seconds": 1, "owner": "pytest@case"}

    from anovos.feature_store.feast_exporter import generate_feature_view

    result = generate_feature_view(
        types=[("field1", "string")],
        exclude_list=[],
        config=config,
        entity_name="test_entity",
    )
    assert 'name="test_view"' in result
    assert 'entities=["test_entity"]' in result
    assert 'Field(name="field1", dtype=String)' in result
    assert "ttl=timedelta(seconds=1)" in result
    assert 'owner="pytest@case"'


def test_generate_field():
    field_name = "field"
    type_name = "type"

    from anovos.feature_store.feast_exporter import generate_field

    result = generate_field(field_name, type_name)

    assert result.strip() == 'Field(name="field", dtype=type),'


def test_generate_file_source():
    config = {
        "owner": "test@owner.com",
        "description": "testcase description",
        "timestamp_col": "eventtime",
        "create_timestamp_col": "test_create_column",
    }

    from anovos.feature_store.feast_exporter import generate_file_source

    result = generate_file_source(config, "testfile")

    assert 'path="testfile"' in result
    assert 'timestamp_field="eventtime"' in result
    assert 'created_timestamp_column="test_create_column"' in result
    assert 'description="testcase description"' in result
    assert 'owner="test@owner.com"' in result


def test_integration():
    config = _build_config()
    file_path = "/output/result.csv"
    from anovos.feature_store.feast_exporter import generate_feature_description

    generate_feature_description([("field1", "string")], config, file_name=file_path)

    output_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "anovos.py"
    )
    with open(output_file_path, "r") as f:
        result = f.read()
        print(result)
        assert config["entity"]["name"] in result
        assert config["file_source"]["owner"] in result
        assert config["feature_view"]["owner"] in result
        assert file_path in result


def test_generate_feature_service():
    service_name = "income_service"
    view_name = "view_name"

    from anovos.feature_store.feast_exporter import generate_feature_service

    result = generate_feature_service(service_name, view_name)

    assert service_name in result
    assert view_name in result
    assert "FeatureService" in result


def test_check_feast_configuration():
    config = _build_config()
    from anovos.feature_store.feast_exporter import check_feast_configuration

    check_feast_configuration(config, 1)


def test_that_missing_blocks_raise_exception_in_check_feast_configuration():
    config = _build_config()
    faulty_cfg = deepcopy(config)
    del faulty_cfg["file_source"]
    from anovos.feature_store.feast_exporter import check_feast_configuration

    with pytest.raises(Exception) as e:
        check_feast_configuration(faulty_cfg, 1)

    assert e.type == ValueError
    assert (
        e.value.args[0]
        == "Please, provide a file source definition in your config yml!"
    )

    faulty_cfg = deepcopy(config)
    del faulty_cfg["entity"]
    from anovos.feature_store.feast_exporter import check_feast_configuration

    with pytest.raises(Exception) as e:
        check_feast_configuration(faulty_cfg, 1)

    assert e.type == ValueError
    assert e.value.args[0] == "Please, provide an entity definition in your config yml!"

    faulty_cfg = deepcopy(config)
    del faulty_cfg["feature_view"]
    from anovos.feature_store.feast_exporter import check_feast_configuration

    with pytest.raises(Exception) as e:
        check_feast_configuration(faulty_cfg, 1)

    assert e.type == ValueError
    assert (
        e.value.args[0]
        == "Please, provide a feature view definition in your config yml!"
    )

    faulty_cfg = deepcopy(config)
    del faulty_cfg["file_path"]
    from anovos.feature_store.feast_exporter import check_feast_configuration

    with pytest.raises(Exception) as e:
        check_feast_configuration(faulty_cfg, 1)

    assert e.type == ValueError
    assert e.value.args[0] == "Please, provide a path to the anovos feast repository!"


def test_that_faulty_repartition_count_raises_exception_in_check_feast_configuration():
    config = _build_config()
    from anovos.feature_store.feast_exporter import check_feast_configuration

    with pytest.raises(Exception) as e:
        check_feast_configuration(config, 2)

    assert e.type == ValueError
    assert (
        e.value.args[0]
        == "Please, set repartition parameter to 1 in write_main block in your config yml!"
    )


def test_that_happy_path_works_for_add_timestamp_columns(mocker):
    idf_mock = Mock()
    config = _build_config()

    from anovos.feature_store.feast_exporter import add_timestamp_columns

    add_timestamp_columns(idf_mock, config["file_source"])

    idf_mock.withColumn.assert_called()
