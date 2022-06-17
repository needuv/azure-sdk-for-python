import re
from pathlib import Path
from typing import Callable
from unittest.mock import patch
import pytest
import uuid
from six import Iterator

from azure.ai.ml import MLClient, load_model
from azure.core.paging import ItemPaged
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities._assets import Model
from azure.ai.ml.constants import LONG_URI_REGEX_FORMAT
from time import sleep

from azure.ai.ml._restclient.v2022_05_01.models import ListViewType

from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


@pytest.fixture
def uuid_name() -> str:
    name = str(uuid.uuid1())
    yield name


@pytest.fixture
def artifact_path(tmpdir_factory) -> str:  # type: ignore
    file_name = tmpdir_factory.mktemp("artifact_testing").join("artifact_file.txt")
    file_name.write("content")
    return str(file_name)


class TestModel(MlRecordedTest):
    @MlPreparer()
    @recorded_by_proxy
    def test_crud_file(self, randstr: Callable[[], str], tmp_path: Path, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        path = Path("./tests/test_configs/model/model_full.yml")
        model_name = randstr()

        model = load_model(path)
        model.name = model_name
        model = client.models.create_or_update(model)
        assert model.name == model_name
        assert model.version == "3"
        assert model.description == "this is my test model"
        assert model.type == "mlflow_model"
        assert re.match(LONG_URI_REGEX_FORMAT, model.path)

        with pytest.raises(Exception):
            with patch("azure.ai.ml._artifacts._artifact_utilities.get_object_hash", return_value="DIFFERENT_HASH"):
                model = load_model(path=artifact_path)
                model = client.models.create_or_update(model)

        model = client.models.get(model.name, "3")
        assert model.name == model_name
        assert model.version == "3"
        assert model.description == "this is my test model"

        models = client.models.list(name=model_name)
        assert isinstance(models, ItemPaged)
        test_model = next(iter(models), None)
        assert isinstance(test_model, Model)

        # client.models.delete(name=model.name, version="3")
        # with pytest.raises(Exception):
        #     client.models.get(name=model.name, version="3")

    @MlPreparer()
    @recorded_by_proxy
    def test_create_autoincrement(self, randstr: Callable[[], str], tmp_path: Path, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        path = Path("./tests/test_configs/model/model_no_version.yml")
        model_name = randstr()

        model = load_model(path)
        model.name = model_name
        assert model.version is None
        assert model._auto_increment_version

        created_model = client.models.create_or_update(model)
        assert created_model.version == "1"
        assert created_model.type == "custom_model"
        assert created_model._auto_increment_version is False

        next_model_asset = client.models.create_or_update(model)
        assert next_model_asset.version == "2"
        assert next_model_asset.type == "custom_model"
        assert next_model_asset._auto_increment_version is False

    @MlPreparer()
    @recorded_by_proxy
    def test_list_no_name(self, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        models = client.models.list()
        assert isinstance(models, Iterator)
        test_model = next(iter(models), None)
        assert isinstance(test_model, Model)

    @MlPreparer()
    @recorded_by_proxy
    def test_models_get_latest_label(self, randstr: Callable[[], str], tmp_path: Path, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = f"model_{randstr()}"
        versions = ["1", "2", "3", "4"]
        model_path = tmp_path / "model.pkl"
        model_path.write_text("hello world")
        for version in versions:
            client.models.create_or_update(Model(name=name, version=version, path=str(model_path)))
            assert client.models.get(name, label="latest").version == version

    @MlPreparer()
    @recorded_by_proxy
    def test_model_archive_restore_version(self, randstr: Callable[[], str], tmp_path: Path, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = f"model_{randstr()}"
        versions = ["1", "2"]
        version_archived = versions[0]
        model_path = tmp_path / "model.pkl"
        model_path.write_text("hello world")
        for version in versions:
            client.models.create_or_update(Model(name=name, version=version, path=str(model_path)))

        def get_model_list():
            # Wait for list index to update before calling list command
            sleep(30)
            model_list = client.models.list(name=name, list_view_type=ListViewType.ACTIVE_ONLY)
            return [m.version for m in model_list if m is not None]

        assert version_archived in get_model_list()
        client.models.archive(name=name, version=version_archived)
        assert version_archived not in get_model_list()
        client.models.restore(name=name, version=version_archived)
        assert version_archived in get_model_list()

    @pytest.mark.skip(reason="Task 1791832: Inefficient, possibly causing testing pipeline to time out.")
    @MlPreparer()
    @recorded_by_proxy
    def test_model_archive_restore_container(
        self, randstr: Callable[[], str], tmp_path: Path, **kwargs
    ) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = f"model_{randstr()}"
        version = "1"
        model_path = tmp_path / "model.pkl"
        model_path.write_text("hello world")
        client.models.create_or_update(Model(name=name, version=version, path=str(model_path)))

        def get_model_list():
            # Wait for list index to update before calling list command
            sleep(30)
            model_list = client.models.list(list_view_type=ListViewType.ACTIVE_ONLY)
            return [m.name for m in model_list if m is not None]

        assert name in get_model_list()
        client.models.archive(name=name)
        assert name not in get_model_list()
        client.models.restore(name=name)
        assert name in get_model_list()
