import pytest
import yaml
from pathlib import Path
from typing import Callable
from itertools import tee

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError
from azure.core.paging import ItemPaged
from azure.ai.ml._utils._arm_id_utils import generate_dataset_arm_id
from azure.ai.ml.entities._assets import Dataset
from azure.ai.ml.constants import LONG_URI_REGEX_FORMAT, LONG_URI_FORMAT
from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


class TestDataset(MlRecordedTest):
    @MlPreparer()
    @recorded_by_proxy
    def test_data_upload_file(self, tmp_path: Path, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        f = tmp_path / "data_local.yaml"
        data_path = tmp_path / "sample1.csv"
        data_path.write_text("hello world")
        name = randstr()
        version = 4
        f.write_text(
            f"""
        name: {name}
        version: {version}
        local_path: {data_path}
    """
        )

        # client.datasets.delete(name=name, version=str(version))
        data_asset = Dataset.load(path=f)
        client.datasets.create_or_update(data_asset)
        internal_data = client.datasets.get(name=name, version=version)

        assert internal_data.name == name
        assert internal_data.version == str(version)
        path = internal_data.paths[0].file
        assert path.endswith("sample1.csv")
        assert "workspaceblobstore" in path
        assert internal_data.id == generate_dataset_arm_id(client._operation_scope, name, version)
        assert internal_data.paths[0].filetaset_arm_id(client._operation_scope, name, version)
        assert internal_data.paths[0].file


    @MlPreparer()
    @recorded_by_proxy
    def test_create_directory(
        self, tmp_path: Path, randstr: Callable[[], str], **kwargs
    ) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_path = tmp_path / "data_directory.yaml"
        name = randstr()
        data_path.write_text(
            f"""
        name: {name}
        version: 1
        description: "this is a test dataset"
        paths:
            - folder: "azureml://datastores/workspaceblobstore/paths/python/"
        """
        )

        with open(data_path, "r") as f:
            config = yaml.safe_load(f)

        name = config["name"]
        version = config["version"]

        data_asset = Dataset.load(path=data_path)
        obj = client.datasets.create_or_update(data_asset)
        assert obj is not None
        assert config["name"] == obj.name
        assert obj.id == generate_dataset_arm_id(client._operation_scope, name, version)

        data_version = client.datasets.get(name, version)

        assert data_version.name == obj.name
        assert data_version.id == generate_dataset_arm_id(client._operation_scope, name, version)
        assert data_version.paths[0].folder


    @MlPreparer()
    @recorded_by_proxy
    def test_create_directory_from_local_path(
        self, tmp_path: Path, randstr: Callable[[], str], **kwargs
    ) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_yaml = tmp_path / "data_directory.yaml"
        name = randstr()
        data_yaml.write_text(
            f"""
                name: {name}
                version: 1
                description: "this is a test dataset"
                local_path: {tmp_path}
            """
        )

        with open(data_yaml, "r") as f:
            config = yaml.safe_load(f)
        name = config["name"]
        version = config["version"]

        data_asset = Dataset.load(path=data_yaml)
        obj = client.datasets.create_or_update(data_asset)
        assert obj is not None
        assert config["name"] == obj.name
        assert obj.id == generate_dataset_arm_id(client._operation_scope, name, version)

        data_version = client.datasets.get(name, version)

        assert data_version.name == obj.name
        assert data_version.version == obj.version
        assert data_version.id == generate_dataset_arm_id(client._operation_scope, name, version)
        assert data_version.paths[0].folder


    @MlPreparer()
    @recorded_by_proxy
    def test_create_file_short_datastore_id(
        self, tmp_path: Path, randstr: Callable[[], str], **kwargs
    ) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_yaml = tmp_path / "data_directory.yaml"
        name = randstr()
        data_yaml.write_text(
            f"""
            name: {name}
            version: 2
            description: "this is a test dataset"
            paths:
            - file: "azureml://datastores/workspaceblobstore/paths/python/data.csv"
        """
        )

        with open(data_yaml, "r") as f:
            config = yaml.safe_load(f)
        name = config["name"]
        version = config["version"]

        data_asset = Dataset.load(path=data_yaml)
        obj = client.datasets.create_or_update(data_asset)
        assert obj is not None
        assert config["name"] == obj.name
        assert obj.id == generate_dataset_arm_id(client._operation_scope, name, version)

        data_version = client.datasets.get(name, version)

        assert data_version.name == obj.name
        assert data_version.version == obj.version
        assert data_version.id == generate_dataset_arm_id(client._operation_scope, name, version)
        assert data_version.paths[0].file

    @MlPreparer()
    @recorded_by_proxy
    def test_create_autoincrement(
        self, randstr: Callable[[], str], tmp_path: Path, **kwargs
    ) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_name = randstr()

        data = tmp_path / "sample1.csv"
        data.write_text("hello world")
        data_path = tmp_path / "data_directory.yaml"
        data_path.write_text(
            f"""
        name: {data_name}
        description: "this is a test dataset"
        local_path: {data_path}
    """
        )

        data_asset = Dataset.load(path=data_path)
        assert data_asset.version is None
        assert data_asset._auto_increment_version

        created_data_asset = client.datasets.create_or_update(data_asset)
        assert created_data_asset.version == "1"
        assert created_data_asset._auto_increment_version is False

        next_dataset_asset = client.datasets.create_or_update(data_asset)
        assert next_dataset_asset.version == "2"
        assert next_dataset_asset._auto_increment_version is False


    @MlPreparer()
    @recorded_by_proxy
    def test_list(self, dataset_with_2_versions: str, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_iterator = client.datasets.list(name=dataset_with_2_versions)
        assert isinstance(data_iterator, ItemPaged)
        # creating a copy of iterator
        data_iterator, data_iterator_copy1, data_iterator_copy2 = tee(data_iterator, 3)
        # iterating the whole iterable object to find the number of elements. Not using list.
        assert sum(1 for e in data_iterator) > 0
        assert sum(1 for e in data_iterator_copy1) == 2
        version_list0 = next(iter(data_iterator_copy2), None)
        assert version_list0.name == dataset_with_2_versions
        assert version_list0.version == "2"
        version_list1 = next(iter(data_iterator_copy2), None)
        assert version_list1.name == dataset_with_2_versions
        assert version_list1.version == "1"


    @MlPreparer()
    @recorded_by_proxy
    def test_dataset_get_latest_label(self, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = randstr()
        versions = ["foo", "bar", "baz", "foobar"]

        for version in versions:
            client.datasets.create_or_update(
                Dataset.load(
                    path="./tests/test_configs/dataset/dataset_local_path_with_datastore.yaml",
                    params_override=[{"name": name}, {"version": version}],
                )
            )
            assert client.datasets.get(name, label="latest").version == version
