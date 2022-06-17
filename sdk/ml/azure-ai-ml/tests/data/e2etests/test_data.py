import pytest
import yaml
from pathlib import Path
from typing import Callable
from itertools import tee
from time import sleep

from azure.ai.ml import MLClient, load_data
from azure.core.paging import ItemPaged
from azure.ai.ml._utils._arm_id_utils import generate_data_arm_id
from azure.ai.ml.entities._assets import Data
from azure.ai.ml._restclient.v2022_05_01.models import ListViewType

from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


class TestData(MlRecordedTest):
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
        path: {data_path}
        type: uri_file
    """
        )

        data_asset = load_data(path=f)
        client.data.create_or_update(data_asset)
        internal_data = client.data.get(name=name, version=str(version))

        assert internal_data.name == name
        assert internal_data.version == str(version)
        assert internal_data.path.endswith("sample1.csv")
        assert internal_data.id == generate_data_arm_id(client._operation_scope, name, version)

    @MlPreparer()
    @recorded_by_proxy
    def test_create_uri_folder(self, tmp_path: Path, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_path = tmp_path / "data_directory.yaml"
        tmp_folder = tmp_path / "tmp_folder"
        tmp_folder.mkdir()
        tmp_file = tmp_folder / "tmp_file.csv"
        tmp_file.write_text("hello world")
        name = randstr()
        data_path.write_text(
            f"""
            name: {name}
            version: 1
            description: "this is a test dataset"
            path: {tmp_folder}
            type: uri_folder
        """
        )

        with open(data_path, "r") as f:
            config = yaml.safe_load(f)
        name = config["name"]
        version = config["version"]

        data_asset = load_data(path=data_path)
        obj = client.data.create_or_update(data_asset)
        assert obj is not None
        assert config["name"] == obj.name
        assert obj.id == generate_data_arm_id(client._operation_scope, name, version)

        data_version = client.data.get(name, version)

        assert data_version.name == obj.name
        assert data_version.id == generate_data_arm_id(client._operation_scope, name, version)
        assert data_version.path.endswith("/tmp_folder/")

    @MlPreparer()
    @recorded_by_proxy
    def test_create_uri_file(
        self, tmp_path: Path, randstr: Callable[[], str], **kwargs
    ) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_yaml = tmp_path / "data_directory.yaml"
        tmp_folder = tmp_path / "tmp_folder"
        tmp_folder.mkdir()
        tmp_file = tmp_folder / "tmp_file.csv"
        tmp_file.write_text("hello world")
        name = randstr()
        data_yaml.write_text(
            f"""
            name: {name}
            version: 2
            description: "this is a test dataset"
            path: {tmp_file}
            type: uri_file
        """
        )

        with open(data_yaml, "r") as f:
            config = yaml.safe_load(f)
        name = config["name"]
        version = config["version"]

        data_asset = load_data(path=data_yaml)
        obj = client.data.create_or_update(data_asset)
        assert obj is not None
        assert config["name"] == obj.name
        assert obj.id == generate_data_arm_id(client._operation_scope, name, version)

        data_version = client.data.get(name, version)

        assert data_version.name == obj.name
        assert data_version.version == obj.version
        assert data_version.id == generate_data_arm_id(client._operation_scope, name, version)
        assert data_version.path.endswith("/tmp_file.csv")

    @MlPreparer()
    @recorded_by_proxy
    def test_create_mltable(self, tmp_path: Path, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_path = tmp_path / "mltable_directory.yaml"
        tmp_folder = tmp_path / "tmp_folder"
        tmp_folder.mkdir()
        tmp_metadata_file = tmp_folder / "MLTable"
        tmp_metadata_file.write_text(
            """
paths:
  - file: ./tmp_file.csv
transformations:
  - read_delimited:
      delimiter: ","
      encoding: ascii
      header: all_files_same_headers
"""
        )
        tmp_file = tmp_folder / "tmp_file.csv"
        tmp_file.write_text(
            """
sepal_length,sepal_width,petal_length,petal_width,species
101,152,123,187,Iris-setosa
4.9,3,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
"""
        )
        name = randstr()
        data_path.write_text(
            f"""
            name: {name}
            version: 1
            description: "this is an mltable dataset"
            path: {tmp_folder}
            type: mltable
        """
        )

        with open(data_path, "r") as f:
            config = yaml.safe_load(f)
        name = config["name"]
        version = config["version"]

        data_asset = load_data(path=data_path)
        obj = client.data.create_or_update(data_asset)
        assert obj is not None
        assert config["name"] == obj.name
        assert obj.id == generate_data_arm_id(client._operation_scope, name, version)

        data_version = client.data.get(name, version)

        assert data_version.name == obj.name
        assert data_version.id == generate_data_arm_id(client._operation_scope, name, version)
        assert data_version.path.endswith("/tmp_folder/")

    @MlPreparer()
    @recorded_by_proxy
    def test_list(self, data_with_2_versions: str, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        data_iterator = client.data.list(name=data_with_2_versions)
        assert isinstance(data_iterator, ItemPaged)

        # iterating the whole iterable object
        data_list = list(data_iterator)

        assert len(data_list) == 2
        assert all(data.name == data_with_2_versions for data in data_list)
        # use a set since ordering of elements returned from list isn't guaranteed
        assert {"1", "2"} == {data.version for data in data_list}

    @MlPreparer()
    @recorded_by_proxy
    def test_data_get_latest_label(self, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = randstr()
        versions = ["foo", "bar", "baz", "foobar"]

        for version in versions:
            client.data.create_or_update(
                load_data(
                    path="./tests/test_configs/dataset/data_file.yaml",
                    params_override=[{"name": name}, {"version": version}],
                )
            )
            sleep(3)
            assert client.data.get(name, label="latest").version == version

    @MlPreparer()
    @recorded_by_proxy
    def test_data_archive_restore_version(self, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = randstr()
        versions = ["1", "2"]
        version_archived = versions[0]
        for version in versions:
            client.data.create_or_update(
                load_data(
                    path="./tests/test_configs/dataset/data_file.yaml",
                    params_override=[{"name": name}, {"version": version}],
                )
            )

        def get_data_list():
            # Wait for list index to update before calling list command
            sleep(30)
            data_list = client.data.list(name=name, list_view_type=ListViewType.ACTIVE_ONLY)
            return [d.version for d in data_list if d is not None]

        assert version_archived in get_data_list()
        client.data.archive(name=name, version=version_archived)
        assert version_archived not in get_data_list()
        client.data.restore(name=name, version=version_archived)
        assert version_archived in get_data_list()

    @pytest.mark.e2etest
    @pytest.mark.skip(reason="Task 1791832: Inefficient, possibly causing testing pipeline to time out.")
    @MlPreparer()
    @recorded_by_proxy
    def test_data_archive_restore_container(self, randstr: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        name = randstr()
        version = "1"
        client.data.create_or_update(
            load_data(
                path="./tests/test_configs/dataset/data_file.yaml",
                params_override=[{"name": name}, {"version": version}],
            )
        )

        def get_data_list():
            # Wait for list index to update before calling list command
            sleep(30)
            data_list = client.data.list(list_view_type=ListViewType.ACTIVE_ONLY)
            return [d.name for d in data_list if d is not None]

        assert name in get_data_list()
        client.data.archive(name=name)
        assert name not in get_data_list()
        client.data.restore(name=name)
        assert name in get_data_list()

