from typing import Callable
import pytest
import uuid
from unittest.mock import patch
from pathlib import Path

from azure.ai.ml import MLClient
from test_utilities.utils import get_arm_id
from azure.ai.ml.entities._assets import Code
from azure.ai.ml._ml_exceptions import ValidationException
from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy

@pytest.fixture
def code_asset_path(tmp_path: Path) -> str:
    code_path = tmp_path / "code.txt"
    code_path.write_text("hello world")
    return str(code_path)


class TestCode(MlRecordedTest):
    @MlPreparer()
    @recorded_by_proxy
    def test_create_and_get(self, code_asset_path: str, randstr: Callable[[], str], **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        name = randstr()
        code_entity = Code(name=name, version="2", path=code_asset_path)
        assert str(code_entity.path) == str(Path(code_asset_path))

        code_asset_1 = client._code.create_or_update(code_entity)

        code_asset_2 = client._code.get(code_asset_1.name, code_asset_1.version)

        arm_id = get_arm_id(
            ws_scope=client._operation_scope,
            entity_name=code_asset_1.name,
            entity_version=code_asset_1.version,
            entity_type="codes",
        )
        assert code_asset_1.id == code_asset_2.id == arm_id

    @MlPreparer()
    @recorded_by_proxy
    def test_asset_path_update(self, randstr: Callable[[], str], code_asset_path: str, **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        name = randstr()
        code_entity = Code(name=name, version="1", path=code_asset_path)

        _ = client._code.create_or_update(code_entity)

        # create same name and version code asset again with different content hash/asset paths
        with pytest.raises(Exception):
            with patch("azure.ai.ml._artifacts._artifact_utilities.get_object_hash", return_value=uuid.uuid4()):
                code_entity.path = code_asset_path
                client._code.create_or_update(code_entity)

    @MlPreparer()
    @recorded_by_proxy
    def test_create_and_get_from_registry(
        self, code_asset_path: str, randstr: Callable[[], str], **kwargs,
    ) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_registry_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        name = randstr()
        code_entity = Code(name=name, version="2", path=code_asset_path)
        assert str(code_entity.path) == str(Path(code_asset_path))
        code_asset_1 = client._code.create_or_update(code_entity)
        code_asset_2 = client._code.get(code_asset_1.name, code_asset_1.version)
        assert code_asset_1.id == code_asset_2.id
