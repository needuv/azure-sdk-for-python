import uuid

import pytest
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchEndpoint
from azure.ai.ml.entities._assets import Model
from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


class TestBatchEndpoint(MlRecordedTest):
    @pytest.mark.usefixtures("data_with_2_versions")
    @pytest.mark.usefixtures("batch_endpoint_model")
    @MlPreparer()
    @recorded_by_proxy
    def test_batch_endpoint_create_and_invoke(
        self, **kwargs
    ) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        endpoint_yaml = "tests/test_configs/endpoints/batch/batch_endpoint.yaml"
        name = "be-e2e-" + uuid.uuid4().hex[:25]
        # Bug in MFE that batch endpoint properties are not preserved, uncomment below after it's fixed in MFE
        # properties = {"property1": "value1", "property2": "value2"}
        endpoint = BatchEndpoint.load(endpoint_yaml)
        endpoint.name = name
        # endpoint.properties = properties
        obj = client.batch_endpoints.begin_create_or_update(endpoint=endpoint, no_wait=False)
        assert obj is not None
        assert name == obj.name
        # assert obj.properties == properties

        get_obj = client.batch_endpoints.get(name=name)
        assert get_obj.name == name

        client.batch_endpoints.begin_delete(name=name)
        try:
            client.batch_endpoints.get(name=name)
        except Exception as e:
            assert type(e) is ResourceNotFoundError
            return

        raise Exception(f"Batch endpoint {name} is supposed to be deleted.")


    @pytest.mark.usefixtures("light_gbm_model")
    @MlPreparer()
    @recorded_by_proxy
    def test_mlflow_batch_endpoint_create_and_update(self, **kwargs) -> None:
        # light_gbm_model fixture is not used directly, but it makes sure the model being used by the batch endpoint exists

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        endpoint_yaml = "tests/test_configs/endpoints/batch/batch_endpoint_mlflow.yaml"
        name = "be-e2e-" + uuid.uuid4().hex[:25]
        endpoint = BatchEndpoint.load(endpoint_yaml)
        endpoint.name = name
        obj = client.batch_endpoints.begin_create_or_update(endpoint=endpoint, no_wait=False)
        assert obj is not None
        assert name == obj.name

        get_obj = client.batch_endpoints.get(name=name)
        assert get_obj.name == name

        client.batch_endpoints.begin_delete(name=name)
        try:
            client.batch_endpoints.get(name=name)
        except Exception as e:
            assert type(e) is ResourceNotFoundError
            return

        raise Exception(f"Batch endpoint {name} is supposed to be deleted.")
