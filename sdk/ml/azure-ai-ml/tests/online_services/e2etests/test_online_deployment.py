import pytest
import uuid

from azure.ai.ml import MLClient, load_online_deployment, load_online_endpoint
from azure.ai.ml.entities import OnlineEndpoint, OnlineDeployment

from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


class TestOnlineDeployment(MlRecordedTest):
    @MlPreparer()
    @recorded_by_proxy
    def test_online_deployment(self, **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        endpoint_yaml = "tests/test_configs/deployments/online/simple_online_endpoint_mir.yaml"
        deployment_yaml = "tests/test_configs/deployments/online/online_deployment_1.yaml"
        name = "online-ept-" + uuid.uuid4().hex[:15]
        endpoint = load_online_endpoint(endpoint_yaml)
        endpoint.name = name

        deployment = load_online_deployment(deployment_yaml)
        deployment.endpoint_name = name
        deployment.name = "online-dpm-" + uuid.uuid4().hex[:15]

        # create a endpoint
        client.online_endpoints.begin_create_or_update(endpoint)

        try:
            # create a deployment
            client.online_deployments.begin_create_or_update(deployment)
            dep = client.online_deployments.get(name=deployment.name, endpoint_name=endpoint.name)
            assert dep.name == deployment.name

            deps = client.online_deployments.list(endpoint_name=endpoint.name)
            assert len(list(deps)) > 0

            endpoint.traffic = {deployment.name: 100}
            client.online_endpoints.begin_create_or_update(endpoint)
            endpoint_updated = client.online_endpoints.get(endpoint.name)
            assert endpoint_updated.traffic[deployment.name] == 100
            client.online_endpoints.invoke(
                endpoint_name=endpoint.name,
                request_file="tests/test_configs/deployments/model-1/sample-request.json",
            )
        except Exception as ex:
            raise ex
        finally:
            client.online_endpoints.begin_delete(name=endpoint.name, no_wait=True)
