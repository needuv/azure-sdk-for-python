import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pytest
from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchDeployment, BatchEndpoint, Environment, Job, Model
from azure.ai.ml._operations.job_ops_helper import _wait_before_polling
from azure.ai.ml._operations.run_history_constants import RunHistoryConstants
from azure.ai.ml._utils._arm_id_utils import AMLVersionedArmId
from azure.ai.ml.entities._inputs_outputs import Input
from azure.ai.ml.constants import AssetTypes
from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


@contextmanager
def deployEndpointAndDeployment(client: MLClient, endpoint: BatchEndpoint, deployment: BatchDeployment):
    """Deploys endpoint and deployment, then automatically deletes the endpoint

    :param MLClient client: _description_
    :param BatchEndpoint endpoint: _description_
    :param BatchDeployment deployment: _description_
    :yield _type_: _description_
    """
    client.batch_endpoints.begin_create_or_update(endpoint)
    client.batch_deployments.begin_create_or_update(deployment)

    yield (endpoint, deployment)

    client.batch_endpoints.begin_delete(name=endpoint.name, no_wait=True)


class TestBatchDeployment(MlRecordedTest):

    @pytest.mark.e2etest
    @pytest.mark.skip(reason="TODO (1546262): Test failing constantly, so disabling it")
    @MlPreparer()
    @recorded_by_proxy
    def test_batch_deployment(self, data_with_2_versions: str, **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        endpoint_yaml = "tests/test_configs/endpoints/batch/batch_endpoint_mlflow_new.yaml"
        deployment_yaml = "tests/test_configs/deployments/batch/batch_deployment_mlflow_new.yaml"
        name = "batch-ept-" + uuid.uuid4().hex[:15]
        endpoint = BatchEndpoint.load(endpoint_yaml)
        endpoint.name = name

        deployment = BatchDeployment.load(deployment_yaml)
        deployment.endpoint_name = name
        deployment.name = "batch-dpm-" + uuid.uuid4().hex[:15]

        # create an endpoint
        client.batch_endpoints.begin_create_or_update(endpoint)
        # create a deployment
        client.batch_deployments.begin_create_or_update(deployment)

        dep = client.batch_deployments.get(name=deployment.name, endpoint_name=endpoint.name)
        assert dep.name == deployment.name

        deps = client.batch_deployments.list(endpoint_name=endpoint.name)
        assert len(list(deps)) > 0

        endpoint.traffic = {deployment.name: 100}
        client.batch_endpoints.begin_create_or_update(endpoint)

        # Invoke with output configs
        client.batch_endpoints.invoke(
            endpoint_name=endpoint.name,
            deployment_name=deployment.name,
            input=":".join((data_with_2_versions, "1")),
        )
        client.batch_endpoints.begin_delete(name=endpoint.name, no_wait=True)


    @pytest.mark.e2etest
    @MlPreparer()
    @recorded_by_proxy
    def test_batch_deployment_dependency_label_resolution(self, randstr: Callable[[], str], **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        endpoint_yaml = "tests/test_configs/endpoints/batch/batch_endpoint_mlflow_new.yaml"
        name = "batch-ept-" + uuid.uuid4().hex[:15]
        deployment_yaml = "tests/test_configs/deployments/batch/batch_deployment_mlflow_new.yaml"
        deployment_name = "batch-dpm-" + uuid.uuid4().hex[:15]

        environment_name = randstr()
        environment_versions = ["foo", "bar"]

        for version in environment_versions:
            client.environments.create_or_update(
                Environment.load(
                    "./tests/test_configs/environment/environment_conda_inline.yml",
                    params_override=[{"name": environment_name}, {"version": version}],
                )
            )

        model_name = randstr()
        model_versions = ["1", "2"]

        for version in model_versions:
            client.models.create_or_update(
                Model.load(
                    "./tests/test_configs/model/model_minimal.yml",
                    params_override=[{"name": model_name}, {"version": version}],
                )
            )

        endpoint = BatchEndpoint.load(endpoint_yaml, params_override=[{"name": name}])
        deployment = BatchDeployment.load(
            deployment_yaml,
            params_override=[
                {"endpoint_name": name},
                {"name": deployment_name},
                {"environment": f"azureml:{environment_name}@latest"},
                {"model": f"azureml:{model_name}@latest"},
            ],
        )

        # create an endpoint
        client.batch_endpoints.begin_create_or_update(endpoint)
        # create a deployment
        client.batch_deployments.begin_create_or_update(deployment)
        dep = client.batch_deployments.get(name=deployment.name, endpoint_name=endpoint.name)
        client.batch_endpoints.begin_delete(name=endpoint.name, no_wait=True)

        resolved_environment = AMLVersionedArmId(dep.environment)
        resolved_model = AMLVersionedArmId(dep.model)
        assert dep.name == deployment.name
        assert (
            resolved_environment.asset_name == environment_name
            and resolved_environment.asset_version == environment_versions[-1]
        )
        assert resolved_model.asset_name == model_name and resolved_model.asset_version == model_versions[-1]


    @pytest.mark.e2etest
    @pytest.mark.skip(reason="TODO (1796799): Test causes time out if job does not have a terminal status")
    @MlPreparer()
    @recorded_by_proxy
    def test_batch_job_download(self, tmp_path: Path, **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        def wait_until_done(job: Job) -> None:
            poll_start_time = time.time()
            while job.status not in RunHistoryConstants.TERMINAL_STATUSES:
                time.sleep(_wait_before_polling(time.time() - poll_start_time))
                job = client.jobs.get(job.name)

        endpoint = BatchEndpoint.load(
            "tests/test_configs/endpoints/batch/batch_endpoint_mlflow_new.yaml",
            params_override=[{"name": "batch-ept-" + uuid.uuid4().hex[:15]}],
        )
        deployment = BatchDeployment.load(
            "tests/test_configs/deployments/batch/batch_deployment_quick.yaml",
            params_override=[{"endpoint_name": endpoint.name}, {"name": "batch-dpm-" + uuid.uuid4().hex[:15]}],
        )
        endpoint.traffic = {deployment.name: 100}

        with deployEndpointAndDeployment(client, endpoint, deployment):
            batchjobresource = client.batch_endpoints.invoke(
                endpoint_name=endpoint.name,
                deployment_name=deployment.name,
                input=Input(
                    path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv", type=AssetTypes.URI_FILE
                ),
            )

            batchjob = client.jobs.get(batchjobresource.name)

            wait_until_done(batchjob)

            client.jobs.download(batchjob.name, download_path=tmp_path)

            assert (tmp_path / "predictions.csv").exists(), "Scoring output was not downloaded"
