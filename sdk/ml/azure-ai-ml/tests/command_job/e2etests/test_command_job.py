import functools
from operator import sub
import time
from typing import Any, Callable
from azure.ai.ml._operations.job_ops_helper import _wait_before_polling
from azure.ai.ml._azure_environments import ENDPOINT_URLS, _get_cloud_details, resource_to_scopes
from azure.ai.ml.entities._assets._artifacts.data import Data
from azure.ai.ml.entities._assets._artifacts.dataset import Dataset
from azure.ai.ml.entities._assets.environment import Environment
import jwt
from mock import patch
from azure.ai.ml.constants import TID_FMT, LOCAL_COMPUTE_TARGET
from azure.ai.ml._operations.operation_orchestrator import OperationOrchestrator
from azure.ai.ml.entities._job.command_job import CommandJob
from azure.ai.ml.entities._job.distribution import MpiDistribution
from azure.ai.ml.entities._job.job import Job
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command
import pytest
from azure.ai.ml._restclient.v2022_02_01_preview.models import AmlToken, ListViewType


from azure.ai.ml import MLClient, Input
from azure.ai.ml._operations.run_history_constants import RunHistoryConstants, JobStatus
from azure.ai.ml._utils._arm_id_utils import AMLVersionedArmId
from azure.ai.ml._ml_exceptions import ValidationException

from devtools_testutils import EnvironmentVariableLoader, AzureRecordedTestCase, recorded_by_proxy
from time import sleep
from pathlib import Path

# These params are logged in ..\test_configs\python\simple_train.py. test_command_job_with_params asserts these parameters are
# logged in the training script, so any changes to parameter logging in simple_train.py must preserve this logging or change it both
# here and in the script.
TEST_PARAMS = {"a_param": "1", "another_param": "2"}


MlPreparer = functools.partial(
    EnvironmentVariableLoader,
    "ml",
    ml_subscription_id="00000000-0000-0000-0000-000000000",
    ml_resource_group="00000"
)

print("yahoo")
@patch.object(OperationOrchestrator, "_match", return_value=True)
class TestCommandJob(AzureRecordedTestCase):
    """def __init__(self, method_name: Any) -> None:
        super().__init__(
            method_name,
        )
        self.kwargs.update(
            {
                "jobName": self.create_random_name(prefix="job-", length=16),
            }
        )
        self.kwargs.update(
            {
                "jobName2": self.create_random_name(prefix="job-", length=16),
            }
        )"""

    def create_ml_client(self, subscription_id, resource_group_name):
        credential = self.get_credential(MLClient)
        client = self.get_create_client_from_credential(MLClient,
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name="sdk_vnext_cli"
        )
        return client

    @MlPreparer()
    @recorded_by_proxy
    @pytest.mark.usefixtures("mock_code_hash")
    def test_command_job(self, ml_subscription_id, ml_resource_group) -> None:
        # TODO: need to create a workspace under a e2e-testing-only subscription and resource group
        client = self.create_ml_client(subscription_id=ml_subscription_id, resource_group_name=ml_resource_group)
        job_name = self.create_random_name(prefix="job-", length=16)
        print(f"Creating job {job_name}")

        try:
            _ = client.jobs.get(job_name)

            # shouldn't happen!
            print(f"Found existing job {job_name}")
        except Exception as ex:
            print(f"Job {job_name} not found: {ex}")

        params_override = [{"name": job_name}]
        job = Job.load(
            path="./tests/test_configs/command_job/command_job_test.yml",
            params_override=params_override,
        )
        command_job: CommandJob = client.jobs.create_or_update(job=job)

        assert command_job.name == job_name
        assert command_job.status in RunHistoryConstants.IN_PROGRESS_STATUSES
        assert command_job.environment == "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1"
        assert command_job.compute == "cpu-cluster"
        check_tid_in_url(client, command_job)

        # Test that uri_folder has a trailing slash
        assert job.inputs["hello_input"].type == "uri_folder"
        assert job.inputs["hello_input"].path.endswith("/")

        command_job_2 = client.jobs.get(job_name)
        assert command_job.name == command_job_2.name
        assert command_job.identity.identity_type == command_job_2.identity.identity_type
        assert command_job_2.environment == "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1"
        assert command_job_2.compute == "cpu-cluster"
        check_tid_in_url(client, command_job_2)

def check_tid_in_url(client: MLClient, job: Job) -> None:
    # test that TID is placed in the URL
    cloud_details = _get_cloud_details()
    default_scopes = resource_to_scopes(cloud_details.get(ENDPOINT_URLS.RESOURCE_MANAGER_ENDPOINT))
    token = client._credential.get_token(*default_scopes).token
    decode = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    formatted_tid = TID_FMT.format(decode["tid"])
    if job.services:
        studio_endpoint = job.services.get("Studio", None)
        if studio_endpoint:
            studio_url = studio_endpoint.endpoint
            assert job.studio_url == studio_url
            if studio_url:
                assert formatted_tid in studio_url
