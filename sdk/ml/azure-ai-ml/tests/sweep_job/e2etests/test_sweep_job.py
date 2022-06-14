import time
from multiprocessing.pool import ThreadPool
from typing import Any

import pytest
from azure.ai.ml._operations.job_ops_helper import _wait_before_polling
from azure.ai.ml._operations.operation_orchestrator import OperationOrchestrator
from azure.ai.ml._operations.run_history_constants import JobStatus, RunHistoryConstants
from azure.ai.ml.entities._builders.command_func import command
from azure.ai.ml.entities._inputs_outputs import Input
from azure.ai.ml.entities._job.job import Job
from azure.ai.ml.constants import AssetTypes
from mock import patch
from azure.ai.ml.entities._job.sweep.early_termination_policy import TruncationSelectionPolicy
from azure.ai.ml.entities._job.sweep.search_space import LogUniform

from tempfile import TemporaryDirectory
from pathlib import Path

from ml_test import MlRecordedTest, MlPreparer, create_random_name
from devtools_testutils import recorded_by_proxy


@pytest.mark.usefixtures("mock_code_hash")
class TestSweepJob(MlRecordedTest):

    @MlPreparer()
    @recorded_by_proxy
    def test_sweep_job_submit(self, randstr, **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        job_name = randstr()

        params_override = [{"name": job_name}]
        sweep_job = Job.load(
            path="./tests/test_configs/sweep_job/sweep_job_test.yaml",
            params_override=params_override,
        )
        sweep_job_resource = client.jobs.create_or_update(job=sweep_job)
        assert sweep_job_resource.name == job_name
        assert sweep_job_resource.trial.environment_variables["test_var1"] == "set"
        assert sweep_job_resource.status in RunHistoryConstants.IN_PROGRESS_STATUSES

        sweep_job_resource_2 = client.jobs.get(job_name)
        assert sweep_job_resource.name == sweep_job_resource_2.name

    @MlPreparer()
    @recorded_by_proxy
    def test_sweep_job_submit_with_inputs(self, randstr, **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        job_name = randstr()

        params_override = [{"name": job_name}]
        sweep_job = Job.load(
            path="./tests/test_configs/sweep_job/sweep_job_test_inputs.yaml",
            params_override=params_override,
        )
        sweep_job_resource = client.jobs.create_or_update(job=sweep_job)
        assert sweep_job_resource.name == job_name
        assert sweep_job_resource.status in RunHistoryConstants.IN_PROGRESS_STATUSES

        sweep_job_resource_2 = client.jobs.get(job_name)
        assert sweep_job_resource.name == sweep_job_resource_2.name
        assert sweep_job_resource_2.inputs is not None
        assert len(sweep_job_resource_2.inputs) == 2
        assert "iris_csv" in sweep_job_resource_2.inputs
        assert "some_number" in sweep_job_resource_2.inputs

    @MlPreparer()
    @recorded_by_proxy
    def test_sweep_job_submit_minimal(self, randstr, **kwargs) -> None:
        """Ensure the Minimal required properties does not fail on submisison"""

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        job_name = randstr()

        params_override = [{"name": job_name}]
        sweep_job = Job.load(
            path="./tests/test_configs/sweep_job/sweep_job_minimal_test.yaml",
            params_override=params_override,
        )
        sweep_job_resource = client.jobs.create_or_update(job=sweep_job)
        assert sweep_job_resource.name == job_name
        assert sweep_job_resource.status in RunHistoryConstants.IN_PROGRESS_STATUSES

        sweep_job_resource_2 = client.jobs.get(job_name)
        assert sweep_job_resource.name == sweep_job_resource_2.name

    @MlPreparer()
    @recorded_by_proxy
    def test_sweep_job_await_completion(self, randstr, **kwargs) -> None:
        """Ensure sweep job runs to completion"""

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        job_name = randstr()

        params_override = [{"name": job_name}]
        sweep_job = Job.load(
            path="./tests/test_configs/sweep_job/sweep_job_minimal_test.yaml",
            params_override=params_override,
        )
        sweep_job_resource = client.jobs.create_or_update(job=sweep_job)

        assert sweep_job_resource.name == job_name
        # wait 3 minutes to check job has not failed.
        if self.is_live:
            time.sleep(3 * 60)
        else:
            time.sleep(1)
        assert sweep_job_resource.status in [JobStatus.COMPLETED, JobStatus.RUNNING]

    @MlPreparer()
    @recorded_by_proxy
    def test_sweep_job_download(self, randstr, **kwargs) -> None:

        def wait_until_done(job: Job) -> None:
            poll_start_time = time.time()
            while job.status not in RunHistoryConstants.TERMINAL_STATUSES:
                time.sleep(_wait_before_polling(time.time() - poll_start_time))
                job = client.jobs.get(job.name)
            time.sleep(_wait_before_polling(time.time() - poll_start_time))

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        job = client.jobs.create_or_update(
            Job.load(
                path="./tests/test_configs/sweep_job/sweep_job_minimal_outputs.yaml",
                params_override=[{"name": randstr()}],
            )
        )

        wait_until_done(job)

        with TemporaryDirectory() as tmp_dirname:
            tmp_path = Path(tmp_dirname)
            client.jobs.download(name=job.name, download_path=tmp_path, all=True)

            best_child_run_artifact_dir = tmp_path / "artifacts"
            best_child_run_output_dir = tmp_path / "named-outputs"
            parent_run_artifact_dir = tmp_path / "hd-artifacts"

            assert best_child_run_artifact_dir.exists()
            assert next(best_child_run_artifact_dir.iterdir(), None), "No artifacts for child run were downloaded"
            assert best_child_run_output_dir.exists()
            assert next(best_child_run_output_dir.iterdir(), None), "No outputs for child run were downloaded"
            assert parent_run_artifact_dir.exists()
            assert next(parent_run_artifact_dir.iterdir(), None), "No artifacts for parent run were downloaded"

    @MlPreparer()
    @recorded_by_proxy
    def test_sweep_job_builder(self, randstr, **kwargs) -> None:

        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)

        inputs = {
            "uri": Input(
                type=AssetTypes.URI_FILE, path="azureml://datastores/workspaceblobstore/paths/python/data.csv"
            ),
            "lr": LogUniform(min_value=0.001, max_value=0.1),
        }

        node = command(
            name=randstr(),
            description="description",
            environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",
            inputs=inputs,
            command="echo ${{inputs.uri}} ${{search_space.learning_rate}}",
            display_name="builder_command_job",
            compute="testCompute",
            experiment_name="mfe-test1-dataset",
        )

        sweep_node = node.sweep(
            sampling_algorithm="random",
            goal="maximize",
            primary_metric="accuracy",
            early_termination_policy=TruncationSelectionPolicy(
                evaluation_interval=100, delay_evaluation=200, truncation_percentage=40
            ),
        )

        sweep_node.set_limits(max_concurrent_trials=2, max_total_trials=10, timeout=300)

        assert sweep_node.trial.environment == "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1"
        assert sweep_node.display_name == "builder_command_job"
        assert sweep_node.compute == "testCompute"
        assert sweep_node.experiment_name == "mfe-test1-dataset"

        sweep_node.description = "new-description"
        sweep_node.display_name = "new_builder_command_job"
        assert sweep_node.description == "new-description"
        assert sweep_node.display_name == "new_builder_command_job"

        result = client.create_or_update(sweep_node)
        assert result.description == "new-description"
        assert result.trial.environment == "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1"
        assert result.display_name == "new_builder_command_job"
        assert result.compute == "testCompute"
        assert result.experiment_name == "mfe-test1-dataset"
