from typing import Callable
from azure.core.paging import ItemPaged

import pytest
from azure.core.polling import LROPoller
from azure.ai.ml import MLClient
from azure.ai.ml.entities._compute.aml_compute import AmlCompute
from azure.ai.ml.entities._compute.compute import Compute

from ml_test import MlRecordedTest, MlPreparer
from devtools_testutils import recorded_by_proxy


@pytest.mark.e2etest
@pytest.mark.mlc
class TestCompute(MlRecordedTest):
    @MlPreparer()
    @recorded_by_proxy
    def test_aml_compute_create_and_delete(self, rand_compute_name: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        compute_name = rand_compute_name()
        params_override = [{"name": compute_name}]
        compute = Compute.load(
            path="./tests/test_configs/compute/compute-aml-no-identity.yaml", params_override=params_override
        )
        compute_resource = client.compute.begin_create_or_update(compute)

        assert compute_resource.name == compute_name
        compute_resource_get: AmlCompute = client.compute.get(name=compute_name)
        assert compute_resource_get.name == compute_name
        assert compute_resource_get.tier == "dedicated"
        outcome = client.compute.begin_delete(name=compute_name, no_wait=True)
        # the compute is getting deleted , but not waiting on the poller! so immediately returning
        # so this is a preferred approach to assert
        assert isinstance(outcome, LROPoller)

    @pytest.mark.skip(reason="not enough capacity")
    @MlPreparer()
    @recorded_by_proxy
    def test_compute_instance_create_and_delete(self, rand_compute_name: Callable[[], str], **kwargs) -> None:
        subscription_id = kwargs.get("ml_subscription_id")
        resource_group = kwargs.get("ml_resource_group")

        client = self.create_ml_client(subscription_id=subscription_id, resource_group_name=resource_group)
        compute_name = rand_compute_name()
        params_override = [{"name": compute_name}]
        compute = Compute.load(
            path="./tests/test_configs/compute/compute-ci.yaml",
            params_override=params_override,
        )
        compute_resource = client.compute.begin_create_or_update(compute=compute)
        assert compute_resource.name == compute_name
        compute_resource_list = client.compute.list()
        assert isinstance(compute_resource_list, ItemPaged)
        compute_resource_get = client.compute.get(name=compute_name)
        assert compute_resource_get.name == compute_name
        outcome = client.compute.begin_delete(name=compute_name, no_wait=True)
        # the compute is getting deleted , but not waiting on the poller! so immediately returning
        # so this is a preferred approach to assert
        assert isinstance(outcome, LROPoller)
