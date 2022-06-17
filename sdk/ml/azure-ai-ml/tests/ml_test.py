import functools
from azure.ai.ml import MLClient
from devtools_testutils import EnvironmentVariableLoader, AzureRecordedTestCase


MlPreparer = functools.partial(
    EnvironmentVariableLoader,
    "ml",
    ml_subscription_id="00000000-0000-0000-0000-000000000",
    ml_resource_group="00000",
    ml_workspace_name="00000"
    ml_test_storage_account_name="teststorageaccount",
    ml_test_storage_account_primary_key="primaryKey",
    ml_test_storage_account_secondary_key="secondaryKey",
)

def create_random_name():
    import random
    return f"test_{str(random.randint(1, 1000000000000))}"


class MlRecordedTest(AzureRecordedTestCase):
    def create_ml_client(self, subscription_id, resource_group_name) -> MLClient:
        credential = self.get_credential(MLClient)
        client = self.create_client_from_credential(
            MLClient,
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name="sdk_vnext_cli"
        )
        return client

    def create_registry_ml_client(self, subscription_id, resource_group_name) -> MLClient:
        credential = self.get_credential(MLClient)
        client = self.create_client_from_credential(
            MLClient,
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name="sdk_vnext_cli",
            registry_name="testFeed",
        )
        return client