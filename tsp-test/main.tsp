import "@typespec/rest";
import "@typespec/versioning";
import "@azure-tools/typespec-azure-core";
import "@azure-tools/typespec-client-generator-core";

using TypeSpec.Http;
using TypeSpec.Rest;
using TypeSpec.Versioning;
using Azure.Core;
using Azure.ClientGenerator.Core;

@doc("Azure OpenAI Deployment Information.")
@usage(Usage.input | Usage.output)
@access(Access.public)
model AzureOpenAIDeployment {
  @doc("The deployment name.")
  name: string;

  @doc("The name of the model to deploy")
  model_name: string;

  @doc("The model version to deploy")
  model_version: string;

  @doc("The name of the connection to deploy to")
  connection_name: string;

  @doc("The ARM resource id of the deployment.")
  @visibility("read")
  id: string;
}
