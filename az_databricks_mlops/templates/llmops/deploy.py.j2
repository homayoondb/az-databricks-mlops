"""Agent deployment via Unity Catalog alias promotion.

Promotes the latest validated agent version to the appropriate alias:
- staging: assigns "challenger" alias (for A/B testing or shadow scoring)
- prod: promotes "challenger" to "champion" alias
"""

from mlflow import MlflowClient


def promote_model(model_name: str, env: str) -> None:
    """Promote the latest agent version based on target environment.

    Args:
        model_name: Full UC model name (catalog.schema.model).
        env: Deployment target (dev, staging, prod).
    """
    client = MlflowClient()
    latest = client.get_model_version_by_alias(model_name, "challenger")

    if env == "prod":
        client.set_registered_model_alias(model_name, "champion", latest.version)
        print(f"Promoted version {latest.version} to 'champion' alias")
    else:
        print(f"Staging: version {latest.version} has 'challenger' alias")
