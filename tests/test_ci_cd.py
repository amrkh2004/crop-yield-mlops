import os

import yaml


def test_github_actions_workflow_exists():
    """
    Verifies that the GitHub Actions CI/CD workflow YAML file exists.
    """
    workflow_path = os.path.join(".github", "workflows", "ci-cd.yml")
    assert os.path.isfile(workflow_path)


def test_github_actions_workflow_valid_yaml():
    """
    Verifies that .github/workflows/ci-cd.yml contains valid YAML syntax and expected jobs.
    """
    workflow_path = os.path.join(".github", "workflows", "ci-cd.yml")
    with open(workflow_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "name" in data
    assert "jobs" in data
    assert "lint-and-test" in data["jobs"]
    assert "docker-build-push" in data["jobs"]

    lint_job = data["jobs"]["lint-and-test"]
    assert lint_job["runs-on"] == "ubuntu-latest"

    docker_job = data["jobs"]["docker-build-push"]
    assert "needs" in docker_job
    assert docker_job["needs"] == "lint-and-test"
