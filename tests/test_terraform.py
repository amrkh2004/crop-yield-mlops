import os


def test_terraform_files_exist():
    """
    Verifies that Terraform HCL manifests exist in the terraform/ directory.
    """
    tf_dir = "terraform"
    assert os.path.isdir(tf_dir)

    main_tf = os.path.join(tf_dir, "main.tf")
    variables_tf = os.path.join(tf_dir, "variables.tf")
    outputs_tf = os.path.join(tf_dir, "outputs.tf")
    tfvars_example = os.path.join(tf_dir, "terraform.tfvars.example")

    assert os.path.isfile(main_tf)
    assert os.path.isfile(variables_tf)
    assert os.path.isfile(outputs_tf)
    assert os.path.isfile(tfvars_example)


def test_terraform_main_tf_contents():
    """
    Verifies that main.tf defines S3 bucket, versioning, encryption, and public block.
    """
    with open("terraform/main.tf", "r", encoding="utf-8") as f:
        content = f.read()

    assert 'resource "aws_s3_bucket" "dvc_remote"' in content
    assert 'resource "aws_s3_bucket_versioning" "dvc_remote_versioning"' in content
    assert 'resource "aws_s3_bucket_server_side_encryption_configuration"' in content
    assert 'resource "aws_s3_bucket_public_access_block"' in content
    assert 'provider "aws"' in content


def test_terraform_outputs_tf_contents():
    """
    Verifies that outputs.tf exports s3_bucket_name, s3_bucket_arn, and dvc_remote_url.
    """
    with open("terraform/outputs.tf", "r", encoding="utf-8") as f:
        content = f.read()

    assert 'output "s3_bucket_name"' in content
    assert 'output "s3_bucket_arn"' in content
    assert 'output "dvc_remote_url"' in content
