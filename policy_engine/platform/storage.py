"""S3/MinIO object storage helper for tenant-scoped file operations."""

import hashlib
import os
import uuid
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

# Support both custom S3_* vars (local MinIO) and Fly Tigris AWS_* vars (production)
S3_ENDPOINT = os.getenv("S3_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL_S3", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("BUCKET_NAME", "pse-artifacts")
S3_REGION = os.getenv("S3_REGION") or os.getenv("AWS_REGION", "us-east-1")


def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


def ensure_bucket():
    """Create the bucket if it doesn't exist."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=S3_BUCKET)
    except Exception:
        client.create_bucket(Bucket=S3_BUCKET)


def upload_file(
    tenant_id: str,
    project_id: str,
    category: str,
    filename: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> tuple[str, str]:
    """Upload a file to S3. Returns (storage_key, checksum)."""
    client = _get_client()
    ext = os.path.splitext(filename)[1]
    unique_name = f"{uuid.uuid4().hex[:12]}{ext}"
    storage_key = f"tenant/{tenant_id}/project/{project_id}/{category}/{unique_name}"
    checksum = hashlib.sha256(data).hexdigest()

    client.put_object(
        Bucket=S3_BUCKET,
        Key=storage_key,
        Body=data,
        ContentType=content_type,
    )
    return storage_key, checksum


def upload_bytes(storage_key: str, data: bytes, content_type: str = "application/octet-stream"):
    """Upload raw bytes to a specific key."""
    client = _get_client()
    client.put_object(Bucket=S3_BUCKET, Key=storage_key, Body=data, ContentType=content_type)


def download_file(storage_key: str) -> bytes:
    """Download a file from S3."""
    client = _get_client()
    response = client.get_object(Bucket=S3_BUCKET, Key=storage_key)
    return response["Body"].read()


def get_presigned_url(storage_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for a stored file."""
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": storage_key},
        ExpiresIn=expires_in,
    )


def delete_file(storage_key: str):
    """Delete a file from S3."""
    client = _get_client()
    client.delete_object(Bucket=S3_BUCKET, Key=storage_key)
