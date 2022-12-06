import logging
import mimetypes

import boto3
from botocore.exceptions import ClientError

import config

boto_session = boto3.Session()
s3_client = boto_session.client('s3')


class InvalidS3Bucket(Exception):
    pass


def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        content_type = mimetypes.guess_type(file_name)[0] or 'binary/octet-stream'
        s3_client.upload_file(file_name, bucket, object_name, ExtraArgs={
            'GrantRead': f'id={config.BRAVE_TODAY_CLOUDFRONT_CANONICAL_ID}',
            'GrantFullControl': f'id={config.BRAVE_TODAY_CANONICAL_ID}',
            'ContentType': content_type
        })

    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    try:
        s3_client.download_file(bucket, object_name, file_name)

    except ClientError as e:
        logging.error(e)
        return False
    return True
