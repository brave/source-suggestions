import logging
import mimetypes

import boto3
from botocore.exceptions import ClientError

import config

boto_session = boto3.Session()
s3_client = boto_session.client("s3")


class InvalidS3Bucket(Exception):
    pass


def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        content_type = mimetypes.guess_type(file_name)[0] or "binary/octet-stream"
        s3_client.upload_file(
            file_name,
            bucket,
            object_name,
            ExtraArgs={
                "GrantRead": f"id={config.BRAVE_TODAY_CLOUDFRONT_CANONICAL_ID}",
                "GrantFullControl": f"id={config.BRAVE_TODAY_CANONICAL_ID}",
                "ContentType": content_type,
            },
        )

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


def clean_source_similarity_file(sources_data, sources_sim_data):
    sources_id = [sources.get("publisher_id") for sources in sources_data]

    for s_id in sources_id:
        if s_id not in sources_sim_data:
            sources_sim_data.pop(s_id, None)
            continue

        if s_id in sources_sim_data:
            for index, suggestion in enumerate(sources_sim_data[s_id]):
                if suggestion["source"] not in sources_id:
                    sources_sim_data[s_id].pop(index)

    return sources_sim_data


def get_source_id_for_title(title, sources_df):
    return sources_df[sources_df.publisher_name == title].publisher_id.to_numpy()[0]
