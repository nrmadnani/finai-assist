import boto3 
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import streamlit as st


def get_s3_client():
    try:
        s3_client = boto3.client('s3')
        return s3_client
    except NoCredentialsError:
        st.error("Credentials not available.")
        return None
    except PartialCredentialsError:
        st.error("Incomplete credentials provided.")
        return None
    except Exception as e:
        st.error(f"Error creating S3 client: {e}")
        return None
    

def list_s3_files(bucket=""): 
    try: 
        s3_client = get_s3_client()

        response = s3_client.list_objects_v2(Bucket=bucket)
        if 'Contents' in response:
            st.write("Files in S3 bucket:")
            st.write(response['Contents'])
        else: 
            st.write("No files found in the S3 bucket.")
    except NoCredentialsError:
        st.error("Credentials not available.")
    except Exception as e:
        st.error(f"Error retrieving files: {e}")



def upload_file_to_s3(file, bucket, object_name=None):
    if object_name is None:
        object_name = file.name

    try:
        s3_client = get_s3_client()
        s3_client.upload_fileobj(file, bucket, object_name)
        st.success(f"File {object_name} uploaded successfully to bucket {bucket}.")
    except NoCredentialsError:
        st.error("Credentials not available.")
    except Exception as e:
        st.error(f"Error uploading file: {e}")


def check_if_file_already_exists(bucket, object_name):
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket, Key=object_name)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            st.error(f"Error checking file existence: {e}")
            return False
    except Exception as e:
        st.error(f"Error checking file existence: {e}")
        return False