import streamlit as st
import pandas as pd
from st_files_connection import FilesConnection 
from s3_services import list_s3_files, upload_file_to_s3, check_if_file_already_exists
import tempfile
import os
from utils import get_accounts, get_indexes, extract_tables_from_s3, parse_textract_tables

df = pd.DataFrame({
    'first': [1, 2, 3, 4],
    'second': [10, 20, 30, 40],
})
df


uploaded_file = st.file_uploader("Upload a 10Q/10K filing", type=["pdf"])
if uploaded_file is not None:
    bucket = "mystreamlituploadfilebucket"
    roleArn = "arn:aws:iam::423636639269:user/testuser"
    document = uploaded_file.name
    region_name = 'us-east-1'
    # Check if the file already exists in S3
    if check_if_file_already_exists(bucket=bucket, object_name=uploaded_file.name):
        st.warning(f"The file '{uploaded_file.name}' already exists in the bucket. Please upload a different file.")
    else: 
        # Upload the file to S3
        upload_file_to_s3(uploaded_file, bucket="mystreamlituploadfilebucket")


    # send the s3 file to BS, CF, IS service for extraction
    st.write(uploaded_file)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path within the temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Write the uploaded file content to the temporary path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.write(f"File saved temporarily at: {file_path}")
        accounts = get_accounts(file_path)
        st.write(accounts)

        result = extract_tables_from_s3(bucket_name=bucket, file_name=uploaded_file.name)
        tables = parse_textract_tables(result)
        relevant_tables = get_indexes(accounts, tables)

        print(relevant_tables[0])
    # list_s3_files(bucket="mystreamlituploadfilebucket")
    st.write("File uploaded successfully!")