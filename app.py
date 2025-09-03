import streamlit as st
import pandas as pd
from st_files_connection import FilesConnection 
from s3_services import list_s3_files, upload_file_to_s3, check_if_file_already_exists
import tempfile
import os
from utils import get_accounts, get_indexes, extract_tables_from_s3, parse_textract_tables, extract_text, chunk_text, build_vector_store, generate_pros_cons_from_chunks
import os 
import json 
import re
import fitz
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from default_prompts import SUMMARY_PROMPT
import boto3
st.title("📊 10-K Filing Analyzer")
uploaded_file = st.file_uploader("Upload 10-K PDF", type=["pdf"])

if uploaded_file is not None:

    # upload file to S3 bucket 
    bucket = "valuationdashboard"
    document = uploaded_file.name
    region_name = 'us-east-1'

    # Check if the file already exists in S3
    if check_if_file_already_exists(bucket=bucket, object_name=uploaded_file.name):
        st.warning(f"The file '{uploaded_file.name}' already exists in the bucket. Please upload a different file.")
    else: 
        # Upload the file to S3
        upload_file_to_s3(uploaded_file, bucket)


    # connecting to knowledge base 
    knowledge_base_id = st.secrets["KNOWLEDGE_BASE_ID"] 
    user_query = "What is the company's main business?"
    model_id = "anthropic.claude-v2"

    client = boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=region_name,
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

    response = client.retrieve_and_generate(
        input={
            'text': user_query
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': f'arn:aws:bedrock:{region_name}::foundation-model/anthropic.claude-v2'
            }
        }
    )

    generated_text = response['output']['text']
    st.write(generated_text)


    # with st.spinner("Extracting text..."):
    #     text = extract_text(uploaded_file, max_pages=max_pages)

    # with st.spinner("Extracting tables..."):
    #     # Create a temporary directory
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         # Construct the full path within the temporary directory
    #         file_path = os.path.join(temp_dir, uploaded_file.name)

    #         # Write the uploaded file content to the temporary path
    #         with open(file_path, "wb") as f:
    #             f.write(uploaded_file.getvalue())

    #         # st.write(f"File saved temporarily at: {file_path}")
    #         accounts = get_accounts(file_path)
    #         # st.write(accounts)

    #         result = extract_tables_from_s3(bucket_name=bucket, file_name=uploaded_file.name)
    #         tables = parse_textract_tables(result)
    #         relevant_tables = get_indexes(accounts, tables)
    
    # with st.spinner("Splitting into chunks..."):
    #     chunks = chunk_text(text)

    # with st.spinner("Building vector store..."):
    #     data, embeddings = build_vector_store(chunks, relevant_tables)


    # # Pros & Cons
    # if "pros" not in st.session_state or "cons" not in st.session_state:
    #     with st.spinner("Extracting positives & negatives..."):
    #         st.session_state.pros, st.session_state.cons = generate_pros_cons_from_chunks(chunks)

    # st.subheader("✅ Positives")
    # st.markdown(st.session_state.pros if st.session_state.pros else "[No positives extracted]")

    # st.subheader("❌ Negatives")
    # st.markdown(st.session_state.cons if st.session_state.cons else "[No negatives extracted]")
