import streamlit as st
import pandas as pd
from st_files_connection import FilesConnection 
from s3_services import list_s3_files, upload_file_to_s3, check_if_file_already_exists
import tempfile
import os
from utils import get_accounts, get_indexes, extract_tables_from_s3, parse_textract_tables
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
from financials_metrics_agent import create_financial_metrics_agent, process_financial_tables_to_json
# financial_tool.py

import datetime as dt
from typing import Dict, Union

# Third-party imports
import streamlit as st
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands_tools import think, http_request




st.title("📊 10-K Filing Analyzer")
uploaded_file = st.file_uploader("Upload 10-K PDF", type=["pdf"])

if uploaded_file is not None:

    # upload file to S3 bucket 
    bucket = "valuationdashboard1"
    document = uploaded_file.name
    region_name = 'us-east-1'

    # Check if the file already exists in S3
    if check_if_file_already_exists(bucket=bucket, object_name=uploaded_file.name):
        st.warning(f"The file '{uploaded_file.name}' already exists in the bucket. Please upload a different file.")
    else: 
        # Upload the file to S3
        upload_file_to_s3(uploaded_file, bucket)


    # connecting to knowledge base  
    knowledge_base_id =  "BNVQMKVQFS"
    user_query = "What is the company's main business?"
    model_id = "anthropic.claude-v2"

    client = boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=region_name,
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],   # AWS_ACCESS_KEY_ID,
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

    with st.spinner("Extracting tables..."):
        # # Create a temporary directory
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     # Construct the full path within the temporary directory
        #     file_path = os.path.join(temp_dir, uploaded_file.name)

        #     # Write the uploaded file content to the temporary path
        #     with open(file_path, "wb") as f:
        #         f.write(uploaded_file.getvalue())
        #         f.close()
            file_path = "C:/Users/nrmadnani/Downloads/" + uploaded_file.name

            st.write(f"File saved temporarily at: {file_path}")
            accounts = get_accounts(file_path)
            st.write(accounts)

            result = extract_tables_from_s3(bucket_name=bucket, file_name=uploaded_file.name)
            tables = parse_textract_tables(result)
            relevant_tables = get_indexes(accounts, tables)
            # create a dataframe for each table and display it
            for table in relevant_tables:
                if "df_cleaned" in table:
                    for account in accounts: 
                        if account["page_number"] == table["page_number"]:
                            st.write(f"Account: {account['account_name']}")
                            st.write(f"Page Number: {account['page_number']}")
                            df = pd.DataFrame(table["df_cleaned"])
                            st.dataframe(df)
                            table["account_name"] = account["account_name"]
                            table["subcategory"] = account["subcategory"]


# # --- NEW SECTION: FINANCIAL METRICS ANALYSIS ---
# st.header("📈 Financial Metrics Analyzer")

# company_symbol = uploaded_file.name.split('_')[0] if '_' in uploaded_file.name else "AMAZON"

# # Pass the bedrock_client and model_id to the processing function
# financial_json_data = process_financial_tables_to_json(
#     relevant_tables, 
#     company_symbol
# )

# if financial_json_data['income_statement'] and financial_json_data['balance_sheet']:
#     financial_metrics_agent = create_financial_metrics_agent()
    
#     with st.spinner("Analyzing financials with the agent..."):
#         try:
#             user_message_text = f"Analyze the financial metrics for {company_symbol} based on the provided data."
#             response = financial_metrics_agent(user_message_text, financial_json_data)
            
#             st.subheader("Analysis Results")
#             st.markdown(response)
            
#             st.write("---")
#             st.subheader("Raw Processed Data (for verification)")
#             st.json(financial_json_data)

#         except Exception as e:
#             st.error(f"Error calling financial agent: {str(e)}")
# else:
#     st.warning("Could not extract enough financial statements to perform analysis. Please check the uploaded PDF.")