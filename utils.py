from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import pymupdf
from pdf2image import convert_from_path
import fitz
import boto3
import json
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from default_prompts import DEFAULT_ACCOUNT_DETECTION_SYSTEM_PROMPT
from pydantic_models import Accounts
import time 
import pandas as pd
import json
import re
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer

import numpy as np


def get_context(doc: pymupdf.Document) -> str:
    all_page_context = ""
    for page in doc:
        page_no = int(page.number) + 1
        for block in page.get_text("blocks"):
            if block[4].replace("\n", "").strip() == "":
                continue
            all_page_context += (
                block[4].strip().replace("\n", "")
            )  # block[4] contains the text content
            all_page_context += f"[Page {page_no}]"
            all_page_context += "\n"

    return all_page_context


def is_text_based_pdf(doc, non_blank_pages):
    i = 0
    for page in doc:
        one_based_pg_no = int(page.number) + 1
        if one_based_pg_no in non_blank_pages:  # check if page is not blank
            if len(page.get_text()) > 0:  # for scanned page it will be 0
                i += 1
            else:
                print(f"Page {one_based_pg_no} is blank or has no text.")

    print(
        f"Total pages with text: {i} out of {len(non_blank_pages)} = {(i/len(non_blank_pages))*100:.2f}%"
    )
    return (i / len(non_blank_pages)) * 100 >= float(95)


def is_file_text_valid(resolved_path) -> bool:
    doc = fitz.open(resolved_path)
    non_blank_pages = []
    for index, image in enumerate(
        convert_from_path(pdf_path=resolved_path, dpi=100, use_pdftocairo=True), start=1
    ):
        histogram = image.convert("1").histogram()
        # Check if there are no black pixels.
        if histogram[0] == 0:
            print("Page", index, "appears to be empty.")
        else:
            non_blank_pages.append(index)

    return is_text_based_pdf(doc, non_blank_pages)



def invoke_bedrock(model_id: str, prompt: str, **kwargs
) -> BaseModel:
    """
    Invoke an AWS Bedrock LLM (Claude, Titan, LLaMA, etc.) and parse the response into a Pydantic model.

    Args:
        model_cls (BaseModel): Pydantic class for structured output
        model_id (str): Bedrock modelId (e.g., "anthropic.claude-v2", "amazon.titan-text-express-v1")
        prompt (str): Prompt text
        kwargs: Extra params (e.g., max_tokens, temperature)

    Returns:
        Pydantic model instance
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    # --- Model-specific request building ---
    body = None
    if model_id.startswith("anthropic.claude"):
        # Claude style
        body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", 0.2),
            "stop_sequences": kwargs.get("stop_sequences", []),
        }
    elif model_id.startswith("amazon.titan"):
        # Titan style
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get("max_tokens", 200),
                "temperature": kwargs.get("temperature", 0.2),
                "stopSequences": kwargs.get("stop_sequences", []),
                "topP": kwargs.get("top_p", 0.9),
            },
        }
    elif model_id.startswith("meta.llama"):
        # LLaMA (similar to Titan)
        body = {
            "prompt": prompt,
            "max_gen_len": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
        }
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    # --- Invoke Bedrock ---
    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())

    # --- Normalize response text ---
    raw_text = None
    if model_id.startswith("anthropic.claude"):
        raw_text = result.get("completion", "")
    elif model_id.startswith("amazon.titan"):
        raw_text = result.get("results", [{}])[0].get("outputText", "")
    elif model_id.startswith("meta.llama"):
        raw_text = result.get("generation", "")

    if not raw_text:
        raise RuntimeError(f"Empty response from {model_id}: {result}")

    return raw_text




def invoke_knowledge_base(knowledge_base_id: str, prompt: str, **kwargs) -> str:
    pass
def get_accounts(
    resolved_path, report_type=["Balance Sheet", "Income Statement", "Cash Flow"]
) -> list[dict]:
    accounts = []
    final_accounts = []
    doc = pymupdf.open(resolved_path)

    extension = Path(resolved_path).suffix[1:].lower()

    if extension != "pdf":
        raise ValueError(f"Unsupported file type {extension}")

    if is_file_text_valid(resolved_path):
        context = get_context(doc)
        account_parser = PydanticOutputParser(pydantic_object=Accounts)
        final_prompt = PromptTemplate(
            template=DEFAULT_ACCOUNT_DETECTION_SYSTEM_PROMPT,
            input_variables=["input"],
            partial_variables={
                "format_instructions": account_parser.get_format_instructions()
            },
        )
        report_type = (
            ",".join(report_type) if isinstance(report_type, list) else report_type
        )
        subcategories = report_type.split(",")
        subcategories = ",".join([s.replace(" ", "") for s in subcategories])
        prompt = final_prompt.format(
            input=context, report_type=report_type, subcategories=subcategories
        )
        raw_result = invoke_bedrock(
            model_id="anthropic.claude-v2",
            prompt=prompt,
            max_tokens=500,
        )
        accounts_result = account_parser.parse(raw_result)

        if isinstance(accounts_result, Accounts):
            accounts.extend(
                [

                   {
                            "account_name": account.account_name,
                            "page_number": account.page_number,
                            "subcategory": account.subcategory,
                            "is_continuous": account.is_continuous,
                }
                    for account in accounts_result.accounts
                    if account.subcategory != "OtherStatement"
                ]
            )

        # check to ensure account names are unique
        name_count = {}
        for account in accounts:
            name = account["account_name"]
            if name in name_count:
                name_count[name] += 1
                new_name = f"{name} {name_count[name]}"
                account["account_name"] = new_name
            else:
                name_count[name] = 0  # First occurrence, no change
        print(accounts)

        # removing profit and loss statement if incorrectly mapped to any valid Subcategory
        for account in accounts:
            name = account["account_name"]
            if "profit" and "loss" in name.lower():
                print(account)
            else:
                final_accounts.append(account)
    return final_accounts




def extract_tables_from_s3(bucket_name: str, file_name: str) -> dict:
    """
    Extracts tables from a document in S3 using AWS Textract.

    Args:
        bucket_name (str): Name of the S3 bucket
        file_name (str): Key (path) of the file in the bucket

    Returns:
        dict: JSON response containing Textract's full analysis of tables
    """
    textract = boto3.client("textract")

    # 1. Start the Textract job
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": file_name}},
        FeatureTypes=["TABLES"],
    )
    job_id = response["JobId"]

    # 2. Wait for job completion (polling)
    while True:
        status = textract.get_document_analysis(JobId=job_id)
        job_status = status["JobStatus"]

        if job_status in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(2)  # wait before polling again

    if job_status == "FAILED":
        raise RuntimeError(f"Textract job {job_id} failed.")

    # 3. Collect all pages (Textract paginates results)
    pages = [status]
    next_token = status.get("NextToken")

    while next_token:
        status = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        pages.append(status)
        next_token = status.get("NextToken")

    # Merge all responses into one JSON (you can return pages list instead if preferred)
    return {
        "DocumentMetadata": pages[0]["DocumentMetadata"],
        "Blocks": sum([p["Blocks"] for p in pages], []),
    }


def parse_textract_tables(textract_json: dict) -> list[dict]:
    """
    Parses TABLE blocks from Textract JSON into a list of dicts.
    Each dict contains:
      - df: Pandas DataFrame of the table
      - footnotes: list of strings (if TABLE_FOOTER rows detected)
      - table_title: string if detected (TABLE_TITLE block), else None
    """
    blocks = textract_json["Blocks"]
    block_map = {block["Id"]: block for block in blocks}

    tables = []
    last_table_title = None  # buffer to store TABLE_TITLE before TABLE

    for block in blocks:
        # Capture table titles
        if block["BlockType"] == "TABLE_TITLE":
            last_table_title = block.get("Text", "").strip()
            continue

        if block["BlockType"] == "TABLE":
            rows = {}
            footnotes = []
            table_title = last_table_title
            last_table_title = None  # reset after attaching

            for rel in block.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for child_id in rel["Ids"]:
                        cell = block_map[child_id]
                        if cell["BlockType"] == "CELL":
                            row_index = cell["RowIndex"]
                            col_index = cell["ColumnIndex"]

                            # Extract words inside the cell
                            words = []
                            for c_rel in cell.get("Relationships", []):
                                if c_rel["Type"] == "CHILD":
                                    for wid in c_rel["Ids"]:
                                        word = block_map[wid]
                                        if word["BlockType"] == "WORD":
                                            words.append(word["Text"])
                                        elif word["BlockType"] == "SELECTION_ELEMENT":
                                            if word["SelectionStatus"] == "SELECTED":
                                                words.append("X")

                            cell_text = " ".join(words).strip()

                            rows.setdefault(row_index, {})[col_index] = {
                                "text": cell_text,
                                "entity": cell.get("EntityTypes", []),
                            }

            if rows:
                # Order rows
                sorted_rows = []
                for row_idx in sorted(rows.keys()):
                    row = rows[row_idx]
                    max_cols = max(len(r) for r in rows.values())
                    sorted_rows.append(
                        [
                            row.get(c, {"text": "", "entity": []})
                            for c in range(1, max_cols + 1)
                        ]
                    )

                # Split header/body/footer
                header = None
                body_rows = []
                for row in sorted_rows:
                    if any("COLUMN_HEADER" in cell["entity"] for cell in row):
                        header = [cell["text"] for cell in row]
                    elif any("TABLE_FOOTER" in cell["entity"] for cell in row):
                        footnotes.append(
                            " ".join([cell["text"] for cell in row if cell["text"]])
                        )
                    else:
                        body_rows.append([cell["text"] for cell in row])

                # Build dataframe
                if header:
                    df = pd.DataFrame(body_rows, columns=header)
                else:
                    df = pd.DataFrame(body_rows)

                tables.append(
                    {"df": df, "footnotes": footnotes, "table_title": table_title, "page_number": block.get("Page")}
                )

    return tables




def balance_parentheses(value):
    if isinstance(value, str):
        if value.endswith(")") and not (value.startswith("–(") or value.startswith("(")):
            value = f"({value}"
        if value.startswith("(") and not value.endswith(")"):
            value = f"{value})"
    return value

def replace_parentheses(value):
    if isinstance(value, str):
        if value.endswith(")") and value.startswith("("):
            value = value.replace("(", "-").replace(")", "")
            # value = f"-{value}"
    return value

def correct_negative_dollar(value):
    if isinstance(value, str):
        value = value.strip()
        if re.match(r"^\$ *-", value):
            value = re.sub(r"^\$ *-", "-$", value)

        # Match patterns like -$ -100, -$-100, -$  -100
        elif re.match(r"^-\$ *-", value):
            value = re.sub(r"^-\$ *-", "-$", value)

        # Match patterns like -$100 or -$ 100
        elif re.match(r"^-\$ *", value):
            value = re.sub(r"^-\$ *", "-$", value)

    return value

def custom_concat(val1, val2):
    if pd.isna(val1) and pd.isna(val2):
        return np.nan
    if pd.isna(val1):
        return val2
    if pd.isna(val2):
        return val1
    return str(val1) + str(val2)

def remove_last_dollar(value):
    if isinstance(value, str):  # Ensure the value is a string
        return re.sub(r"(\d+)\$$", r"\1", value)
    return value  # If the value is not a string, return it unchanged


def find_last_valid_row(df):
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        first_col_valid = (pd.notnull(row.iloc[0])) and (row.iloc[0] != "")
        other_cols_empty = (row.iloc[1:].isnull() | (row.iloc[1:] == "")).all()

        if first_col_valid and other_cols_empty:
            #print(row)
            continue
        return idx

    return None



def get_indexes(accounts, table_elements) -> list[dict]:
    array = ["$", np.nan, "s", "S"]

    relevant_page_numbers = []
    relevant_tables = []
    for account in accounts:
        relevant_page_numbers.append(account["page_number"])

    for table in table_elements:
        if int(table["page_number"]) in relevant_page_numbers:
            df = table["df"]
            df = df.applymap(lambda x: re.sub(r"\t", " ", x) if isinstance(x, str) else x)
            df = df.applymap(lambda x: re.sub(r"(\d+)\s*\$$", r"\1", x) if isinstance(x, str) else x)
            df = df.applymap(
                lambda x: re.sub(r"(\b-cid:\d+|\(cid:\d+\)|cid:\d+)", "", x) if isinstance(x, str) else x
            )
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            
            df.replace(to_replace=r"^$|^-+$|^—+$|^–+$", value=np.nan, regex=True, inplace=True)
            df.replace(to_replace=r"[\u2013\u2014]", value="", regex=True, inplace=True)
            df.replace(to_replace=r"^S$|^s$", value=r"$", regex=True, inplace=True)

            df.iloc[:, 1:] = df.iloc[:, 1:].map(balance_parentheses).map(replace_parentheses)
            df_new = df.copy()    
                # df_new = df_new.map(self.remove_last_dollar)
            df_new.columns = [f"{i}|{col}" for i, col in enumerate(df_new.columns)]
            df_new.columns = [col.replace("None", "") for col in df_new.columns]

            for i, col in enumerate(df_new.columns[:-1]):
                col1, col2 = df_new.columns[i], df_new.columns[i + 1]
                if (col1 and col1 != f"{i}|") and (col2 == f"{i+1}|"):
                    
                    if bool(df_new[col1].isna().all()):
                        if df_new[col2].notna().sum().sum() > 0:
                            df_new[col1] = df_new[col1].fillna(df_new[col2])
                            df_new.rename(columns={col2: "deleted"}, inplace=True)
                        # continue
            if "deleted" in list(df_new.columns):
                df_new.drop(columns=["deleted"], inplace=True)

            df_new.replace(to_replace=r",", value=r"", regex=True, inplace=True)
            
            # df_new.columns = [("column "+str(j)) if df_new.columns[j] == f'{j}_' else df_new.columns[j] for j in range(0,len(df_new.columns))]
            result = df_new.apply(lambda col: col.isin(array) | col.isna()).all()
            df_copy = df_new.copy()
            most_recent_true_col = None

            for col in df_new.columns:
                if result[col].all():
                    most_recent_true_col = col
                elif most_recent_true_col:
                    df_copy[col] = df_copy.apply(
                        lambda row: custom_concat(row[most_recent_true_col], row[col]), axis=1
                    )

            df_copy = df_copy.loc[:, ~result]
            df_copy = df_copy.map(correct_negative_dollar)

            idx = find_last_valid_row(df_copy)
            #print(idx)
            df_final = df_copy[: idx + 1]

            if df_final.index[-1] != df_copy.index[-1]:
                print("Add data to text elements as footnotes")
                table["footnotes"] = df_copy[idx + 1 :].to_string()
            else:
                pass
                #print("Both are same")

            input_data = json.loads(df_final.to_json(orient="records"))
            output_data = []

            # Pre-check condition
            valid_to_process = True
            for item in input_data:
                keys = list(item.keys())
                if len(keys) == 3:
                    second_key, third_key = keys[1], keys[2]

                    second_value = item.get(second_key, "")
                    # if second_value == np.nan:
                    #     second_value = ""
                    third_value = item.get(third_key, "")
                    # if third_value == np.nan:
                    #     third_value = ""

                    if second_value and third_value:  # Both have valid values
                        # print(second_value)
                        # print(third_value)
                        # print("Idhar fata")
                        valid_to_process = False
                        break  # No need to check further
                else:
                    valid_to_process = False

            # Process data only if the condition is met
            if valid_to_process:
                for item in input_data:
                    keys = list(item.keys())
                    if len(keys) == 3:
                        first_key = keys[0]
                        second_key, third_key = keys[1], keys[2]

                        merged_value = item.get(second_key, "") or item.get(third_key, "")
                        output_data.append({first_key: item[first_key], second_key: merged_value})

            else:
                pass
                # print("Condition failed: Both second and third keys have valid values for at least one item.")

            if output_data:
                df_op = pd.DataFrame(output_data)
                df_op = df_op.applymap(
                    lambda x: re.sub(r"(-?\$)[ \t]+(?=\d)", r"\1", x) if isinstance(x, str) else x
                )
                table["df_cleaned"] = df_op
            else:
                df_final = df_final.applymap(
                    lambda x: re.sub(r"(-?\$)[ \t]+(?=\d)", r"\1", x) if isinstance(x, str) else x
                )
                table["df_cleaned"] = df_final
            
            relevant_tables.append(table)

            print("Table cleaner component completed successfully")
    return relevant_tables



def extract_text(uploaded_file, max_pages=10):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        text += page.get_text("text") + "\n"
    return text.strip()

# # ---------------------- Load Models ----------------------
# @st.cache_resource
# def load_summarizer():
#     return pipeline(
#         "text2text-generation",
#         model="MBZUAI/LaMini-Flan-T5-248M",
#         tokenizer="MBZUAI/LaMini-Flan-T5-248M"
#     )

# @st.cache_resource
# def load_embedder():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# summarizer = load_summarizer()
# embedder = load_embedder()

# ---------------------- Chunking ----------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# ---------------------- Pros & Cons ----------------------
def generate_pros_cons_from_chunks(chunks):
    pros, cons = [], []

    for chunk in chunks:
        prompt = (
            "Classify the following company disclosure text into strengths/opportunities "
            "and risks/weaknesses. Provide concise bullet points. "
            "Do not repeat statements. If nothing relevant, return 'None'.\n\n"
            f"{chunk}"
        )
        result = summarizer(prompt, max_length=300, min_length=100, do_sample=False)
        output = result[0].get("summary_text") or result[0].get("generated_text", "")

        if "opportunity" in output.lower() or "strength" in output.lower():
            pros.append(output)
        if "risk" in output.lower() or "weakness" in output.lower():
            cons.append(output)

    return "\n".join(pros), "\n".join(cons)



# ---------------------- Embeddings & Search ----------------------
def build_vector_store(chunks, tables):
    data = []

    # text chunks
    for i, ch in enumerate(chunks):
        data.append({"id": f"text_{i}", "content": ch, "type": "text"})

    # tables (flatten rows)
    for ti, table in enumerate(tables):
        for ri, row in enumerate(table):
            row_str = " | ".join(str(c) for c in row if c)
            data.append({"id": f"table_{ti}_{ri}", "content": row_str, "type": "table"})

    texts = [d["content"] for d in data]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return data, embeddings

def semantic_search(query, data, embeddings, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embeddings, q_emb)
    top_idx = sims.argsort()[-top_k:][::-1]
    return [data[i] for i in top_idx]
