# financial_tool.py

import datetime as dt
from typing import Dict, Union, List
import json
import streamlit as st
import pandas as pd
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands_tools import think, http_request
import boto3
from utils import invoke_bedrock
# --- LLM-Powered Mapping Function ---
def map_table_rows_with_llm(
    df_rows: list, 
    metric_map: Dict
) -> Dict:
    """
    Uses an LLM to dynamically map a list of DataFrame row names to a target metric map.
    
    Args:
        df_rows: A list of row names (index) from the DataFrame.
        metric_map: The target dictionary of metrics to find.
        bedrock_client: The boto3 Bedrock client.
        model_id: The ID of the LLM to use.
        
    Returns:
        A dictionary mapping target metric keys to the best-matched row names.
    """
    system_prompt = f"""
    You are an expert at mapping financial line items. Your task is to match the provided list of financial statement line items with a list of target financial metrics.

    Your output must be a single JSON object. The keys of the JSON object must be the target metric keys, and the values must be the best-matched line item from the provided list. If no good match is found for a target metric, use "N/A" as the value.

    Target Metrics to find:
    {json.dumps(metric_map, indent=2)}

    Input Financial Statement Line Items:
    {json.dumps(df_rows, indent=2)}

    Example Output:
    ```json
    {{
        "net_income": "Net income (loss) attributable to the company",
        "total_revenue": "Total Revenue",
        "total_assets": "Total Assets, excluding intangibles"
    }}
    ```
    """

    raw_result = invoke_bedrock(
            model_id="anthropic.claude-v2",
            prompt=system_prompt,
            max_tokens=500,
        )

    return raw_result

# --- The rest of the `process_financial_tables_to_json` function is updated ---
def process_financial_tables_to_json(
    tables: List[Dict], 
    company_symbol: str
) -> Dict:
    """
    Converts a list of financial statement DataFrames into a standardized JSON structure
    using an LLM for dynamic row name mapping.
    
    Args:
        tables: A list of dictionaries, each with 'df_cleaned' (DataFrame) and 'subcategory'.
        company_symbol: The ticker symbol for the company.
        bedrock_client: The boto3 Bedrock client.
        model_id: The ID of the LLM to use.
        
    Returns:
        A dictionary in the required JSON format.
    """
    financial_data = {
        "symbol": company_symbol,
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow_statement": [],
        "metrics": {},
    }

    statement_map = {
        "IncomeStatement": "income_statement",
        "BalanceSheet": "balance_sheet",
        "CashFlowStatement": "cash_flow_statement",
    }
    
    # Define the target metrics for the LLM to find
    target_metric_map = {
        "IncomeStatement": {
            "net_income": "Net income",
            "total_revenue": "Total revenue",
            "profit_margins": "Profit margins",
            "revenue_growth": "Revenue growth"
        },
        "BalanceSheet": {
            "total_assets": "Total assets",
            "total_liabilities": "Total liabilities",
            "total_equity": "Total equity",
            "long_term_debt": "Long-term debt",
            "book_value_per_share": "Book value per share",
            "shares_outstanding": "Shares outstanding"
        },
        "CashFlowStatement": {
            "dividends_paid": "Dividends paid",
            "net_income": "Net income"
        }
    }

    for table_dict in tables:
        df = table_dict.get('df_cleaned')
        subcategory = table_dict.get('subcategory')

        if not isinstance(df, pd.DataFrame) or not subcategory or subcategory not in statement_map:
            continue

        output_key = statement_map[subcategory]
        
        # Use LLM to get the dynamic mapping for the current table
        llm_mapping = map_table_rows_with_llm(
            df.index.tolist(), 
            target_metric_map[subcategory]
        )

        # Iterate through the columns, assuming they are fiscal dates
        for col in df.columns:
            period_data = {"fiscal_date": str(col)}
            
            # Use the LLM's dynamic mapping to find values
            for json_key, row_name in llm_mapping.items():
                if row_name != "N/A" and row_name in df.index:
                    try:
                        value = df.loc[row_name, col]
                        if pd.notna(value):
                            period_data[json_key] = value
                    except KeyError:
                        continue 

            if len(period_data) > 1: # Check if any data was successfully mapped
                financial_data[output_key].append(period_data)

    return financial_data


@tool
def get_financial_metrics(financial_data: Dict) -> Union[Dict, str]:
    """Calculates and returns key financial metrics from provided financial data."""
    # This function's logic remains the same.
    # It receives the standardized JSON structure from the helper function above.
    try:
        if not financial_data or "symbol" not in financial_data:
            return {"status": "error", "message": "Invalid financial data provided."}

        # Use .get() to safely access nested data
        income_stmt = financial_data.get("income_statement", [{}])[0]
        balance_sheet = financial_data.get("balance_sheet", [{}])[0]
        cash_flow = financial_data.get("cash_flow_statement", [{}])[0]
        metrics = financial_data.get("metrics", {})

        # Calculate metrics from the provided data
        total_debt = balance_sheet.get("long_term_debt", 0) + balance_sheet.get("short_term_debt", 0)
        debt_to_equity = (total_debt / balance_sheet.get("total_equity", 1)) if balance_sheet.get("total_equity") != 0 else "N/A"
        
        net_income = income_stmt.get("net_income")
        return_on_equity = (net_income / balance_sheet.get("total_equity", 1)) if balance_sheet.get("total_equity") != 0 else "N/A"

        metrics_data = {
            "symbol": financial_data["symbol"],
            "market_cap": metrics.get("market_cap", "N/A"),
            "pe_ratio": metrics.get("trailing_pe", "N/A"),
            "forward_pe": metrics.get("forward_pe", "N/A"),
            "peg_ratio": metrics.get("peg_ratio", "N/A"),
            "price_to_book": metrics.get("price_to_book", "N/A"),
            "dividend_yield": metrics.get("dividend_yield", "N/A"),
            "profit_margins": income_stmt.get("profit_margins", "N/A"),
            "revenue_growth": income_stmt.get("revenue_growth", "N/A"),
            "debt_to_equity": debt_to_equity,
            "return_on_equity": return_on_equity,
            "current_ratio": balance_sheet.get("current_ratio", "N/A"),
            "beta": metrics.get("beta", "N/A"),
            "date": dt.datetime.now().strftime("%Y-%m-%d"),
        }

        # Convert percentages
        for key in ["dividend_yield", "profit_margins", "revenue_growth", "return_on_equity"]:
            if isinstance(metrics_data[key], (int, float)) and metrics_data[key] != "N/A":
                metrics_data[key] = round(metrics_data[key] * 100, 2)

        return {"status": "success", "data": metrics_data}

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing financial data: {str(e)}",
        }


def create_initial_messages():
    """Create initial conversation messages for the agent."""
    return [
        {"role": "user", "content": [{"text": "Hello, I need help analyzing company financial metrics."}]},
        {"role": "assistant", "content": [{"text": "I'm ready to help you analyze financial metrics from the 10-K filing."}]},
    ]


@st.cache_resource
def create_financial_metrics_agent():
    """Create and configure the financial metrics analysis agent."""
    agent = Agent(
        system_prompt="""You are a financial analysis specialist. Follow these steps:

<input>
When user provides a JSON file with company financial data:
1. Use get_financial_metrics to retrieve data
2. Analyze key financial metrics
3. Provide comprehensive analysis in the format below
</input>

<output_format>
1. Company Overview:
    - Market Cap
    - Beta
    - Key Ratios

2. Valuation Metrics:
    - P/E Ratio
    - PEG Ratio
    - Price to Book

3. Financial Health:
    - Profit Margins
    - Debt Metrics
    - Growth Indicators

4. Investment Metrics:
    - Dividend Information
    - Return on Equity
    - Risk Assessment
</output_format>""",
        model=BedrockModel(model_id="us.amazon.nova-pro-v1:0", region="us-east-1"),
        tools=[get_financial_metrics, http_request, think],
    )
    agent.messages = create_initial_messages()
    return agent