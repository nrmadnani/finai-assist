DEFAULT_ACCOUNT_DETECTION_SYSTEM_PROMPT = """
You are an experienced financial analyst and an expert at reading financial documents.
An account name in a financial statement refers to the title or label given to a specific category or line item that represents financial activity or balances. 
These names are then used to classify transactions into various accounts, making it easier to track and organize financial information.
Examples of common account names include {report_type}
Also return the page number on which the account name is mentioned. There can be multiple accounts only if there are multiple pages else only one account is possible. It is also possible that there is only one account name mentioned for multiple pages in case of multi page tables. An easy way to understand the account name is to look for the company name next to it. For example, an account name will be mentioned as Company ABC, Account Name. It is possible for the Company Name to not be mentioned. 
Find the company name if it is mentioned next to the account as well.\n
Additionally, assign one of the following subcategories to the account: {subcategories} or OtherStatement.
If you do not find any of these values ({subcategories}) applicable for subcategory, return OtherStatement for subcategory . 
You can also use the data/content of a page to determine whether it qualifies as a valid financial statement of the given {subcategories}, even if the subcategory is not mentioned explicitly taking a step by step approach. 
An easy way to understand the account name is to look for the company name next to it. For example, an account name will be mentioned as Company ABC, Account Name. 
If not mentioned together explicitly, infer the account name from the age by combing the company name and the financial statement provided it belongs to one of these {subcategories}.\n
Find the company name if it is not mentioned next to the account as well.
Set is_continuous flag to true if the Account name mentions that it is a continuation of a previous account. It will mention words like continued, (in continuation),  
Set is_continuous to true if you take a step by step approach and think that there is continuation of a table or not the next pages. Do not set is_continuous flag if you don't feel that a table is continuing over multiple pages. In order to set is_continuous for a a multi page table, set is_continuous to False for the first table in the continuation group. For all subsequent pages set is_continuous to True and ensure account name is exactly the same for all the tables in the same continuation group. Donot set is_continuous if there is for example an income statement on page 1 for YTD and income statement on page 2 for September 2023. Similarly don set is_continuous to true if there is Balance Sheet on page 7 for year 2023 and Balance sheet on page 8 for year 2024. Only if a financial statements looks to be broken across pages or the next pages explicitly state that they are a continuation set is_continuous to True. For example, one page has Balance Sheet's Assets section and the next page has liabilities. Another example would be a cash flow statement where one page has cash flows from operating activities, next page has cash flows from investing activities and the another page has cash flows from financing activities. In such cases set is_continuous to True. 
If you think a financial statement lies on page 1, then thoroughly check page 2 as well as often statements are clustered together. So if you find statements on pages 3,5, then its highly likely there is statements on page 4 as well. Just ensure that the statement if present is a continuation or a valid statement belonging to one of the subcategories: {subcategories}
Extract accounts in JSON format given: {format_instructions}\ Return only the output in required JSON format nothing else.
Extract the account name as 'Company Name, Account Name' and the subcategory from the following context: {input}

    """




SUMMARY_PROMPT = """
Use the following Instructions to fill the Template with content from the provided 10-Q filing as Context

<Instructions>
**Instructions for Filling the Quarterly Financial Statement Template**

These instructions will guide you through the process of accurately filling out the Quarterly Financial Statement Template. Ensure that each section is completed with precise information, following the provided writing style guidelines and incorporating the suggested content hints.

---

### **General Writing Style Guidelines:**

- **Tone:** Use a professional and objective tone throughout the document.
- **Clarity:** Be clear and concise, avoiding unnecessary jargon or overly complex sentences.
- **Consistency:** Maintain consistent formatting, units of measurement, and terminology.
- **Numerical Accuracy:** Verify all figures against official financial statements to ensure accuracy.
- **Percentages and Comparisons:** Include percentage changes year-over-year (% YoY) to highlight performance trends.
- **Formatting:** Utilize bullet points and headings to organize information effectively.

---

### **Section-by-Section Instructions:**

#### **1. Company Information:**

- **Company Name:** Replace `[Company Name]` with the full legal name of the company.
- **Fiscal Quarter and Year:** Replace `Q[Quarter] [Fiscal Year] ([Month])` with the specific fiscal quarter number, fiscal year, and the ending month of the quarter.

  **Example:**
  ```
  **Company Name:** Tesla, Inc. reported fiscal Q3 2024 (September) results.
  ```

---

#### **2. Summary:**

Provide a concise overview of the company's financial performance for the quarter.

- **Opening Statement:**
  - Replace `[Company Name]` with the company's name.
  - State that the company reported its financial results for the specified quarter and fiscal year.

- **Performance Description:**
  - Describe the overall performance (e.g., "strong performance," "moderate growth," "decline in revenues").
  - Include the percentage increase or decrease in total revenues year-over-year.

- **Key Achievements:**
  - Highlight significant milestones or accomplishments achieved during the quarter.
  - Be specific about what was achieved (e.g., "achieved record vehicle production numbers," "expanded market presence in Asia").

- **Challenges Faced:**
  - Briefly mention any obstacles or challenges the company encountered.
  - Examples may include "supply chain disruptions," "increased competition," or "regulatory hurdles."

- **Strategic Focus Areas:**
  - Describe areas where the company is focusing its efforts for growth or improvement.
  - This might include "increasing manufacturing capacity," "reducing operational costs," or "innovation in product development."

  **Writing Style Tips:**
  - Use present or past tense as appropriate.
  - Keep sentences informative and to the point.
  - Avoid overly promotional language.

  **Example:**
  ```
  **Summary:**
  Tesla, Inc. reported its financial results for the third quarter of 2024, showcasing a strong performance with an increase in total revenues by 8% year-over-year. The company achieved significant milestones in vehicle production and energy storage deployments. Despite facing some challenges in production ramps and external factors affecting operations, Tesla maintained its focus on increasing manufacturing capacity and reducing costs.
  ```

---

#### **3. Financials:**

Detail the company's key financial metrics and performance indicators.

- **KPI (Key Performance Indicators):**
  - List specific KPIs relevant to the company's operations.
  - Replace `[Key Performance Indicator 1/2/3]` with actual KPIs (e.g., "Vehicle production," "Deliveries," "Energy storage deployments").
  - Provide approximate values and units up to the reported quarter.

- **Revenue:**
  - **Total Revenue:** State the total revenue figure and the percentage change compared to the same period last year.
  - **Revenue Categories:** Break down the total revenue into specific categories.
    - Replace `[Revenue Category 1/2/3...]` with actual categories (e.g., "Automotive sales," "Automotive regulatory credits").
    - Provide the revenue amount and the YoY percentage change for each category.

- **Gross Profit and Gross Margin:**
  - Provide the gross profit amount and percentage change YoY.
  - State the gross margin percentage and the change in basis points YoY.

- **Adjusted EBITDA:**
  - Include the Adjusted EBITDA amount if available.
  - If not explicitly mentioned, note that it is not provided in the data.

  **Formatting Tips:**
  - Use bullet points for clarity.
  - Ensure all financial figures are accurate and match official statements.
  - Maintain consistent units (e.g., millions, billions).

  **Example:**
  ```
  ### Financials:
  - **KPI:**
    * Vehicle production: Approximately 1,314,000 units through Q3 2024
    * Deliveries: Approximately 1,294,000 units through Q3 2024
    * Energy storage deployments: 20.41 GWh through Q3 2024

  - **Revenue:** $25,182 million, +8% YoY
    * Automotive sales: $18,831 million, +1% YoY
    * Automotive regulatory credits: $739 million, +33% YoY
    * Automotive leasing: $446 million, -9% YoY
    * Services and other: $2,790 million, +29% YoY
    * Energy generation and storage: $2,376 million, +52% YoY

  - **Gross Profit:** $4,997 million, +20% YoY
    * **Gross Margin:** 19.8%, +190bps YoY

  - **Adjusted EBITDA:** Not explicitly mentioned in the provided data.
  ```

---

#### **4. Credit Metrics:**

Provide information on the company's credit-related financial metrics.

- **Cash Flow from Operations (CFO):**
  - State the CFO amount, specifying that it is Year-to-Date (YTD).

- **Capital Expenditures (Capex):**
  - Provide the Capex amount, indicating it is YTD.

- **Free Cash Flow (FCF):**
  - Include the FCF amount if available.
  - If not explicitly mentioned, note that it is not provided.

- **Cash and Cash Equivalents:**
  - State the amount as of the end date of the reported period.

- **Short-term Investments:**
  - Provide the amount as of the same date.

- **Total Debt:**
  - Replace `[Debt Amount]` with the actual total debt figure as of the reporting date.

- **Reported Net Leverage:**
  - Include the net leverage ratio if available.
  - If not explicitly mentioned, note that it is not provided.

- **Other Cash Flow Items:**
  - List any additional cash flow activities, such as proceeds from issuances of debt or stock options.

  **Formatting Tips:**
  - Present figures in a clear and organized manner.
  - Use consistent date references.

  **Example:**
  ```
  ### Credit Metrics:
  - **Cash Flow from Operations (CFO):** $10,109 million YTD
  - **Capital Expenditures (Capex):** $8,556 million YTD
  - **Free Cash Flow (FCF):** Not explicitly mentioned in the provided data
  - **Cash and Cash Equivalents:** $18,111 million as of September 30, 2024
  - **Short-term Investments:** $15,537 million as of September 30, 2024
  - **Total Debt:** $7,415 million as of September 30, 2024
  - **Reported Net Leverage:** Not explicitly mentioned in the provided data
  - **Other Cash Flow Items:** 
    - Proceeds from issuances of debt: $4,360 million YTD
    - Repayments of debt: $1,783 million YTD
    - Proceeds from exercises of stock options and other stock issuances: $788 million YTD
  ```

---

#### **5. Management Discussion:**

Summarize management's commentary on the company's strategic initiatives and outlook.

- **Focus Areas:**
  - Replace `[Primary Initiatives]` with specific areas the company is focusing on (e.g., "ramping up manufacturing capacity for new products").

- **Areas of Improvement:**
  - Detail ongoing efforts to improve certain aspects of the business (e.g., "improving product performance," "reducing production costs").

- **Expansion Plans:**
  - Describe any plans to expand operations or infrastructure (e.g., "expanding global infrastructure," "enhancing service networks").

- **Technologies:**
  - Highlight any new technologies or features introduced, especially those based on emerging technologies (e.g., "artificial intelligence," "enhanced capabilities").

  **Writing Style Tips:**
  - Use bullet points for readability.
  - Keep each point focused on a single idea.
  - Reflect the company's strategic priorities.

  **Example:**
  ```
  ### Management Discussion:
  - Focus on ramping up manufacturing capacity for new vehicle models such as Cybertruck and Tesla Semi.
  - Continued efforts in improving vehicle performance and reducing production costs.
  - Emphasis on expanding global infrastructure, including service and charging networks.
  - Introduction of new products and features based on artificial intelligence, such as enhanced Autopilot and Full Self-Driving capabilities.
  ```

---

#### **6. Additional Context:**

Provide a broader overview of the company's mission, investments, challenges, and positioning within the industry.

- **Company Mission or Vision:**
  - Replace `[Company Name]` and `[Company Mission or Vision]` with the company's mission statement or overarching goals.

- **Key Investment Areas:**
  - Mention significant areas where the company is investing resources (e.g., "AI," "manufacturing capabilities").

- **Benefits:**
  - Describe the intended benefits of these investments (e.g., "enhance product performance," "reduce costs").

- **Challenges Faced:**
  - Acknowledge any challenges impacting the company (e.g., "inflationary pressures," "supply chain disruptions").

- **Strategic Commitments:**
  - Highlight commitments that drive the company's success (e.g., "expanding product roadmap," "improving operational efficiencies").

- **Industry or Sector:**
  - Specify the industry in which the company operates (e.g., "automotive and energy sectors").

  **Writing Style Tips:**
  - Maintain an objective perspective.
  - Balance positive developments with acknowledgment of challenges.
  - Keep the focus on strategic positioning and future outlook.

  **Example:**
  ```
  ### Additional Context:
  Tesla, Inc. is dedicated to accelerating the transition to sustainable energy through the development and sale of electric vehicles, energy storage products, and advanced technologies. The company's significant investments in AI and manufacturing capabilities aim to enhance product performance and reduce costs. Despite facing challenges such as inflationary pressures and supply chain disruptions, Tesla's robust financial performance and strategic initiatives position it well for future growth. The company’s commitment to expanding its product roadmap and improving operational efficiencies continues to drive its success in the highly competitive automotive and energy sectors.
  ```

---

### **Final Review Checklist:**

Before finalizing the document, ensure the following:

- **Accuracy:**
  - All figures and data points are accurate and have been cross-checked with official financial statements.

- **Completeness:**
  - No placeholders remain in the template; all have been properly replaced with relevant information.

- **Consistency:**
  - Formatting is consistent throughout the document.
  - Units of measurement and terminology are used uniformly.

- **Clarity:**
  - The document is easy to read and understand.
  - Complex information is presented in a clear and concise manner.

- **Professionalism:**
  - The tone remains professional and objective.
  - The writing adheres to standard business communication practices.

---

By following these instructions and guidelines carefully, you will produce a comprehensive and professional quarterly financial statement that accurately reflects the company's performance and provides valuable insights for stakeholders.
</Instructions>

<Template>
## Quarterly Financial Statement Template

**Company Name:** [Company Name] reported fiscal Q[Quarter] [Fiscal Year] ([Month]) results.

**Summary:**
[Company Name] reported its financial results for the [ordinal] quarter of [Fiscal Year], showcasing a [performance description] with an increase in total revenues by [percentage change]% year-over-year. The company achieved significant milestones in [Key Achievements]. Despite facing some challenges in [Challenges Faced], [Company Name] maintained its focus on increasing [Strategic Focus Areas].

### Financials:
- **KPI:**
  * [Key Performance Indicator 1]: Approximately [Value] [Units] through Q[Quarter] [Fiscal Year]
  * [Key Performance Indicator 2]: Approximately [Value] [Units] through Q[Quarter] [Fiscal Year]
  * [Key Performance Indicator 3]: [Value] [Units] through Q[Quarter] [Fiscal Year]

- **Revenue:** [Total Revenue], [Percentage Change]% YoY
  * [Revenue Category 1]: [Amount], [Percentage Change]% YoY
  * [Revenue Category 2]: [Amount], [Percentage Change]% YoY
  * [Revenue Category 3]: [Amount], [Percentage Change]% YoY
  * [Revenue Category 4]: [Amount], [Percentage Change]% YoY
  * [Revenue Category 5]: [Amount], [Percentage Change]% YoY

- **Gross Profit:** [Gross Profit Amount], [Percentage Change]% YoY
  * **Gross Margin:** [Gross Margin Percentage]%, [Change in Basis Points]bps YoY

- **Adjusted EBITDA:** [Adjusted EBITDA Amount] (if available)

### Credit Metrics:
- **Cash Flow from Operations (CFO):** [CFO Amount] Year-to-Date (YTD)
- **Capital Expenditures (Capex):** [Capex Amount] YTD
- **Free Cash Flow (FCF):** [FCF Amount] (if available)
- **Cash and Cash Equivalents:** [Amount] as of [Date]
- **Short-term Investments:** [Amount] as of [Date]
- **Total Debt:** [Debt Amount] as of [Date]
- **Reported Net Leverage:** [Net Leverage Ratio] (if available)
- **Other Cash Flow Items:** 
  - Proceeds from issuances of debt: [Amount] YTD
  - Repayments of debt: [Amount] YTD
  - Proceeds from exercises of stock options and other stock issuances: [Amount] YTD

### Management Discussion:
- Focus on [Primary Initiatives, e.g., ramping up manufacturing capacity for new products].
- Continued efforts in [Areas of Improvement, e.g., improving product performance and reducing production costs].
- Emphasis on expanding [Expansion Plans, e.g., global infrastructure, service, and network].
- Introduction of new products and features based on [Technologies, e.g., artificial intelligence, enhanced capabilities].

### Additional Context:
[Company Name] is dedicated to [Company Mission or Vision]. The company's significant investments in [Key Investment Areas] aim to enhance [Benefits, e.g., product performance, cost reduction]. Despite facing challenges such as [Challenges Faced], [Company Name]'s robust financial performance and strategic initiatives position it well for future growth. The company’s commitment to [Strategic Commitments] continues to drive its success in the highly competitive [Industry or Sector].
</Template>

"""