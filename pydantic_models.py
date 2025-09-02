from pydantic import BaseModel, Field


class Accounts(BaseModel):
    class Account(BaseModel):
        account_name: str = Field(description="The name of the account")
        page_number: int = Field(description="The page number of the account")
        subcategory: str = Field(
            description="Subcategory of the account. Possible values: BalanceSheet, IncomeStatement, CashFlow or OtherStatement."
        )
        is_continuous: bool = Field(
            description="Set to true if one account is continuation of another account. Set to False by default",
            default=False,
        )

    accounts: list[Account]