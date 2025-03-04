import json
import pandas as pd
import numpy as np
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Dict, List

import re

load_dotenv()


def get_subset(df):
    if len(df) <= 12:
        return df.copy()
    else:
        # Extract first and last five rows
        part1 = df.head(5)
        part3 = df.tail(4)

        # Middle section (excluding first five and last five)
        middle_part = df.iloc[5:-4]

        # Compute missing values per row in middle_part
        missing_counts = middle_part.isnull().sum(axis=1)

        # Create helper DataFrame to sort and sample
        helper = pd.DataFrame(
            {
                "original_index": middle_part.index,
                "missing": missing_counts,
                "random": np.random.rand(len(middle_part)),
            }
        )

        # Sort by missing count and then by random to shuffle within same missing counts
        helper_sorted = helper.sort_values(["missing", "random"])

        # Select top 5 indices
        selected_indices = helper_sorted["original_index"].head(3).tolist()

        # Extract part2 using the selected indices
        part2 = df.loc[selected_indices]

        # Combine all parts
        subset = pd.concat([part1, part2, part3])

        return subset


def is_dataframe_valid(df):
    invalid_reason = []

    # Condition 1: Check if the entire DataFrame contains missing values
    if df.isna().all().all():
        invalid_reason.append(1)

    # Condition 2: Check if more than 75% of values in every row are similar
    if not df.empty:
        rows_check = df.apply(
            lambda row: (row.value_counts(dropna=False).max() / len(row)) > 0.75, axis=1
        ).all()
        if rows_check:
            invalid_reason.append(2)

    # Condition 3: Check if more than 75% of column names are the same
    col_names = df.columns.tolist()
    if col_names:
        counts = Counter(col_names)
        max_col_count = max(counts.values())
        if (max_col_count / len(col_names)) > 0.75:
            invalid_reason.append(3)

    # Condition 4: Check for none formula
    total_elements = df.size
    # Count occurrences of '#DIV/0!'
    div_by_zero_count = (df == "#DIV/0!").sum().sum()

    # Calculate the percentage
    percentage = (div_by_zero_count / total_elements) * 100
    if percentage >= 25:
        invalid_reason.append(4)

    if len(invalid_reason) > 0:
        return False, invalid_reason

    return True, []


def unique_per_row(df):
    if df is not None:
        unique_str = ""
        unique_rows = []

        # Process columns first
        seen_cols = {}
        unique_columns = []
        for i, col in enumerate(df.columns):
            if col not in seen_cols:
                seen_cols[col] = i
                unique_columns.append(col)
            else:
                unique_columns.append(None)
        unique_rows.append(unique_columns)

        # Process each row
        for _, row in df.iterrows():
            seen = {}
            unique_row = []
            for i, val in enumerate(row):
                if val not in seen:
                    seen[val] = i
                    unique_row.append(val)
                else:
                    unique_row.append(None)
            unique_rows.append(unique_row)

        for i in unique_rows:
            for j in i:
                if j is not None:
                    if str(j) not in unique_str:
                        unique_str = unique_str + "\n" + str(j)

        return unique_str
    else:
        return "No rows removed above header"


def is_int_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class data_validator(BaseModel):
    multi_level_header: Literal["YES", "NO"] = Field(
        ...,
        description="Determine if the header is multi level header or not. 'YES' if multi level headers exist, else 'NO'",
    )
    invalid_header: Literal["YES", "NO"] = Field(
        ...,
        description="Verify whether column headers in the current dataframe are valid. 'YES' if headers are valid, else 'NO'",
    )
    aggregate_row_count: Literal[0, 1, 2, 3, 4] = Field(
        ...,
        description="count how many of the **last 4 rows** contain **totals/aggregates** (not raw data). This will be Integer between `0` (no aggregates) and `4` (all aggregates)",
    )
    column_name_suggestion: Dict[str, str] = Field(
        ...,
        description="Propose logical names for the columns based on given indices and current column name, in form of key value pair. if they are given in from of key value pair then this will Python dictionary where key will be column indices and value will be the suggested column names else empty dictionary {}",
    )
    data_description: str = Field(
        ...,
        description="Data scription in short",
    )


def llm_qc_check(
    instructor_client, model_id, sheet_name, main_table, removed_table, file_name
):
    removed_unique_values_per_row = unique_per_row(removed_table)

    subset_table = get_subset(main_table)
    subset_table = subset_table.to_markdown(index=False)

    dataframe_columns = list(main_table.columns)
    global missing_column_indices

    missing_column_indices = [
        i
        for i, v in enumerate(dataframe_columns)
        if v is None or (isinstance(v, str) and (is_int_string(v) or v.strip() == ""))
    ]

    missing_column_dict = {i: dataframe_columns[i] for i in missing_column_indices}

    if len(missing_column_indices) > 0:
        missing_column_task = f"""
**TASK 4: Column Name Suggestion**
- **Goal**: Propose logical names for the following columns, given in the form of key value pair where key is column indices and value is the current column name `{json.dumps(missing_column_dict)}`.
- **Criteria**:  
    - Analyze **data** and **missing columan value** and also refer to the Removed rows above header if any if the contrains any useful value for this.   
- **Output**: Python dictionary where key will be missing column indices and value will be the new suggested column names."""
    else:
        missing_column_task = """
**TASK 4: Column Name Suggestion**
- **Goal**: In the given data, none of column names are missing.  
- **Output**: Return empty dict if no missing columns."""

    llm_qc_check_res = instructor_client.chat.completions.create(
        model=model_id,  # "claude-3-7-sonnet-20250219", #"gpt-4o",
        max_tokens=8000,
        response_model=data_validator,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an expert spreadsheet data analyst specializing in detecting and resolving errors from spreadsheet-to-dataframe conversions in the given Dataframe in markdown. Your analysis must be rigorous and error-free.

**Background**:  
- Merged cells in spreadsheets are unmerged by **duplicating their value** into all constituent cells during conversion.  
- Risks include duplicate entries, misaligned columns/rows, or logical inconsistencies.  
- The `row_num` column represents the original spreadsheet row number.  
- Provided dataframe includes the **first 5 rows, 3 random rows, and last 4 rows** of the DataFrame. Total rows: `{str(len(main_table))}`. Sheet name is {sheet_name} and file name is {file_name}
- Removed rows above header**: {removed_unique_values_per_row}

**TASK 1: Multi Level Header**  
- **Goal**: Determine if the header (column names) has multi level headers or not. As values are unmerged and due sheet styling, there can be a possibility of multi level headers.
- **Criteria**:  
    - If the **first few rows** contain values that logically group/describe subsequent columns (e.g., "Region: Q1 Sales"), classify as multi level.  
    - If initial rows are **data values** (not metadata), headers are single level.
    - If current column names include duplicate entries, headers are multi leve.
- **Output**: `"YES"` if multi level headers exist, else `"NO"`.

**TASK 2: Invalid Header**  
- **Goal**: Verify whether column headers in the current dataframe are valid.
- **Criteria**:  
  - Headers are invalid if they contains actual data not the expected table column name
  - Headers are invalid if the majority of Header names are either null or numeric or date time values instead of meaningful table column name
- **Output**: `"YES"` if headers are invalid, else `"NO"`.


**TASK 3: Aggregate Row Count**  
- **Goal**: Identify how many of the **last 4 rows** contain **totals/aggregates** (not raw data).  
- **Criteria**:
    1. For **each of the last 4 rows**:
        - Check for keywords: "Total," "Sum," "Average," or **null values** in most of the non-aggregate columns. 
        - Verify if numeric values in these row are **statistically outliers** (e.g., 5x to 10x larger than column value).
    2. Count all number rows meeting **at least one criterion**.   
- **Output**: Integer between `0` (no aggregates) and `4` (all aggregates).

{missing_column_task}

**TASK 5: Data Description**  
- **Goal**: describe the data in short and do not give any suggestions or include information related to above task, refer data sheet & file name and removed rows above header if any, removed rows above header some useful in describing the data.



    """,
            },
            {"role": "user", "content": f"### Pandas dataframe\n{subset_table}"},
        ],
    )
    return llm_qc_check_res


def rename_columns_by_index(df, column_mapping):
    """
    Rename DataFrame columns using a dictionary mapping of column indices (as strings) to new names

    Args:
        df (pd.DataFrame): DataFrame whose columns need to be renamed
        column_mapping (dict): Dictionary with format {str_index: new_column_name}

    Returns:
        pd.DataFrame: DataFrame with updated column names
    """
    # Convert string indices to integers and filter valid indices
    valid_indices = {}
    for str_idx, new_name in column_mapping.items():
        try:
            idx = int(str_idx)
            if 0 <= idx < len(df.columns):
                valid_indices[idx] = new_name
        except ValueError:
            continue  # skip invalid index format

    # Create new column names list
    new_columns = list(df.columns)
    for idx, new_name in valid_indices.items():
        new_columns[idx] = new_name

    # Assign new column names to DataFrame
    df.columns = new_columns
    return df


def process_dataframe_header(df, row_number):
    # Convert 1-based row_number to 0-based data start index
    if row_number != 0:
        data_start_idx = row_number - 1

        # If row_number is 1, there's no header row to process
        if data_start_idx <= 0:
            new_df = df.iloc[data_start_idx:].copy()
            return new_df

        header_row_idx = data_start_idx - 1

        # Extract the header row values
        header_row = df.iloc[: header_row_idx + 1].values.tolist()

        original_cols = df.columns.tolist()

        for header_vals in header_row:
            new_columns = []
            for col, val in zip(original_cols, header_vals):
                parts = []
                # Process original column name
                if col is not None and str(col).strip() != "":
                    parts.append(str(col).strip())
                # Process header row value
                if (
                    val is not None
                    and str(val).strip() != ""
                    and not any(str(val).strip() in s for s in parts)
                ):
                    parts.append(str(val).strip())
                # Join parts with underscore, empty if both are missing
                new_col = "_".join(parts) if parts else ""
                new_columns.append(new_col)
            original_cols = new_columns

        # Create a copy and update column names
        new_df = df.copy()
        new_df.columns = new_columns

        # Slice the DataFrame to start from data_start_idx
        new_df = new_df.iloc[data_start_idx:]

        return new_df
    return df


class flatten_multi_level_header(BaseModel):
    row_number: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = Field(
        ...,
        description="The exact row number from which the actual data begins, If there is no multi-header issue then this will be `0`. Otherwise, specify the row number where actual data (not headers) begins",
    )


def are_column_names_not_unique(df):
    columns = df.columns.tolist()
    # Filter out NaN, null, and empty string column names
    filtered_columns = [
        name for name in columns if not (pd.isna(name) or name.strip() == "")
    ]
    # Check if all remaining column names are unique
    return len(filtered_columns) != len(set(filtered_columns))


def check_header_subset_row_condition(df):
    # Filter valid columns (ignore None, empty string, or whitespace-only names)
    valid_columns = []
    for col in df.columns:
        if col is None:
            continue
        col_str = str(col).strip()
        if not col_str:
            continue
        valid_columns.append(col)

    if not valid_columns:
        return False  # No valid columns to check

    df_valid = df[valid_columns]

    # Check if there is at least one row in the valid DataFrame
    if df_valid.empty:
        return False

    first_row = df_valid.iloc[0]

    # Prepare set of valid column names as stripped strings
    # valid_cols_processed = {str(col).strip() for col in valid_columns}

    count = 0
    total_non_ignored = 0

    for i, val in enumerate(first_row):
        # Skip NaN values
        if pd.isna(val):
            continue
        # Convert value to stripped string
        val_str = str(val).strip()
        # Skip empty strings or whitespace-only
        if not val_str or val_str == "":
            continue
        total_non_ignored += 1
        if val_str.lower() in valid_columns[i].lower():
            count += 1

    # If all values in the first row are ignored, return False
    if total_non_ignored == 0:
        return False

    percentage = (count / total_non_ignored) * 100
    print("percentage", percentage)
    return percentage > 50


def llm_multi_level_header_correction(instructor_client, model_id, main_table):
    subset_table = main_table.head(6)
    subset_table = subset_table.to_markdown(index=False)

    llm_multi_level_header_correction_res = instructor_client.chat.completions.create(
        model=model_id,  # "gpt-4o",
        max_tokens=8000,
        response_model=flatten_multi_level_header,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an expert spreadsheet data analyst specializing in detecting and resolving errors from spreadsheet-to-dataframe conversions in the provided Dataframe (shown in markdown format below). Your analysis must be rigorous and error-free.

**Background Context:**
- Merged cells in spreadsheets are converted by **duplicating their value** into all constituent cells during conversion
- Common conversion issues include duplicate entries, misaligned columns/rows, and logical inconsistencies
- The provided dataframe has potential multi-level header issues.
- Only the **first 6 rows** of the DataFrame are shown below. Total rows in the full dataset: `{str(len(main_table))}`

**Your Tasks:**
1. Carefully analyze the provided data and headers, paying special attention to multi-level header issues
2. Identify and report the exact row number from which the actual data begins(row_number) e.g, if acutal data begins from fist row then row_number=1, if actual data begins from 2nd row then row_number = 2 and on : 
   - If there is no multi-header issue, report `0`
   - Otherwise, specify the row number where actual data (not headers) begins
   - Actual data can also have missing values in a row
   - If any row contains numeric values, date or datetime values, then that is the row from where actual data starts and all rows above it belong to headers
""",
            },
            {
                "role": "user",
                "content": f"### Pandas dataframe Preview\n{subset_table}",
            },
        ],
    )
    if llm_multi_level_header_correction_res.row_number != 0:
        print("row_number", llm_multi_level_header_correction_res.row_number)
        main_table = process_dataframe_header(
            main_table, llm_multi_level_header_correction_res.row_number
        )

        return main_table

    else:
        return main_table


def check_aggregation_headers(df):
    """
    Validates DataFrame headers based on specified rules and returns a reason code:
    - 0: All headers are strings and cannot be converted to numeric
    - 1: All headers are numeric/convertible to numeric with values < 100, or all headers are null/empty
    - 2: All headers are numeric/convertible to numeric with values >= 100, or headers are a mix of string and numeric

    Parameters:
    df (pandas.DataFrame): DataFrame to validate headers

    Returns:
    int: Reason code (0, 1, or 2)
    """
    # Get column headers
    headers = list(df.columns)

    # Case 5: If all headers are null/none/empty/whitespace
    if all(not str(h).strip() if pd.notna(h) else True for h in headers):
        return 1

    # Filter out null/none/empty/whitespace headers for cases 1-4
    valid_headers = [h for h in headers if pd.notna(h) and str(h).strip()]

    if not valid_headers:
        return 1  # All headers were filtered out as invalid

    # Function to check if a header can be converted to a numeric value
    def extract_numeric(header):
        if pd.isna(header):
            return None

        header_str = str(header).strip()
        if not header_str:
            return None

        # Extract number from brackets if present
        # Handles formats like "(123)", "[456]"
        bracket_pattern = r"[\(\[]([-]?\d+\.?\d*)[\)\]]"
        bracket_match = re.search(bracket_pattern, header_str)
        if bracket_match:
            try:
                return float(bracket_match.group(1))
            except ValueError:
                return None

        # Try direct conversion if no brackets
        try:
            return float(header_str.replace('"', "").replace("'", ""))
        except ValueError:
            return None

    # Extract numeric values from headers
    numeric_values = [extract_numeric(h) for h in valid_headers]

    # Count how many headers are convertible to numeric
    convertible_count = sum(1 for val in numeric_values if val is not None)

    # Case 4: All headers are strings (not convertible to numeric)
    if convertible_count == 0:
        return 0

    # Case 3: Mix of string and numeric headers
    if convertible_count != len(valid_headers):
        return 2

    # Case 1 & 2: All headers are numeric or convertible to numeric
    all_less_than_100 = all(abs(val) < 100 for val in numeric_values if val is not None)

    if all_less_than_100:
        return 1  # Case 1: All numeric values < 100
    else:
        return 2  # Case 2: At least one numeric value >= 100


class aggregartion_table_header_suggestion(BaseModel):
    list_of_header_names: List[str] = Field(
        ...,
        description="List of suggested header names",
    )
    link_to_base: Literal["YES", "NO"] = Field(
        ...,
        description="Verify whether current aggregated dataframe is linked to the given base dataframe. 'YES' if aggregated dataframe is linked to the given base dataframe, else 'NO'",
    )
    current_dataframe_description: str = Field(
        ...,
        description="describe current dataframe in short",
    )


def llm_aggregartion_table_headers(
    instructor_client, model_id, main_table, sheet_name, final_tables
):
    old_data = ""
    old_data_desciption = ""
    if len(final_tables) > 0:
        if (
            sheet_name == final_tables[-1]["sheet_name"]
            and final_tables[-1]["invalid_reason"] is None
        ):
            old_data = final_tables[-1]["table"]
            if len(old_data) > 6:
                old_data = pd.concat([old_data.head(3), old_data.tail(3)]).to_markdown(
                    index=False
                )
            else:
                old_data = old_data.to_markdown(index=False)
            old_data_desciption = final_tables[-1]["description"]

            old_data = (
                "**Base dataframe**\n\n"
                + old_data
                + f"\n\n **Above Base dataframe description**: {old_data_desciption}"
            )

    subset_table = main_table.head(5)
    subset_table = subset_table.to_markdown(index=False)

    llm_aggregartion_table_headers_res = instructor_client.chat.completions.create(
        model=model_id,  # "gpt-4o",
        max_tokens=8000,
        response_model=aggregartion_table_header_suggestion,
        messages=[
            {
                "role": "system",
                "content": f"""

{old_data if old_data != "" else ""}

You are an expert data analyst specializing in comparing dataframes and suggesting appropriate column names for datasets.

**Background Context:**
{"- The above base dataframe includes the first 3 rows and last 3 rows of the actual raw DataFrame. Both the base dataframe and the following current dataframe belong to the same sheet, with the current dataframe positioned below the base dataframe." if old_data != "" else "- The following current dataframe is the only dataframe in the sheet."}
- The current dataframe may be an aggregated version {"of the base dataframe shown above" if old_data != "" else "of underlying data that isn't shown"}
- Only the **first 6 rows** of the current DataFrame are shown below. Total rows in current dataframe: `{str(len(main_table))}`
- The current dataframe has missing or incorrect header/column names

**Your Tasks:**
1. Carefully analyze thes dataframe, their structure and base dataframe description if given.
2. {"Based on both the base dataframe and base dataframe description and current dataframe" if old_data != "" else "Based on the current dataframe"}, suggest the most appropriate column names for the current dataframe.
3. Provide your suggested column names as a Python list, where the first element corresponds to the first column of the current dataframe, the second element to the second column, and so on.
4. {"Determine if the current dataframe is linked to (likely an aggregation of) the base dataframe. Output 'YES' if the current data appears to be linked to the base data, otherwise output 'NO'." if old_data != "" else "Since there is no base dataframe provided, the linkage question is not applicable, so output 'NO'."}
5. Generate the description of current dataframe in short and never menteion the base and current dataframe just refer to their description and create stand along description with the deatils

Your response should contain ONLY:
1. A Python list of suggested column names
2. The word 'YES' or 'NO' regarding linkage
3. Description of current dataframe

""",
            },
            {"role": "user", "content": f"### **Current dataframe**:\n{subset_table}"},
        ],
    )
    return llm_aggregartion_table_headers_res


def aggregartion_table_header_correction(
    instructor_client, model_id, main_table, sheet_name, final_tables
):
    aggregation_headers_reason = check_aggregation_headers(main_table)
    if aggregation_headers_reason == 0:
        llm_aggregartion_table_headers_res = llm_aggregartion_table_headers(
            instructor_client, model_id, main_table, sheet_name, final_tables
        )

        return (
            main_table,
            llm_aggregartion_table_headers_res.link_to_base,
            llm_aggregartion_table_headers_res.current_dataframe_description,
        )
    else:
        llm_aggregartion_table_headers_res = llm_aggregartion_table_headers(
            instructor_client, model_id, main_table, sheet_name, final_tables
        )
        if aggregation_headers_reason == 1:
            if len(llm_aggregartion_table_headers_res.list_of_header_names) == len(
                main_table.columns
            ):
                main_table.columns = (
                    llm_aggregartion_table_headers_res.list_of_header_names
                )
        else:
            if len(llm_aggregartion_table_headers_res.list_of_header_names) == len(
                main_table.columns
            ):
                main_table = pd.concat(
                    [
                        pd.DataFrame([main_table.columns], columns=main_table.columns),
                        main_table,
                    ],
                    ignore_index=True,
                )
                main_table.columns = (
                    llm_aggregartion_table_headers_res.list_of_header_names
                )

    return (
        main_table,
        llm_aggregartion_table_headers_res.link_to_base,
        llm_aggregartion_table_headers_res.current_dataframe_description,
    )


def clean_tables(instructor_client, model_id, table_dict, file_name, final_tables):
    final_table = {}
    sheet_name = table_dict["sheet_name"]
    main_table = table_dict["table"]
    removed_table = table_dict["removed"]

    final_table["sheet_name"] = sheet_name
    final_table["link_to_previous_table"] = "NO"

    df_valid, invalid_reason = is_dataframe_valid(main_table)
    if df_valid:
        llm_qc_check_res = llm_qc_check(
            instructor_client,
            model_id,
            sheet_name,
            main_table,
            removed_table,
            file_name,
        )

        final_table["description"] = llm_qc_check_res.data_description

        if llm_qc_check_res.column_name_suggestion != {}:
            main_table = rename_columns_by_index(
                main_table, llm_qc_check_res.column_name_suggestion
            )

        if llm_qc_check_res.multi_level_header == "YES":
            if are_column_names_not_unique(main_table):
                main_table = llm_multi_level_header_correction(
                    instructor_client, model_id, main_table
                )

            header_subset_row_condition = check_header_subset_row_condition(main_table)

            while header_subset_row_condition:
                print("check_header_subset_row_condition")
                main_table = process_dataframe_header(main_table, 2)
                header_subset_row_condition = check_header_subset_row_condition(
                    main_table
                )

        if llm_qc_check_res.aggregate_row_count > 0:
            if len(main_table) == llm_qc_check_res.aggregate_row_count:
                final_table["description"] = (
                    final_table["description"]
                    + "\n\nThis table presents a comprehensive aggregation of data"
                )
                (
                    main_table,
                    final_table["link_to_previous_table"],
                    final_table["description"],
                ) = aggregartion_table_header_correction(
                    instructor_client, model_id, main_table, sheet_name, final_tables
                )
                if final_table["link_to_previous_table"] == "YES":
                    final_table["description"] = (
                        final_table["description"]
                        + "\n\nThis table presents a comprehensive aggregation of data of previous table"
                    )

            else:
                if llm_qc_check_res.aggregate_row_count > 1:
                    final_table["description"] = (
                        final_table["description"]
                        + f"\n\nThe final {llm_qc_check_res.aggregate_row_count} rows in the table represent aggregated data."
                    )
                else:
                    final_table["description"] = (
                        final_table["description"]
                        + f"\n\nThe Last row in the table represent aggregated data."
                    )

        ## TODO LLM prompting for invalid_header.. skipping this because of time crunch
        for col in main_table.columns:
            try:
                if main_table[col].dtype == "object":
                    main_table[col] = (
                        main_table[col].astype(str).str.replace("\xa0", "")
                    )
            except:
                pass

        final_table["table"] = main_table
        final_table["invalid_reason"] = None

        return final_table

    else:
        if 4 in invalid_reason:
            final_table["table"] = main_table
            final_table["invalid_reason"] = invalid_reason
            final_table["description"] = "Invalid table"
            return final_table
        else:
            return None
