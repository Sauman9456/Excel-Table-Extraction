from anthropic import Anthropic
import os
from openai import OpenAI
import instructor
from pre_processing import detect_and_extract_tables
from post_processing import clean_tables


def run(excel_file_path, model_name="anthropic"):
    file_name = os.path.basename(excel_file_path)
    if model_name == "anthropic":
        instructor_client = instructor.from_anthropic(Anthropic())
        model_id = "claude-3-7-sonnet-20250219"
    elif model_name == "openai":
        instructor_client = instructor.from_openai(OpenAI())
        model_id = "gpt-4o-2024-11-20"

    tables = detect_and_extract_tables(excel_file_path)

    final_tables = []
    for table in tables:
        cln_table = clean_tables(
            instructor_client, model_id, table, file_name, final_tables
        )
        if cln_table is not None:
            final_tables.append(cln_table)
    return final_tables
