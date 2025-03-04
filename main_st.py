import streamlit as st
import pandas as pd
import os
import tempfile
from anthropic import Anthropic
from openai import OpenAI
import instructor
from pre_processing import detect_and_extract_tables
from post_processing import clean_tables

# Set page configuration
st.set_page_config(page_title="Excel Table Extractor", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown(
    """
<style>
    .invalid-table {
        border: 2px solid red;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for pagination
if "table_pages" not in st.session_state:
    st.session_state.table_pages = {}


# Function for running table extraction and processing
def process_excel_file(excel_file_path, model_name="anthropic"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Initialize the appropriate client
        status_text.write("Initializing client...")
        if model_name == "anthropic":
            instructor_client = instructor.from_anthropic(Anthropic())
            model_id = "claude-3-7-sonnet-20250219"
        elif model_name == "openai":
            instructor_client = instructor.from_openai(OpenAI())
            model_id = "gpt-4o-2024-11-20"

        progress_bar.progress(10)

        # Step 2: Detect and extract tables
        status_text.write("Detecting and extracting tables from Excel...")
        tables = detect_and_extract_tables(excel_file_path)

        if not tables:
            status_text.warning("No tables detected in the Excel file.")
            return []

        progress_bar.progress(30)

        # Step 3: Process each table
        file_name = os.path.basename(excel_file_path)
        final_tables = []

        for i, table in enumerate(tables):
            progress = 30 + ((i / len(tables)) * 70)  # Scale progress from 30-100%
            progress_bar.progress(int(progress))
            status_text.write(f"Processing table {i + 1} of {len(tables)}...")

            cln_table = clean_tables(
                instructor_client, model_id, table, file_name, final_tables
            )
            if cln_table is not None:
                final_tables.append(cln_table)

        # Complete the progress bar
        progress_bar.progress(100)
        status_text.success("Processing complete!")
        return final_tables

    except Exception as e:
        status_text.error(f"Error during processing: {str(e)}")
        return []


# Function to display a table with pagination
def display_table_with_pagination(table_dict, table_index):
    # Initialize pagination for this table if not already done
    if table_index not in st.session_state.table_pages:
        st.session_state.table_pages[table_index] = 0

    # Check if table has invalid_reason and set appropriate styling
    is_invalid = table_dict.get("invalid_reason") is not None

    # Start container with red border if invalid
    # if is_invalid:
    #     st.markdown(f"<div class='invalid-table'>", unsafe_allow_html=True)

    # Display table metadata (skip None values and 'table' key)
    for key, value in table_dict.items():
        if key != "table" and value is not None:
            if key == "invalid_reason" and is_invalid:
                st.error(f"**Invalid Reason**: {value}")
            else:
                st.write(f"**{key.replace('_', ' ').title()}**: {value}")

    # Get the pandas DataFrame
    df = table_dict["table"]

    # Display the dataframe
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except:
        st.markdown(
            f"""
<div style="
        height: 60%;       /* Fixed height */
        width: 90%;         /* Full width */
        overflow: auto;      /* Enable both X and Y scrolling */
        border: 1px solid #ddd;  /* Optional border */
        padding: 10px;       /* Optional padding */
        background-color: #f9f9f9;
    ">
                    
{df.to_markdown(index=False)}

</div>
""",
            unsafe_allow_html=True,
        )

    # Close the container if it was invalid
    if is_invalid:
        st.markdown("</div>", unsafe_allow_html=True)


# Main app function
def main():
    st.title("📊 Excel Table Extractor and Processor")
    st.write("Upload an Excel file to extract and process tables")

    # Model selection dropdown
    model_name = st.selectbox(
        "Select LLM Model",
        options=["anthropic", "openai"],
        index=0,  # Default to anthropic
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Display file information
        st.write(f"**File uploaded:** {uploaded_file.name}")

        # Process button
        if st.button("Process Tables", type="primary"):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            try:
                # Call the processing function with a spinner
                with st.spinner("Processing tables... This may take a while."):
                    tables = process_excel_file(temp_file_path, model_name)

                    # Store the results in session state for display
                    st.session_state.tables = tables
                    st.session_state.processed = True
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

    # Display results if available
    if "processed" in st.session_state and st.session_state.processed:
        st.header("Extracted Tables")

        if not st.session_state.tables:
            st.warning("No tables were extracted from the file.")
        else:
            st.success(f"Successfully extracted {len(st.session_state.tables)} tables!")

            # Display each table with pagination
            for i, table_dict in enumerate(st.session_state.tables):
                st.subheader(f"Table {i + 1}:")
                display_table_with_pagination(table_dict, i)
                st.markdown("---")  # Separator between tables


# Run the app
if __name__ == "__main__":
    main()
