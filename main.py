"""
app.py - B2B Report Analyzer
Automated SKU and Reason extraction from Odoo support tickets
Processes multiple sheets and fills only empty cells in columns C & D
For Vive Health Quality Management
"""

import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="B2B Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import AI handlers
try:
    from ai_handler import AIHandler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.error("AI handler module not found. Please ensure ai_handler.py is in the same directory.")

# Professional CSS styling
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary: #1e88e5;
        --secondary: #43a047;
        --accent: #ff6f00;
        --danger: #e53935;
        --dark: #212121;
        --light: #f5f5f5;
        --muted: #9e9e9e;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid var(--primary);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid var(--secondary);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid var(--accent);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 4px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #1565c0;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: var(--primary);
    }
    
    /* Data preview */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'original_file' not in st.session_state:
        st.session_state.original_file = None
    if 'ai_handler' not in st.session_state:
        st.session_state.ai_handler = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'report_type' not in st.session_state:
        st.session_state.report_type = 'Return'
    if 'total_rows' not in st.session_state:
        st.session_state.total_rows = 0
    if 'skus_extracted' not in st.session_state:
        st.session_state.skus_extracted = 0
    if 'success_rate' not in st.session_state:
        st.session_state.success_rate = 0.0

def extract_sku_from_description(description, ai_handler=None):
    """Extract SKU from description using AI or patterns"""
    if pd.isna(description):
        return ""
    
    description = str(description)
    
    # Try AI extraction first if available
    if ai_handler:
        try:
            sku = ai_handler.extract_sku(description)
            if sku and sku != "NOT_FOUND":
                return sku
        except Exception as e:
            logger.warning(f"AI SKU extraction failed: {e}")
    
    # Fallback to pattern matching
    # Common SKU patterns for Vive Health products
    patterns = [
        r'\b(LVA\d{4}[A-Z0-9\-]*)\b',  # LVA1004-UPC
        r'\b(SUP\d{4}[A-Z0-9\-]*)\b',  # SUP series
        r'\b(MOB\d{4}[A-Z0-9\-]*)\b',  # MOB series
        r'\b(RHB\d{4}[A-Z0-9\-]*)\b',  # RHB series
        r'\b([A-Z]{3}\d{4}[A-Z0-9\-]*)\b',  # General pattern
        r'SKU[:\s]+([A-Z0-9\-]+)',  # Explicit SKU mention
        r'Item[:\s]+([A-Z0-9\-]+)',  # Item number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return ""

def extract_reason_from_description(description, ai_handler=None):
    """Extract return/refund reason from description using AI"""
    if pd.isna(description):
        return ""
    
    description = str(description)
    
    # Use AI extraction if available
    if ai_handler:
        try:
            reason = ai_handler.extract_reason(description)
            if reason and reason != "NOT_FOUND":
                return reason
        except Exception as e:
            logger.warning(f"AI reason extraction failed: {e}")
    
    # Fallback to keyword matching
    keywords = {
        'defective': ['defect', 'broken', 'not working', 'malfunction', 'damaged'],
        'wrong item': ['wrong', 'incorrect', 'not what ordered'],
        'not needed': ['no longer need', "don't need", 'changed mind'],
        'quality issue': ['poor quality', 'cheap', 'flimsy'],
        'size issue': ['too small', 'too large', 'wrong size', "doesn't fit"],
        'missing parts': ['missing', 'incomplete', 'not all parts'],
    }
    
    description_lower = description.lower()
    for reason, terms in keywords.items():
        if any(term in description_lower for term in terms):
            return reason
    
    return "Other"

def validate_sheet_columns(df, sheet_name):
    """Validate if sheet has proper columns C and D as SKU and Reason"""
    # Check if we have at least 4 columns (A, B, C, D)
    if len(df.columns) < 4:
        logger.warning(f"Sheet '{sheet_name}' has less than 4 columns")
        return False
    
    # Get column C (index 2) and column D (index 3)
    col_c = df.columns[2] if len(df.columns) > 2 else None
    col_d = df.columns[3] if len(df.columns) > 3 else None
    
    # Check if columns are named SKU and Reason (case-insensitive)
    has_sku = col_c and 'sku' in str(col_c).lower()
    has_reason = col_d and 'reason' in str(col_d).lower()
    
    if not has_sku or not has_reason:
        logger.info(f"Sheet '{sheet_name}' - Column C: {col_c}, Column D: {col_d}")
        if not has_sku and col_c:
            st.warning(f"⚠️ Sheet '{sheet_name}': Column C is '{col_c}' but should be 'SKU'")
        if not has_reason and col_d:
            st.warning(f"⚠️ Sheet '{sheet_name}': Column D is '{col_d}' but should be 'Reason'")
        return False
    
    return True

def process_single_sheet(df, sheet_name, report_type, ai_handler):
    """Process a single sheet to extract SKU and Reason"""
    # Validate columns
    if not validate_sheet_columns(df, sheet_name):
        return None, 0, 0
    
    # Get actual column names
    col_display = df.columns[0] if len(df.columns) > 0 else 'Display Name'
    col_description = df.columns[1] if len(df.columns) > 1 else 'Description'
    col_sku = df.columns[2] if len(df.columns) > 2 else 'SKU'
    col_reason = df.columns[3] if len(df.columns) > 3 else 'Reason'
    
    # Process only rows with empty SKU or Reason
    processed_df = df.copy()
    rows_processed = 0
    successful_extractions = 0
    
    for idx, row in df.iterrows():
        # Check if SKU or Reason is empty
        sku_empty = pd.isna(row[col_sku]) or str(row[col_sku]).strip() == ''
        reason_empty = pd.isna(row[col_reason]) or str(row[col_reason]).strip() == ''
        
        # Skip if both are already filled
        if not sku_empty and not reason_empty:
            continue
        
        rows_processed += 1
        
        # Get description
        description = row[col_description] if col_description in df.columns else ''
        
        # Extract SKU if empty
        if sku_empty:
            sku = extract_sku_from_description(description, ai_handler)
            processed_df.at[idx, col_sku] = sku
            if sku and sku != "":
                successful_extractions += 1
        
        # Extract Reason if empty
        if reason_empty:
            reason = extract_reason_from_description(description, ai_handler)
            processed_df.at[idx, col_reason] = reason
    
    # Add metadata
    processed_df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    processed_df['Report_Type'] = report_type
    processed_df['Sheet_Name'] = sheet_name
    
    return processed_df, rows_processed, successful_extractions

def process_excel_file_all_sheets(file, report_type):
    """Process all sheets in the Excel file"""
    # Initialize AI handler if available
    ai_handler = None
    if AI_AVAILABLE and st.session_state.ai_handler is None:
        try:
            ai_handler = AIHandler()
            st.session_state.ai_handler = ai_handler
            
            # Test AI connection
            status = ai_handler.get_status()
            if status['available']:
                st.success(f"✅ AI Connected: {status['provider']}")
            else:
                st.warning("⚠️ AI not available, using pattern matching")
                ai_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            st.warning("⚠️ AI initialization failed, using pattern matching")
    else:
        ai_handler = st.session_state.ai_handler
    
    # Read all sheets
    excel_file = pd.ExcelFile(file)
    sheet_names = excel_file.sheet_names
    
    st.info(f"📊 Found {len(sheet_names)} sheet(s) in the Excel file")
    
    # Process each sheet
    all_processed_dfs = []
    total_rows_processed = 0
    total_successful = 0
    sheet_summary = []
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        sheet_details = st.empty()
    
    for idx, sheet_name in enumerate(sheet_names):
        # Update overall progress
        overall_progress.progress((idx) / len(sheet_names))
        status_text.text(f"Processing sheet {idx + 1} of {len(sheet_names)}: {sheet_name}")
        
        # Read sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Show sheet details
        sheet_details.info(f"Sheet '{sheet_name}': {len(df)} rows")
        
        # Process sheet
        processed_df, rows_processed, successful = process_single_sheet(
            df, sheet_name, report_type, ai_handler
        )
        
        if processed_df is not None and rows_processed > 0:
            all_processed_dfs.append(processed_df)
            total_rows_processed += rows_processed
            total_successful += successful
            
            sheet_summary.append({
                'Sheet': sheet_name,
                'Total Rows': len(df),
                'Rows Processed': rows_processed,
                'Successful Extractions': successful
            })
        elif processed_df is not None:
            # Sheet had no empty cells to process
            sheet_summary.append({
                'Sheet': sheet_name,
                'Total Rows': len(df),
                'Rows Processed': 0,
                'Successful Extractions': 0,
                'Note': 'All SKU and Reason cells already filled'
            })
        else:
            # Sheet validation failed
            sheet_summary.append({
                'Sheet': sheet_name,
                'Total Rows': len(df),
                'Rows Processed': 0,
                'Successful Extractions': 0,
                'Note': 'Invalid column structure (SKU/Reason not in columns C/D)'
            })
    
    # Final progress update
    overall_progress.progress(1.0)
    status_text.text("Processing complete!")
    sheet_details.empty()
    
    # Clear progress container
    progress_container.empty()
    
    # Show summary
    if sheet_summary:
        st.markdown("### 📊 Processing Summary by Sheet")
        summary_df = pd.DataFrame(sheet_summary)
        st.dataframe(summary_df)
    
    # Check if any sheets were successfully processed
    if not all_processed_dfs:
        st.error("""
        ❌ No sheets could be processed. Please ensure:
        - Column C is titled "SKU"
        - Column D is titled "Reason"
        - There are empty cells in columns C or D to fill
        
        Current column positions matter - SKU must be in column C and Reason in column D.
        """)
    
    return all_processed_dfs, total_rows_processed, total_successful

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">B2B Report Analyzer</h1>
        <p class="subtitle">Automated SKU and Reason Extraction from Odoo Support Tickets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Report Configuration")
        
        # Report type selection
        st.session_state.report_type = st.radio(
            "Select Report Type",
            ["Return", "Refund"],
            help="This will determine the output filename"
        )
        
        st.markdown("---")
        
        # Help section
        with st.expander("📖 How to Use", expanded=True):
            st.markdown("""
            **Step 1: Export from Odoo**
            1. Go to Support Tickets in Odoo
            2. Filter by contains "return"
            3. Select all tickets
            4. Export to Excel
            
            **Step 2: File Requirements**
            - Excel file with one or more sheets
            - **Column C**: Must be titled "SKU"
            - **Column D**: Must be titled "Reason"
            - **Column B**: Should contain Description
            - Tool processes ALL sheets automatically
            - Only updates empty cells in columns C & D
            
            **Step 3: Process**
            - AI extracts SKU and Reason from Description
            - Existing data in columns C & D is preserved
            - Each sheet is processed independently
            - Download the fully processed file
            
            **Expected SKU Formats:**
            - LVA1004-UPC
            - SUP1001
            - MOB2003
            - RHB3002
            
            **Note:** The tool ONLY processes rows where 
            Column C (SKU) or Column D (Reason) is empty.
            """)
        
        st.markdown("---")
        
        # AI Status
        if AI_AVAILABLE:
            st.markdown("### 🤖 AI Status")
            if st.session_state.ai_handler:
                status = st.session_state.ai_handler.get_status()
                if status['available']:
                    st.success(f"Provider: {status['provider']}")
                else:
                    st.warning("AI unavailable")
            else:
                st.info("AI not initialized")
        
        # Stats
        if st.session_state.processing_complete:
            st.markdown("---")
            st.markdown("### 📊 Processing Stats")
            st.metric("Total Rows Processed", st.session_state.total_rows)
            st.metric("SKUs Extracted", st.session_state.skus_extracted)
            st.metric("Success Rate", f"{st.session_state.success_rate:.1f}%")
            if isinstance(st.session_state.processed_data, list):
                st.metric("Sheets Processed", len(st.session_state.processed_data))
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Odoo Export")
        st.markdown("""
        <div class="info-box">
        <strong>File Requirements:</strong><br>
        • Excel file from Odoo with multiple sheets<br>
        • Column C must be titled "SKU"<br>
        • Column D must be titled "Reason"<br>
        • Tool will scan ALL sheets and process only empty cells<br>
        • Existing data in columns C & D will be preserved
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📝 Selected Report Type")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">B2B {st.session_state.report_type} Report</div>
            <div class="metric-label">Output filename format</div>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel file",
        type=['xlsx', 'xls'],
        help="Upload the Excel export from Odoo"
    )
    
    if uploaded_file:
        # Read file
        try:
            # Store the file in session state for processing
            st.session_state.original_file = uploaded_file
            
            # Get basic file info
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Display file info
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sheets", len(sheet_names))
            with col2:
                total_rows = sum(len(pd.read_excel(excel_file, sheet_name=sheet)) for sheet in sheet_names)
                st.metric("Total Rows", total_rows)
            with col3:
                # Check for empty cells in columns C and D across all sheets
                empty_cells = 0
                for sheet in sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    if len(df.columns) >= 4:
                        col_c = df.columns[2]
                        col_d = df.columns[3]
                        empty_cells += df[col_c].isna().sum() + df[col_d].isna().sum()
                st.metric("Empty Cells (C&D)", empty_cells)
            
            # Show sheets info
            with st.expander("📑 Sheet Information", expanded=True):
                sheet_info = []
                for sheet_name in sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Check columns C and D
                    if len(df.columns) >= 4:
                        col_c = df.columns[2]
                        col_d = df.columns[3]
                        col_c_empty = df[col_c].isna().sum()
                        col_d_empty = df[col_d].isna().sum()
                        
                        sheet_info.append({
                            'Sheet Name': sheet_name,
                            'Rows': len(df),
                            'Column C': col_c,
                            'Column D': col_d,
                            'Empty in C': col_c_empty,
                            'Empty in D': col_d_empty,
                            'Ready to Process': 'Yes' if (col_c_empty > 0 or col_d_empty > 0) else 'No'
                        })
                    else:
                        sheet_info.append({
                            'Sheet Name': sheet_name,
                            'Rows': len(df),
                            'Column C': 'N/A',
                            'Column D': 'N/A',
                            'Empty in C': 0,
                            'Empty in D': 0,
                            'Ready to Process': 'No - Insufficient columns'
                        })
                
                sheet_df = pd.DataFrame(sheet_info)
                st.dataframe(sheet_df)
                
                # Show preview of first sheet with data
                for sheet_name in sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if len(df) > 0:
                        st.markdown(f"#### Preview: {sheet_name}")
                        # Show first 5 rows, focusing on columns A-D
                        preview_cols = df.columns[:4] if len(df.columns) >= 4 else df.columns
                        st.dataframe(df[preview_cols].head(5))
                        break
            
            # Process button
            if st.button("🚀 Process All Sheets", type="primary", use_container_width=True):
                with st.spinner("Processing all sheets..."):
                    # Reset the file pointer
                    uploaded_file.seek(0)
                    
                    # Process all sheets
                    processed_dfs, rows_processed, successful = process_excel_file_all_sheets(
                        uploaded_file, 
                        st.session_state.report_type
                    )
                    
                    if processed_dfs:
                        st.session_state.processed_data = processed_dfs
                        st.session_state.processing_complete = True
                        st.session_state.total_rows = rows_processed
                        st.session_state.skus_extracted = successful
                        st.session_state.success_rate = (successful / rows_processed * 100) if rows_processed > 0 else 0
                        
                        st.balloons()
                        st.success("✅ Processing complete!")
                        
                        # Show results
                        st.markdown("### 📊 Processing Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="success-box">
                            <h4>✅ Successfully Processed</h4>
                            <p>Processed {len(processed_dfs)} sheet(s) with {rows_processed} rows needing extraction.</p>
                            <p>SKUs and Reasons have been extracted from the Description column.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Summary metrics
                            unique_skus = set()
                            unique_reasons = set()
                            
                            for df in processed_dfs:
                                col_sku = df.columns[2] if len(df.columns) > 2 else 'SKU'
                                col_reason = df.columns[3] if len(df.columns) > 3 else 'Reason'
                                
                                if col_sku in df.columns:
                                    unique_skus.update(df[col_sku].dropna().unique())
                                if col_reason in df.columns:
                                    unique_reasons.update(df[col_reason].dropna().unique())
                            
                            st.metric("Unique SKUs", len(unique_skus))
                            st.metric("Unique Reasons", len(unique_reasons))
                        
                        # Reason distribution across all sheets
                        all_reasons = []
                        for df in processed_dfs:
                            col_reason = df.columns[3] if len(df.columns) > 3 else 'Reason'
                            if col_reason in df.columns:
                                all_reasons.extend(df[col_reason].dropna().tolist())
                        
                        if all_reasons:
                            st.markdown("### 📈 Reason Distribution (All Sheets)")
                            reason_series = pd.Series(all_reasons)
                            reason_counts = reason_series.value_counts()
                            st.bar_chart(reason_counts)
                        
                        # Download section
                        st.markdown("### 💾 Download Processed File")
                        
                        # Generate filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"B2B_{st.session_state.report_type}_Report_{timestamp}.xlsx"
                        
                        # Create download button
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            workbook = writer.book
                            
                            # Write each processed sheet
                            for idx, df in enumerate(processed_dfs):
                                sheet_name = df['Sheet_Name'].iloc[0] if 'Sheet_Name' in df.columns else f'Sheet_{idx+1}'
                                
                                # Remove the Sheet_Name column before writing
                                df_to_write = df.drop(columns=['Sheet_Name']) if 'Sheet_Name' in df.columns else df
                                
                                df_to_write.to_excel(writer, index=False, sheet_name=sheet_name)
                                
                                # Add formatting
                                worksheet = writer.sheets[sheet_name]
                                
                                # Header format
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'bg_color': '#1e88e5',
                                    'font_color': 'white',
                                    'border': 1
                                })
                                
                                # Write headers with format
                                for col_num, value in enumerate(df_to_write.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                
                                # Highlight columns C and D (SKU and Reason)
                                highlight_format = workbook.add_format({
                                    'bg_color': '#E8F5E9',
                                    'border': 1
                                })
                                
                                # Apply format to columns C and D (excluding header)
                                if len(df_to_write.columns) >= 4:
                                    for row_num in range(1, len(df_to_write) + 1):
                                        worksheet.write(row_num, 2, df_to_write.iloc[row_num-1, 2], highlight_format)
                                        worksheet.write(row_num, 3, df_to_write.iloc[row_num-1, 3], highlight_format)
                                
                                # Adjust column widths
                                worksheet.set_column('A:A', 30)  # Display Name
                                worksheet.set_column('B:B', 100)  # Description
                                worksheet.set_column('C:C', 20)  # SKU
                                worksheet.set_column('D:D', 25)  # Reason
                        
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label=f"📥 Download {filename}",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ Ready for Download</h4>
                        <p>Your processed B2B {st.session_state.report_type} Report is ready with all sheets processed.</p>
                        <p>Only rows with empty SKU or Reason cells were updated.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No sheets could be processed. Please check that your file has SKU and Reason columns in positions C and D.")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            logger.error(f"File reading error: {e}")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="warning-box">
        <h4>👆 Upload your Odoo export file to begin</h4>
        <p><strong>Required Format:</strong></p>
        <ul style="text-align: left;">
            <li>Excel file (.xlsx or .xls) with one or more sheets</li>
            <li>Column C must be titled "SKU"</li>
            <li>Column D must be titled "Reason"</li>
            <li>Column B should contain the Description</li>
        </ul>
        <p style="margin-top: 1rem;">The tool will automatically process all sheets and fill empty cells in columns C & D.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
