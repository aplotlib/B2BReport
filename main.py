"""
app.py - B2B Report Analyzer (Optimized Version)
Automated SKU and Reason extraction from Odoo support tickets
Optimized for faster scanning and processing
For Vive Health Quality Management
"""

import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

# Professional CSS styling (keeping existing styles)
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
    defaults = {
        'processed_data': None,
        'original_data': None,
        'original_file': None,
        'ai_handler': None,
        'processing_complete': False,
        'report_type': 'Return',
        'total_rows': 0,
        'skus_extracted': 0,
        'success_rate': 0.0,
        'sheet_scan_results': {},  # Store quick scan results
        'processing_time': 0.0,
        'sheets_to_process': [],
        'sheets_skipped': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def quick_scan_sheet(excel_file, sheet_name):
    """Quickly scan a sheet to check if columns C and D need processing"""
    try:
        # Read only first row to get headers and a sample of data
        df_headers = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=0)
        
        # Check if we have at least 4 columns
        if len(df_headers.columns) < 4:
            return {
                'processable': False,
                'reason': 'Insufficient columns (< 4)',
                'empty_cells': 0,
                'total_rows': 0
            }
        
        # Get column names
        col_c = df_headers.columns[2]
        col_d = df_headers.columns[3]
        
        # Check if columns are SKU and Reason
        has_sku = 'sku' in str(col_c).lower()
        has_reason = 'reason' in str(col_d).lower()
        
        if not has_sku or not has_reason:
            return {
                'processable': False,
                'reason': f'Invalid headers - C: {col_c}, D: {col_d}',
                'empty_cells': 0,
                'total_rows': 0
            }
        
        # Now do a quick scan of just columns C and D
        # Use usecols to read only specific columns for speed
        df_subset = pd.read_excel(
            excel_file, 
            sheet_name=sheet_name,
            usecols=[2, 3],  # Only columns C and D
            dtype=str  # Read as strings for faster processing
        )
        
        # Count empty cells
        empty_c = df_subset.iloc[:, 0].isna().sum() + (df_subset.iloc[:, 0] == '').sum()
        empty_d = df_subset.iloc[:, 1].isna().sum() + (df_subset.iloc[:, 1] == '').sum()
        total_empty = empty_c + empty_d
        
        return {
            'processable': True,
            'reason': 'Valid structure',
            'empty_cells': total_empty,
            'empty_c': empty_c,
            'empty_d': empty_d,
            'total_rows': len(df_subset),
            'needs_processing': total_empty > 0
        }
        
    except Exception as e:
        logger.error(f"Error scanning sheet {sheet_name}: {e}")
        return {
            'processable': False,
            'reason': f'Scan error: {str(e)}',
            'empty_cells': 0,
            'total_rows': 0
        }

def quick_scan_all_sheets(excel_file):
    """Quickly scan all sheets to determine which need processing"""
    sheet_names = excel_file.sheet_names
    scan_results = {}
    
    # Use progress bar for scanning
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, sheet_name in enumerate(sheet_names):
        progress_bar.progress((idx + 1) / len(sheet_names))
        status_text.text(f"Scanning sheet: {sheet_name}")
        
        scan_results[sheet_name] = quick_scan_sheet(excel_file, sheet_name)
    
    progress_bar.empty()
    status_text.empty()
    
    return scan_results

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
    patterns = [
        r'\b(LVA\d{4}[A-Z0-9\-]*)\b',
        r'\b(SUP\d{4}[A-Z0-9\-]*)\b',
        r'\b(MOB\d{4}[A-Z0-9\-]*)\b',
        r'\b(RHB\d{4}[A-Z0-9\-]*)\b',
        r'\b([A-Z]{3}\d{4}[A-Z0-9\-]*)\b',
        r'SKU[:\s]+([A-Z0-9\-]+)',
        r'Item[:\s]+([A-Z0-9\-]+)',
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

def process_single_row(row_data, ai_handler):
    """Process a single row for SKU and Reason extraction"""
    idx, row, col_sku, col_reason, col_description = row_data
    
    sku_empty = pd.isna(row[col_sku]) or str(row[col_sku]).strip() == ''
    reason_empty = pd.isna(row[col_reason]) or str(row[col_reason]).strip() == ''
    
    result = {'index': idx, 'processed': False}
    
    if sku_empty or reason_empty:
        description = row[col_description] if col_description else ''
        
        if sku_empty:
            result['sku'] = extract_sku_from_description(description, ai_handler)
        else:
            result['sku'] = row[col_sku]
            
        if reason_empty:
            result['reason'] = extract_reason_from_description(description, ai_handler)
        else:
            result['reason'] = row[col_reason]
            
        result['processed'] = True
    else:
        result['sku'] = row[col_sku]
        result['reason'] = row[col_reason]
    
    return result

def process_sheet_optimized(df, sheet_name, report_type, ai_handler):
    """Process a single sheet with optimized extraction"""
    # Get column names
    col_display = df.columns[0] if len(df.columns) > 0 else 'Display Name'
    col_description = df.columns[1] if len(df.columns) > 1 else 'Description'
    col_sku = df.columns[2] if len(df.columns) > 2 else 'SKU'
    col_reason = df.columns[3] if len(df.columns) > 3 else 'Reason'
    
    # Find rows that need processing
    needs_processing = df[
        (df[col_sku].isna() | (df[col_sku] == '')) |
        (df[col_reason].isna() | (df[col_reason] == ''))
    ]
    
    if len(needs_processing) == 0:
        return df, 0, 0
    
    processed_df = df.copy()
    rows_processed = 0
    successful_extractions = 0
    
    # Process in batches for better performance
    batch_size = 10
    row_data_list = []
    
    for idx in needs_processing.index:
        row_data_list.append((
            idx,
            df.loc[idx],
            col_sku,
            col_reason,
            col_description
        ))
    
    # Process batches
    for i in range(0, len(row_data_list), batch_size):
        batch = row_data_list[i:i+batch_size]
        
        # Process batch (could be parallelized if AI handler supports it)
        for row_data in batch:
            result = process_single_row(row_data, ai_handler)
            
            if result['processed']:
                idx = result['index']
                processed_df.at[idx, col_sku] = result['sku']
                processed_df.at[idx, col_reason] = result['reason']
                rows_processed += 1
                
                if result['sku'] and result['sku'] != "":
                    successful_extractions += 1
    
    # Add metadata
    processed_df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    processed_df['Report_Type'] = report_type
    processed_df['Sheet_Name'] = sheet_name
    
    return processed_df, rows_processed, successful_extractions

def process_excel_file_optimized(file, report_type, sheets_to_process):
    """Process only sheets that need processing"""
    # Initialize AI handler
    ai_handler = None
    if AI_AVAILABLE and st.session_state.ai_handler is None:
        try:
            ai_handler = AIHandler()
            st.session_state.ai_handler = ai_handler
            
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
    
    # Read Excel file
    excel_file = pd.ExcelFile(file)
    
    # Process only selected sheets
    all_processed_dfs = []
    total_rows_processed = 0
    total_successful = 0
    
    progress_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        current_metrics = st.columns(4)
    
    for idx, sheet_name in enumerate(sheets_to_process):
        overall_progress.progress((idx) / len(sheets_to_process))
        status_text.text(f"Processing sheet {idx + 1} of {len(sheets_to_process)}: {sheet_name}")
        
        # Read and process sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Update metrics
        with current_metrics[0]:
            st.metric("Current Sheet", sheet_name[:20] + "..." if len(sheet_name) > 20 else sheet_name)
        with current_metrics[1]:
            st.metric("Rows", len(df))
        
        # Process sheet
        processed_df, rows_processed, successful = process_sheet_optimized(
            df, sheet_name, report_type, ai_handler
        )
        
        if processed_df is not None:
            all_processed_dfs.append(processed_df)
            total_rows_processed += rows_processed
            total_successful += successful
            
            # Update metrics
            with current_metrics[2]:
                st.metric("Processed", rows_processed)
            with current_metrics[3]:
                st.metric("SKUs Found", successful)
    
    # Final progress update
    overall_progress.progress(1.0)
    status_text.text("Processing complete!")
    
    # Clear progress container
    time.sleep(1)
    progress_container.empty()
    
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
        
        # Performance settings
        st.markdown("### ⚡ Performance Settings")
        
        batch_processing = st.checkbox(
            "Enable Batch Processing",
            value=True,
            help="Process multiple rows simultaneously for speed"
        )
        
        skip_filled = st.checkbox(
            "Quick Scan Mode",
            value=True,
            help="Quickly identify sheets that don't need processing"
        )
        
        st.markdown("---")
        
        # Help section
        with st.expander("📖 How to Use", expanded=False):
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
            
            **Step 3: Process**
            - Quick scan identifies sheets needing processing
            - AI extracts SKU and Reason from Description
            - Only empty cells are updated
            """)
        
        # Stats
        if st.session_state.processing_complete:
            st.markdown("---")
            st.markdown("### 📊 Processing Stats")
            st.metric("Total Rows Processed", st.session_state.total_rows)
            st.metric("SKUs Extracted", st.session_state.skus_extracted)
            st.metric("Success Rate", f"{st.session_state.success_rate:.1f}%")
            st.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")
    
    # Main content
    st.markdown("### 📤 Upload Odoo Export")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel file",
        type=['xlsx', 'xls'],
        help="Upload the Excel export from Odoo"
    )
    
    if uploaded_file:
        try:
            start_time = time.time()
            
            # Quick scan all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            
            with st.spinner("🔍 Quick scanning all sheets..."):
                scan_results = quick_scan_all_sheets(excel_file)
                st.session_state.sheet_scan_results = scan_results
            
            scan_time = time.time() - start_time
            
            # Display scan results
            st.success(f"✅ Quick scan complete in {scan_time:.1f} seconds")
            
            # Categorize sheets
            sheets_need_processing = []
            sheets_already_complete = []
            sheets_invalid = []
            
            for sheet_name, result in scan_results.items():
                if result['processable']:
                    if result['needs_processing']:
                        sheets_need_processing.append(sheet_name)
                    else:
                        sheets_already_complete.append(sheet_name)
                else:
                    sheets_invalid.append(sheet_name)
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📋 Total Sheets", len(scan_results))
                st.caption("In the Excel file")
            
            with col2:
                st.metric("🚀 Need Processing", len(sheets_need_processing))
                st.caption("Have empty cells in C or D")
            
            with col3:
                st.metric("✅ Already Complete", len(sheets_already_complete))
                st.caption("All cells filled")
            
            # Detailed results
            with st.expander("📊 Detailed Scan Results", expanded=True):
                # Sheets needing processing
                if sheets_need_processing:
                    st.markdown("#### 🚀 Sheets Requiring Processing")
                    for sheet in sheets_need_processing:
                        result = scan_results[sheet]
                        st.success(f"""
                        **{sheet}**: {result['empty_cells']} empty cells 
                        (C: {result['empty_c']}, D: {result['empty_d']}) 
                        in {result['total_rows']} rows
                        """)
                
                # Sheets already complete
                if sheets_already_complete:
                    st.markdown("#### ✅ Sheets Already Complete")
                    for sheet in sheets_already_complete:
                        st.info(f"**{sheet}**: All SKU and Reason cells are filled")
                
                # Invalid sheets
                if sheets_invalid:
                    st.markdown("#### ❌ Invalid Sheets")
                    for sheet in sheets_invalid:
                        result = scan_results[sheet]
                        st.error(f"**{sheet}**: {result['reason']}")
            
            # Process button
            if sheets_need_processing:
                st.markdown("---")
                
                if st.button(
                    f"🚀 Process {len(sheets_need_processing)} Sheet(s)",
                    type="primary",
                    use_container_width=True
                ):
                    process_start = time.time()
                    
                    with st.spinner(f"Processing {len(sheets_need_processing)} sheets..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Process only necessary sheets
                        st.session_state.sheets_to_process = sheets_need_processing
                        st.session_state.sheets_skipped = sheets_already_complete
                        
                        processed_dfs, rows_processed, successful = process_excel_file_optimized(
                            uploaded_file,
                            st.session_state.report_type,
                            sheets_need_processing
                        )
                        
                        st.session_state.processing_time = time.time() - process_start
                        
                        if processed_dfs:
                            st.session_state.processed_data = processed_dfs
                            st.session_state.processing_complete = True
                            st.session_state.total_rows = rows_processed
                            st.session_state.skus_extracted = successful
                            st.session_state.success_rate = (successful / rows_processed * 100) if rows_processed > 0 else 0
                            
                            st.balloons()
                            st.success(f"""
                            ✅ Processing complete in {st.session_state.processing_time:.1f} seconds!
                            - Processed {len(sheets_need_processing)} sheets
                            - Updated {rows_processed} rows
                            - Extracted {successful} SKUs
                            """)
                            
                            # Download section
                            st.markdown("### 💾 Download Processed File")
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"B2B_{st.session_state.report_type}_Report_{timestamp}.xlsx"
                            
                            # Create Excel with all sheets (processed + unmodified)
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                workbook = writer.book
                                
                                # Write processed sheets
                                processed_sheet_names = []
                                for df in processed_dfs:
                                    sheet_name = df['Sheet_Name'].iloc[0] if 'Sheet_Name' in df.columns else 'Processed'
                                    processed_sheet_names.append(sheet_name)
                                    
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
                                    
                                    for col_num, value in enumerate(df_to_write.columns.values):
                                        worksheet.write(0, col_num, value, header_format)
                                    
                                    # Highlight columns C and D
                                    highlight_format = workbook.add_format({
                                        'bg_color': '#E8F5E9',
                                        'border': 1
                                    })
                                    
                                    if len(df_to_write.columns) >= 4:
                                        for row_num in range(1, len(df_to_write) + 1):
                                            worksheet.write(row_num, 2, df_to_write.iloc[row_num-1, 2], highlight_format)
                                            worksheet.write(row_num, 3, df_to_write.iloc[row_num-1, 3], highlight_format)
                                
                                # Copy over sheets that were already complete
                                for sheet_name in sheets_already_complete:
                                    df_original = pd.read_excel(excel_file, sheet_name=sheet_name)
                                    df_original.to_excel(writer, index=False, sheet_name=sheet_name)
                            
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label=f"📥 Download {filename}",
                                data=excel_data,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            
                            # Summary
                            with st.expander("📊 Processing Summary"):
                                st.markdown(f"""
                                **Performance Metrics:**
                                - Total scan time: {scan_time:.1f}s
                                - Processing time: {st.session_state.processing_time:.1f}s
                                - Sheets processed: {len(sheets_need_processing)}
                                - Sheets skipped: {len(sheets_already_complete)}
                                - Speed: {rows_processed/st.session_state.processing_time:.0f} rows/second
                                """)
            else:
                st.info("""
                ✨ All sheets are already complete! No processing needed.
                All SKU and Reason cells (columns C and D) are already filled.
                """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {e}")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="warning-box">
        <h4>👆 Upload your Odoo export file to begin</h4>
        <p>The optimized analyzer will:</p>
        <ul style="text-align: left;">
            <li>⚡ Quickly scan all sheets to identify which need processing</li>
            <li>🎯 Skip sheets where columns C & D are already filled</li>
            <li>🚀 Process only rows with empty cells</li>
            <li>💾 Preserve all your data and formatting</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
