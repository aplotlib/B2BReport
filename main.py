"""
B2B Report Analyzer
Automated SKU and Reason extraction from Odoo support tickets
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
    if 'ai_handler' not in st.session_state:
        st.session_state.ai_handler = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'report_type' not in st.session_state:
        st.session_state.report_type = 'Return'

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

def process_excel_file(df, report_type):
    """Process the Excel file to extract SKU and Reason"""
    # Validate required columns
    required_columns = ['Display Name', 'Description', 'SKU', 'Reason']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return None
    
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
    
    # Process each row
    total_rows = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_df = df.copy()
    successful_extractions = 0
    
    for idx, row in df.iterrows():
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing row {idx + 1} of {total_rows}...")
        
        # Extract SKU
        if pd.isna(row['SKU']) or str(row['SKU']).strip() == '':
            sku = extract_sku_from_description(row['Description'], ai_handler)
            processed_df.at[idx, 'SKU'] = sku
            if sku:
                successful_extractions += 1
        
        # Extract Reason
        if pd.isna(row['Reason']) or str(row['Reason']).strip() == '':
            reason = extract_reason_from_description(row['Description'], ai_handler)
            processed_df.at[idx, 'Reason'] = reason
    
    progress_bar.empty()
    status_text.empty()
    
    # Add metadata
    processed_df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    processed_df['Report_Type'] = report_type
    
    return processed_df, successful_extractions

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
            
            **Step 2: Upload File**
            - File must have columns:
              - Display Name
              - Description
              - SKU (empty)
              - Reason (empty)
            
            **Step 3: Process**
            - AI will extract SKU and Reason from Description
            - Download the processed file
            
            **Expected SKU Formats:**
            - LVA1004-UPC
            - SUP1001
            - MOB2003
            - RHB3002
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
            st.metric("Total Rows", st.session_state.total_rows)
            st.metric("SKUs Extracted", st.session_state.skus_extracted)
            st.metric("Success Rate", f"{st.session_state.success_rate:.1f}%")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Odoo Export")
        st.markdown("""
        <div class="info-box">
        Upload your Excel file exported from Odoo Support Tickets. 
        The file should contain tickets filtered by "return" with empty SKU and Reason columns.
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
            df = pd.read_excel(uploaded_file)
            st.session_state.original_data = df
            
            # Display file info
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                empty_skus = df['SKU'].isna().sum() if 'SKU' in df.columns else 0
                st.metric("Empty SKUs", empty_skus)
            with col3:
                empty_reasons = df['Reason'].isna().sum() if 'Reason' in df.columns else 0
                st.metric("Empty Reasons", empty_reasons)
            
            # Preview data
            with st.expander("📋 Preview Data", expanded=True):
                st.dataframe(df.head(10))
            
            # Process button
            if st.button("🚀 Process File", type="primary", use_container_width=True):
                with st.spinner("Processing file..."):
                    processed_df, successful = process_excel_file(df, st.session_state.report_type)
                    
                    if processed_df is not None:
                        st.session_state.processed_data = processed_df
                        st.session_state.processing_complete = True
                        st.session_state.total_rows = len(processed_df)
                        st.session_state.skus_extracted = successful
                        st.session_state.success_rate = (successful / len(processed_df) * 100) if len(processed_df) > 0 else 0
                        
                        st.balloons()
                        st.success("✅ Processing complete!")
                        
                        # Show results
                        st.markdown("### 📊 Processing Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            <div class="success-box">
                            <h4>✅ Successfully Processed</h4>
                            <p>SKUs and Reasons have been extracted from the Description column.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Summary metrics
                            unique_skus = processed_df['SKU'].nunique()
                            unique_reasons = processed_df['Reason'].nunique()
                            
                            st.metric("Unique SKUs", unique_skus)
                            st.metric("Unique Reasons", unique_reasons)
                        
                        # Preview processed data
                        st.markdown("### 📋 Processed Data Preview")
                        display_cols = ['Display Name', 'SKU', 'Reason', 'Description']
                        st.dataframe(processed_df[display_cols].head(20))
                        
                        # Reason distribution
                        if 'Reason' in processed_df.columns:
                            st.markdown("### 📈 Reason Distribution")
                            reason_counts = processed_df['Reason'].value_counts()
                            st.bar_chart(reason_counts)
                        
                        # Download section
                        st.markdown("### 💾 Download Processed File")
                        
                        # Generate filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"B2B_{st.session_state.report_type}_Report_{timestamp}.xlsx"
                        
                        # Create download button
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            processed_df.to_excel(writer, index=False, sheet_name='Processed_Data')
                            
                            # Add formatting
                            workbook = writer.book
                            worksheet = writer.sheets['Processed_Data']
                            
                            # Header format
                            header_format = workbook.add_format({
                                'bold': True,
                                'bg_color': '#1e88e5',
                                'font_color': 'white',
                                'border': 1
                            })
                            
                            # Write headers with format
                            for col_num, value in enumerate(processed_df.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                            
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
                        
                        st.markdown("""
                        <div class="success-box">
                        <h4>✅ Ready for Download</h4>
                        <p>Your processed B2B {} Report is ready. Click the button above to download.</p>
                        </div>
                        """.format(st.session_state.report_type), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            logger.error(f"File reading error: {e}")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="warning-box">
        <h4>👆 Upload your Odoo export file to begin</h4>
        <p>Make sure your file has the required columns: Display Name, Description, SKU, and Reason.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
