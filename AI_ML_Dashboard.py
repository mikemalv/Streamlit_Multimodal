import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import time
import numpy as np
from snowflake.snowpark.context import get_active_session

# Configure the page
st.set_page_config(
    page_title="AI/ML Cost Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Get the active Snowflake session
session = get_active_session()

# Add title and description
st.title("🤖 AI/ML Cost Monitoring Dashboard")
st.markdown("Monitor your Snowflake AI/ML usage, tokens, costs, and performance across Cortex services.")

# Sidebar filters (shared across tabs)
st.sidebar.header("Filters")
date_range = st.sidebar.selectbox(
    "Date Range",
    ["Last 1 day", "Last 7 days", "Last 14 days", "Last 30 days", "Last 60 days", "Last 90 days", "Last 180 days", "Last 365 days"],
    index=0  # Default to 14 days for better overview
)

# Price per credit selector
st.sidebar.markdown("---")
st.sidebar.header("Pricing")
price_option = st.sidebar.selectbox(
    "Price per Credit",
    ["$2.00", "$3.00", "$4.00", "Custom"],
    index=0
)

if price_option == "Custom":
    price_per_credit = st.sidebar.number_input(
        "Enter custom price per credit ($)",
        min_value=0.01,
        max_value=100.00,
        value=2.00,
        step=0.01,
        format="%.2f"
    )
else:
    price_per_credit = float(price_option.replace("$", ""))

# Add a refresh button in the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Convert date range to days
days_mapping = {
    "Last 1 day": 1,
    "Last 7 days": 7,
    "Last 14 days": 14,
    "Last 30 days": 30,
    "Last 60 days": 60,
    "Last 90 days": 90,
    "Last 180 days": 180,
    "Last 365 days": 365
}
days_to_fetch = days_mapping[date_range]

# Helper function to check table schema with proper error handling
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_table_columns(table_name):
    """Get available columns for a table using collect() method"""
    try:
        query = f"DESCRIBE TABLE {table_name}"
        rows = session.sql(query).collect()
        return [row['name'] for row in rows] if rows else []
    except Exception as e:
        return []

@st.cache_data(ttl=86400)
def test_table_access(table_name):
    """Test if we can access a table and return basic info about it"""
    try:
        # Try a simple SELECT with LIMIT 1 to test access
        query = f"SELECT * FROM {table_name} LIMIT 1"
        result = session.sql(query).collect()
        if result:
            # Get column names from the first row
            return list(result[0].as_dict().keys())
        else:
            return []
    except Exception as e:
        return []

# Test table access and get schemas
st.info("🔍 Detecting available AI/ML data sources...")

# Test each table
table_info = {
    'cortex_functions': {
        'table': 'SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY',
        'columns': test_table_access('SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY'),
        'available': False
    },
    'query_usage': {
        'table': 'SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY', 
        'columns': test_table_access('SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY'),
        'available': False
    },
    'document_ai': {
        'table': 'SNOWFLAKE.ACCOUNT_USAGE.DOCUMENT_AI_USAGE_HISTORY',
        'columns': test_table_access('SNOWFLAKE.ACCOUNT_USAGE.DOCUMENT_AI_USAGE_HISTORY'),
        'available': False
    },
    'search_serving': {
        'table': 'SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY',
        'columns': test_table_access('SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY'),
        'available': False
    }
}

# Update availability
for key, info in table_info.items():
    info['available'] = len(info['columns']) > 0

# Show available data sources
available_sources = [key for key, info in table_info.items() if info['available']]
#if available_sources:
#    st.success(f"✅ **Available data sources:** {', '.join(available_sources)}")
#else:
#    st.error("❌ No AI/ML usage tables are accessible. Please check your account permissions.")

# Display detected schemas for debugging
with st.expander("🔧 Debug: Detected Table Schemas"):
    for key, info in table_info.items():
        if info['available']:
            st.write(f"**{key}:** {info['columns']}")
        else:
            st.write(f"**{key}:** Not accessible")

def create_multiselect_with_all(label, options, key_prefix):
    """Create multiselect with Select All/Deselect All functionality"""
    if not options:
        st.info(f"No {label.lower()} available")
        return []
        
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        # Initialize session state for this multiselect if it doesn't exist
        if f"{key_prefix}_selected" not in st.session_state:
            st.session_state[f"{key_prefix}_selected"] = options
        
        selected = st.multiselect(
            label,
            options,
            default=st.session_state[f"{key_prefix}_selected"],
            key=f"{key_prefix}_multiselect"
        )
    
    with col2:
        if st.button("Select All", key=f"{key_prefix}_select_all"):
            st.session_state[f"{key_prefix}_selected"] = options
            st.rerun()
    
    with col3:
        if st.button("Deselect All", key=f"{key_prefix}_deselect_all"):
            st.session_state[f"{key_prefix}_selected"] = []
            st.rerun()
    
    # Update session state
    st.session_state[f"{key_prefix}_selected"] = selected
    return selected

# Robust data fetching functions
@st.cache_data(ttl=3600)
def get_cortex_functions_data(days):
    """Fetch Cortex Functions data with dynamic column detection"""
    if 'cortex_functions' not in table_info or not table_info['cortex_functions']['available']:
        return pd.DataFrame()
    
    try:
        columns = table_info['cortex_functions']['columns']
        table = table_info['cortex_functions']['table']
        
        # Find the correct column names (they might vary between Snowflake versions)
        start_time_col = None
        for col in columns:
            if 'START_TIME' in col.upper() or 'TIME' in col.upper():
                start_time_col = col
                break
        
        if not start_time_col:
            st.warning("Could not find timestamp column in CORTEX_FUNCTIONS_USAGE_HISTORY")
            return pd.DataFrame()
        
        # Build query with available columns
        select_parts = [
            f"DATE_TRUNC('DAY', {start_time_col}) AS USAGE_DATE"
        ]
        
        # Add hour info for recent data
        if days <= 7:
            select_parts.append(f"DATE_TRUNC('HOUR', {start_time_col}) AS USAGE_HOUR")
        
        # Required columns
        required_cols = ['FUNCTION_NAME', 'MODEL_NAME', 'TOKENS', 'TOKEN_CREDITS']
        available_required = [col for col in required_cols if col in columns]
        
        if len(available_required) < 3:  # Need at least function, tokens, credits
            st.warning(f"Missing required columns in CORTEX_FUNCTIONS_USAGE_HISTORY. Available: {columns}")
            return pd.DataFrame()
        
        select_parts.extend(available_required)
        
        # Optional columns
        optional_cols = ['USER_NAME', 'ROLE_NAME', 'WAREHOUSE_NAME']
        for col in optional_cols:
            if col in columns:
                select_parts.append(col)
        
        # Aggregation columns
        agg_parts = [
            "COUNT(*) AS TOTAL_REQUESTS"
        ]
        
        if 'TOKENS' in columns:
            agg_parts.append("SUM(TOKENS) AS TOTAL_TOKENS")
        if 'TOKEN_CREDITS' in columns:
            agg_parts.append("SUM(TOKEN_CREDITS) AS TOTAL_CREDITS")
        
        # Build GROUP BY
        group_by = ["USAGE_DATE"] + available_required
        if days <= 7:
            group_by.insert(1, "USAGE_HOUR")
        
        # Optional grouping columns
        for col in optional_cols:
            if col in columns:
                group_by.append(col)
        
        # Time filter
        if days == 1:
            time_filter = f"{start_time_col} >= DATE_TRUNC('DAY', CURRENT_TIMESTAMP)"
        else:
            time_filter = f"{start_time_col} >= DATEADD(day, -{days}, CURRENT_TIMESTAMP)"
        
        # Final query
        query = f"""
        SELECT {', '.join(select_parts + agg_parts)}
        FROM {table}
        WHERE {time_filter}
        GROUP BY {', '.join(group_by)}
        ORDER BY USAGE_DATE DESC, TOTAL_CREDITS DESC
        LIMIT 5000
        """
        
        return session.sql(query).to_pandas()
        
    except Exception as e:
        st.error(f"Error fetching Cortex Functions data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_simple_usage_data():
    """Fallback to original simple query if advanced detection fails"""
    try:
        if days_to_fetch == 1:
            query = f"""
            SELECT
              DATE_TRUNC('DAY', START_TIME) AS USAGE_DATE,
              FUNCTION_NAME,
              MODEL_NAME,
              SUM(TOKENS) AS TOTAL_TOKENS,
              SUM(TOKEN_CREDITS) AS TOTAL_CREDITS,
              COUNT(*) AS TOTAL_REQUESTS
            FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
            WHERE START_TIME >= DATE_TRUNC('DAY', CURRENT_TIMESTAMP)
            GROUP BY USAGE_DATE, FUNCTION_NAME, MODEL_NAME
            ORDER BY USAGE_DATE DESC, TOTAL_CREDITS DESC
            LIMIT 5000
            """
        else:
            query = f"""
            SELECT
              DATE_TRUNC('DAY', START_TIME) AS USAGE_DATE,
              FUNCTION_NAME,
              MODEL_NAME,
              SUM(TOKENS) AS TOTAL_TOKENS,
              SUM(TOKEN_CREDITS) AS TOTAL_CREDITS,
              COUNT(*) AS TOTAL_REQUESTS
            FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
            WHERE START_TIME >= DATEADD(day, -{days_to_fetch}, CURRENT_TIMESTAMP)
            GROUP BY USAGE_DATE, FUNCTION_NAME, MODEL_NAME
            ORDER BY USAGE_DATE DESC, TOTAL_CREDITS DESC
            LIMIT 5000
            """
        
        return session.sql(query).to_pandas()
        
    except Exception as e:
        st.error(f"Error with simple query: {str(e)}")
        return pd.DataFrame()

# Helper functions
def format_currency(value):
    return f"${value:,.2f}"

def format_number(value):
    if pd.isna(value):
        return "0"
    return f"{value:,.0f}"

def safe_get_column_values(df, column_name):
    """Safely get unique values from a column"""
    if df.empty or column_name not in df.columns:
        return []
    return df[column_name].dropna().unique().tolist()

# Load data with progress tracking
with st.spinner("Loading AI/ML data..."):
    progress_bar = st.progress(0)
    
    # Try advanced detection first, fallback to simple query
    df_functions = get_cortex_functions_data(days_to_fetch)
    
    if df_functions.empty:
        st.info("Advanced detection failed, trying simple approach...")
        df_functions = get_simple_usage_data()
    
    progress_bar.progress(100)
    progress_bar.empty()

# Check if we have any data
if df_functions.empty:
    st.error(f"""
    **No AI/ML usage data found for {date_range}.**
    
    **Possible reasons:**
    - No Cortex AI/ML services have been used in this time period
    - Account doesn't have access to ACCOUNT_USAGE views
    - Data retention period has been exceeded
    - Tables may not exist in your Snowflake edition
    
    **Troubleshooting:**
    1. Try a longer date range (Last 30 days)
    2. Check with your Snowflake administrator about ACCOUNT_USAGE permissions
    3. Verify that Cortex functions have been used recently
    4. Check if your Snowflake edition supports these usage tables
    """)
    st.stop()

# Add TOTAL_SPEND column
if 'TOTAL_CREDITS' in df_functions.columns:
    df_functions['TOTAL_SPEND'] = df_functions['TOTAL_CREDITS'] * price_per_credit

# Convert date column
if 'USAGE_DATE' in df_functions.columns:
    df_functions['USAGE_DATE'] = pd.to_datetime(df_functions['USAGE_DATE'])

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "📋 Raw Data"])

with tab1:
    st.header("💰 Cost Overview & Key Metrics")
    
    # Calculate metrics
    total_credits = df_functions['TOTAL_CREDITS'].sum() if 'TOTAL_CREDITS' in df_functions.columns else 0
    total_spend = total_credits * price_per_credit
    total_tokens = df_functions['TOTAL_TOKENS'].sum() if 'TOTAL_TOKENS' in df_functions.columns else 0
    total_requests = df_functions['TOTAL_REQUESTS'].sum() if 'TOTAL_REQUESTS' in df_functions.columns else 0
    
    # Calculate daily averages
    unique_dates = df_functions['USAGE_DATE'].dt.date.nunique() if 'USAGE_DATE' in df_functions.columns else 1
    avg_daily_credits = total_credits / unique_dates if unique_dates > 0 else 0
    avg_daily_spend = total_spend / unique_dates if unique_dates > 0 else 0
    
    # Top metrics cards
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            "💰 Total Spend",
            format_currency(total_spend),
            delta=f"${avg_daily_spend:.2f}/day avg"
        )
    
    with cols[1]:
        st.metric(
            "⚡ Total Credits",
            f"{total_credits:.1f}",
            delta=f"{avg_daily_credits:.1f}/day avg"
        )
    
    with cols[2]:
        st.metric(
            "🔢 Total Tokens", 
            format_number(total_tokens)
        )
    
    with cols[3]:
        st.metric(
            "📊 Total Requests",
            format_number(total_requests)
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'USAGE_DATE' in df_functions.columns and 'TOTAL_SPEND' in df_functions.columns:
            st.subheader("Daily Spend Trend")
            daily_spend = df_functions.groupby(df_functions['USAGE_DATE'].dt.date)['TOTAL_SPEND'].sum().reset_index()
            
            chart = alt.Chart(daily_spend).mark_line(point=True, color='#1f77b4').encode(
                x=alt.X('USAGE_DATE:T', title='Date'),
                y=alt.Y('TOTAL_SPEND:Q', title='Cost ($)', scale=alt.Scale(zero=True)),
                tooltip=['USAGE_DATE:T', 'TOTAL_SPEND:Q']
            ).properties(
                title="Daily AI/ML Costs",
                height=300
            )
            
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        if 'MODEL_NAME' in df_functions.columns and 'TOTAL_CREDITS' in df_functions.columns:
            st.subheader("Model Usage Distribution")
            model_usage = df_functions.groupby('MODEL_NAME')['TOTAL_CREDITS'].sum().reset_index()
            
            chart = alt.Chart(model_usage).mark_arc().encode(
                theta=alt.Theta('TOTAL_CREDITS:Q'),
                color=alt.Color('MODEL_NAME:N'),
                tooltip=['MODEL_NAME:N', 'TOTAL_CREDITS:Q']
            ).properties(
                title="Credits by Model",
                height=300
            )
            
            st.altair_chart(chart, use_container_width=True)

with tab2:
    st.header("🔍 Detailed Analysis")
    
    if 'FUNCTION_NAME' in df_functions.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Function Performance")
            func_perf = df_functions.groupby('FUNCTION_NAME').agg({
                'TOTAL_CREDITS': 'sum',
                'TOTAL_REQUESTS': 'sum',
                'TOTAL_TOKENS': 'sum' if 'TOTAL_TOKENS' in df_functions.columns else 'count'
            }).reset_index()
            
            chart = alt.Chart(func_perf).mark_bar().encode(
                x=alt.X('TOTAL_CREDITS:Q', title='Total Credits'),
                y=alt.Y('FUNCTION_NAME:N', title='Function', sort='-x'),
                color=alt.Color('TOTAL_CREDITS:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['FUNCTION_NAME:N', 'TOTAL_CREDITS:Q', 'TOTAL_REQUESTS:Q']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            if 'TOTAL_TOKENS' in df_functions.columns:
                st.subheader("Token Efficiency")
                model_eff = df_functions.groupby('MODEL_NAME').agg({
                    'TOTAL_TOKENS': 'sum',
                    'TOTAL_CREDITS': 'sum'
                }).reset_index()
                model_eff['TOKENS_PER_CREDIT'] = model_eff['TOTAL_TOKENS'] / model_eff['TOTAL_CREDITS']
                model_eff = model_eff.sort_values('TOKENS_PER_CREDIT', ascending=False)
                
                chart = alt.Chart(model_eff).mark_bar().encode(
                    x=alt.X('TOKENS_PER_CREDIT:Q', title='Tokens per Credit'),
                    y=alt.Y('MODEL_NAME:N', title='Model', sort='-x'),
                    color=alt.Color('TOKENS_PER_CREDIT:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['MODEL_NAME:N', 'TOKENS_PER_CREDIT:Q', 'TOTAL_TOKENS:Q', 'TOTAL_CREDITS:Q']
                ).properties(height=400)
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Token information not available in this dataset")

with tab3:
    st.header("📋 Raw Data & Export")
    
    # Enhanced filters with select all functionality
    if not df_functions.empty:
        all_functions = safe_get_column_values(df_functions, 'FUNCTION_NAME')
        all_models = safe_get_column_values(df_functions, 'MODEL_NAME')
        all_users = safe_get_column_values(df_functions, 'USER_NAME')
        
        if all_functions:
            selected_functions = create_multiselect_with_all("Filter by Function", all_functions, "functions")
            function_filter = df_functions['FUNCTION_NAME'].isin(selected_functions) if selected_functions else True
        else:
            function_filter = True
            
        if all_models:
            selected_models = create_multiselect_with_all("Filter by Model", all_models, "models")
            model_filter = df_functions['MODEL_NAME'].isin(selected_models) if selected_models else True
        else:
            model_filter = True
            
        if all_users:
            selected_users = create_multiselect_with_all("Filter by User", all_users, "users")
            user_filter = df_functions['USER_NAME'].isin(selected_users) if selected_users else True
        else:
            user_filter = True
        
        # Apply filters
        filtered_df = df_functions[function_filter & model_filter & user_filter]
        
        # Format display
        display_df = filtered_df.copy()
        if 'TOTAL_SPEND' in display_df.columns:
            display_df['TOTAL_SPEND'] = display_df['TOTAL_SPEND'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered Data as CSV",
                csv,
                "ai_ml_usage_data.csv",
                "text/csv"
            )
    else:
        st.info("No data available for export")

# Footer with comprehensive information
st.markdown("---")
st.markdown(f"""
### 📊 Dashboard Information:
- **Date Range**: {date_range} {"(today since midnight)" if days_to_fetch == 1 else ""}
- **Current Pricing**: ${price_per_credit:.2f} per credit
- **Data Source**: SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
- **Last Refreshed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Records Shown**: {len(df_functions):,} (limited to 5,000 for performance)

### 🔧 Available Columns in Dataset: 
{', '.join(df_functions.columns.tolist()) if not df_functions.empty else 'No data available'}

### 🚨 Troubleshooting:
- **No data showing?** Try a longer date range or check if Cortex functions have been used
- **Permission errors?** Contact your Snowflake admin for ACCOUNT_USAGE access
- **Missing columns?** Different Snowflake editions may have different table schemas
- **Need more data sources?** Advanced features require additional table access permissions
""")