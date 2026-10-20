import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & TITLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GoldLens - Mech Dynasty",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 GoldLens: Multi-Contract Gold Relative Value & Arbitrage Engine")
st.markdown("""
**Team Mech Dynasty** (Leader: Koyel Patra | Members: Aritra Ghosh, Sibam Mondal)  
*Same gold, different price: measure it, test it, trust it.* This prototype normalizes MCX gold settlement data into **₹/gram (100% fine gold equivalent)**, models relative pricing gaps using rolling Z-scores, and validates performance after real transaction costs[span_3](start_span)[span_3](end_span)[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span)[span_6](start_span)[span_6](end_span).
""")

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS & HYPERPARAMETERS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Strategy Hyperparameters")

window_size = st.sidebar.slider("Rolling Window Size (Days)", min_value=3, max_value=20, value=5)
z_entry = st.sidebar.slider("Z-Score Entry Threshold", min_value=1.0, max_value=2.5, value=1.5, step=0.1)
z_exit = st.sidebar.slider("Z-Score Exit Threshold", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
cost_bps = st.sidebar.number_input("Transaction Cost (bps)", min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 10000.0

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("Upload MCX BhavCopy CSV Files", type=["csv"], accept_multiple_files=True)

# Contract Specifications
UNITS = {"GOLDM": 10.0, "GOLDTEN": 10.0, "GOLDGUINEA": 8.0, "GOLDPETAL": 1.0}
PURITY = {"GOLDM": 995.0, "GOLDTEN": 999.0, "GOLDGUINEA": 999.0, "GOLDPETAL": 999.0}

# -----------------------------------------------------------------------------
# 3. DATA PROCESSING & NORMALIZATION PIPELINE
# -----------------------------------------------------------------------------
if uploaded_files:
    raw_data = []
    for file in uploaded_files:
        df_file = pd.read_csv(file)
        df_file.columns = df_file.columns.str.strip()
        raw_data.append(df_file)
    
    df_raw = pd.concat(raw_data, ignore_index=True)
    df_raw['Symbol'] = df_raw['Symbol'].astype(str).str.strip().str.upper()
    df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.strftime('%Y-%m-%d')
    
    # Filter and Normalize to ₹/g 100% Fine Gold
    df_filtered = df_raw[df_raw['Symbol'].isin(UNITS.keys())].copy()
    df_filtered['Unit'] = df_filtered['Symbol'].map(UNITS)
    df_filtered['Purity'] = df_filtered['Symbol'].map(PURITY)
    df_filtered['Norm_Price'] = (df_filtered['Close'] / df_filtered['Unit']) * (1000.0 / df_filtered['Purity'])
    
    # Pivot Dataset Side-by-Side by Date
    df_pivot = df_filtered.pivot_table(index='Date', columns='Symbol', values='Norm_Price', aggfunc='mean').sort_index()
    
    if 'GOLDM' in df_pivot.columns and 'GOLDTEN' in df_pivot.columns:
        df_pivot = df_pivot.dropna(subset=['GOLDM', 'GOLDTEN']).copy()
        
        # Calculate Spread and Z-Score
        df_pivot['Spread'] = df_pivot['GOLDTEN'] - df_pivot['GOLDM']
        df_pivot['Mean'] = df_pivot['Spread'].rolling(window=window_size).mean()
        df_pivot['Std'] = df_pivot['Spread'].rolling(window=window_size).std().replace(0, 0.0001)
        df_pivot['Z_Score'] = (df_pivot['Spread'] - df_pivot['Mean']) / df_pivot['Std']
        
        # -----------------------------------------------------------------------------
        # 4. WALK-FORWARD BACKTEST & COST FRICTION ENGINE
        # -----------------------------------------------------------------------------
        positions = []
        current_pos = 0
        
        for z in df_pivot['Z_Score']:
            if pd.isna(z):
                positions.append(0)
            elif z > z_entry:
                current_pos = -1  # Short Spread (Sell GOLDTEN / Buy GOLDM)
                positions.append(-1)
            elif z < -z_entry:
                current_pos = 1   # Long Spread (Buy GOLDTEN / Sell GOLDM)
                positions.append(1)
            elif abs(z) < z_exit:
                current_pos = 0   # Flat / Exit
                positions.append(0)
            else:
                positions.append(current_pos)
                
        df_pivot['Position'] = positions
        df_pivot['Spread_Change'] = df_pivot['Spread'].diff().fillna(0)
        df_pivot['Gross_PnL'] = df_pivot['Position'].shift(1).fillna(0) * df_pivot['Spread_Change']
        
        # Deduct transaction fees on position change
        df_pivot['Pos_Change'] = df_pivot['Position'].diff().fillna(0)
        df_pivot['Cost'] = np.where(df_pivot['Pos_Change'] != 0, df_pivot['GOLDM'] * cost_bps, 0)
        df_pivot['Net_PnL'] = df_pivot['Gross_PnL'] - df_pivot['Cost']
        
        df_pivot['Cum_Gross'] = df_pivot['Gross_PnL'].cumsum()
        df_pivot['Cum_Net'] = df_pivot['Net_PnL'].cumsum()
        
        # Summary Metrics
        tot_gross = df_pivot['Gross_PnL'].sum()
        tot_cost = df_pivot['Cost'].sum()
        tot_net = df_pivot['Net_PnL'].sum()
        gold_beta = ((df_pivot['GOLDM'].iloc[-1] - df_pivot['GOLDM'].iloc[0]) / df_pivot['GOLDM'].iloc[0]) * 100
        
        # -----------------------------------------------------------------------------
        # 5. DASHBOARD METRICS DISPLAY
        # -----------------------------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Strategy Alpha", f"₹{tot_gross:.2f} /g")
        col2.metric("Transaction Drag", f"₹{tot_cost:.2f} /g")
        col3.metric("Net Strategy Alpha", f"₹{tot_net:.2f} /g", delta=f"{tot_net:.2f}")
        col4.metric("Underlying Gold Shift (Beta)", f"{gold_beta:+.2f}%")
        
        st.markdown("---")
        
        # -----------------------------------------------------------------------------
        # 6. PLOTLY CHARTS
        # -----------------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["Cumulative Performance", "Normalized Prices", "Spread & Z-Score"])
        
        with tab1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Cum_Gross'], name='Gross PnL (₹/g)', line=dict(color='gray', dash='dash')))
            fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Cum_Net'], name='Net PnL (₹/g)', line=dict(color='green', width=2)))
            fig1.update_layout(title="Cumulative Strategy Returns (Gross vs Net Alpha)", xaxis_title="Date", yaxis_title="PnL (₹/g)")
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            fig2 = px.line(df_pivot, y=[col for col in ['GOLDM', 'GOLDTEN', 'GOLDGUINEA', 'GOLDPETAL'] if col in df_pivot.columns],
                           title="Normalized Contract Prices (100% Fine Gold Equivalent in ₹/g)")
            fig2.update_layout(xaxis_title="Date", yaxis_title="Normalized Price (₹/g)")
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Z_Score'], name='Z-Score', line=dict(color='purple')))
            fig3.add_hline(y=z_entry, line_dash="dash", line_color="red", annotation_text="Sell Threshold (+Z)")
            fig3.add_hline(y=-z_entry, line_dash="dash", line_color="green", annotation_text="Buy Threshold (-Z)")
            fig3.update_layout(title="Rolling Z-Score & Trading Thresholds", xaxis_title="Date", yaxis_title="Z-Score")
            st.plotly_chart(fig3, use_container_width=True)
            
        # Data Export Button
        st.download_button(
            label="📥 Download Backtest Results CSV",
            data=df_pivot.to_csv().encode('utf-8'),
            file_name="GoldLens_Backtest_Results.csv",
            mime="text/csv"
        )
    else:
        st.warning("Uploaded files must contain both 'GOLDM' and 'GOLDTEN' contract symbols for spread analysis.")
else:
    st.info("👈 Please upload your MCX BhavCopy CSV files using the sidebar file uploader to launch the dashboard.")
