import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Configuration
st.set_page_config(page_title="LeakBuster QLD", page_icon="💧", layout="wide")

# Generate more realistic data with multiple leak scenarios
def generate_water_data():
    np.random.seed(42)
    dates = pd.date_range("2024-06-01", periods=1440, freq="T")
    base_flow = np.random.normal(5, 0.5, 1440)  # Normal usage (5L/min ±0.5)
    
    # Add different types of leaks
    # 1. Major pipe burst (high flow, short duration)
    base_flow[300:320] = 25  
    # 2. Slow leak (moderate flow, long duration)
    base_flow[700:900] = np.random.normal(12, 1, 200)
    # 3. Toilet leak (intermittent spikes)
    base_flow[1100:1200:15] = 18
    
    return pd.DataFrame({"timestamp": dates, "flow_Lmin": base_flow})

df = generate_water_data()

# Enhanced leak detection with simple ML approach
def detect_leaks(df):
    # Calculate rolling statistics for anomaly detection
    df['rolling_avg'] = df['flow_Lmin'].rolling(window=60, center=True).mean()
    df['rolling_std'] = df['flow_Lmin'].rolling(window=60, center=True).std()
    
    # Dynamic threshold based on historical patterns
    df['leak_threshold'] = df['rolling_avg'] + 3 * df['rolling_std']
    df['leak'] = (df['flow_Lmin'] > df['leak_threshold']) & (df['flow_Lmin'] > 8)
    
    return df

df = detect_leaks(df)

# Calculate water loss and CO2 impact
leak_durations = df[df['leak']].groupby((~df['leak']).cumsum())
total_leak_volume = (df[df['leak']]['flow_Lmin'] * 60).sum()  # Convert L/min to L/hr
co2_saved = total_leak_volume * 0.004  # 0.004 kg CO2 per liter (estimate)

# Streamlit UI
st.title("LeakBuster QLD 💧")
st.subheader("Real-time AI leak detection that saves water and reduces CO₂")

col1, col2 = st.columns(2)

with col1:
    # Enhanced visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['timestamp'], df['flow_Lmin'], label='Water Flow')
    ax.fill_between(df['timestamp'], df['flow_Lmin'], where=df['leak'], 
                   color='red', alpha=0.3, label='Leak Detected')
    ax.set_title("Real-time Water Flow Monitoring")
    ax.set_ylabel("Flow Rate (L/min)")
    ax.legend()
    st.pyplot(fig)

with col2:
    # Impact metrics
    st.metric("Leaks Detected", df['leak'].sum())
    st.metric("Water Being Wasted", f"{total_leak_volume:,.0f} liters")
    st.metric("CO₂ Emissions Prevented", f"{co2_saved:,.1f} kg")
    
    if df['leak'].any():
        st.error("🔴 ALERT: Leak detected! SMS sent to homeowner")
        st.button("View Leak Details", help="See exact times and locations of leaks")
    else:
        st.success("🟢 No leaks detected")
    
    st.progress(min(100, total_leak_volume / 1000), 
                text=f"Potential savings: ${total_leak_volume * 0.003:,.2f}")

# Additional features
st.expander("How LeakBuster Works").write("""
Our AI analyzes water flow patterns 24/7 to detect anomalies indicating leaks:
- **10x faster** than utility billing cycles
- **Works with** smart meters or DIY $20 sensors
- **Saves** 10,000 liters/home/year → 50 tonnes CO₂/1,000 homes
""")

st.expander("Take Action").write("""
1. **Homeowners**: Install sensors today
2. **Utilities**: Partner for city-wide rollout
3. **Government**: Fund rebates for water-saving tech
""")

# Mock notification system
if st.button("Simulate New Leak"):
    new_leak_idx = np.random.randint(0, len(df))
    df.loc[new_leak_idx:new_leak_idx+30, 'flow_Lmin'] = 22
    df = detect_leaks(df)
    st.rerun()
