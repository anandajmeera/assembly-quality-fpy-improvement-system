import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Page Config
st.set_page_config(page_title="FPY Dashboard", page_icon="📈", layout="wide")

# Custom Styling (Red, Burgundy, Beige)
st.markdown("""
<style>
    .main { background-color: #F8F4EE; }
    h1, h2, h3 { color: #5B1A24; font-family: 'Arial'; }
    .stMetric { background-color: #FFFFFF; border-top: 5px solid #892131; padding: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.title("🏭 Assembly Line Quality & FPY System")
st.markdown("**(Data Simulation September 2025 - November 2025)**")

# Load Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('../data/defect_data.csv')
    except:
        return pd.read_csv('data/defect_data.csv')

df = load_data()

# Calculate Global KPIs
total_units = df['Total_Units_Produced'].sum()
total_passed = df['Units_Passed_First_Time'].sum()
total_reworked = df['Units_Reworked'].sum()
total_rejected = df['Units_Rejected'].sum()

fpy = (total_passed / total_units) * 100
rework_pct = (total_reworked / total_units) * 100
reject_pct = (total_rejected / total_units) * 100

st.markdown("### 📊 Key Performance Indicators (KPIs)")
m1, m2, m3 = st.columns(3)
m1.metric("First Pass Yield (FPY)", f"{fpy:.2f}%")
m2.metric("Rework %", f"{rework_pct:.2f}%")
m3.metric("Rejection %", f"{reject_pct:.2f}%")

st.markdown("---")

col1, col2 = st.columns(2)

# Pareto Chart
with col1:
    st.markdown("### 🛑 Pareto Analysis (Top Defect Causes)")
    pareto_data = df.groupby('Root_Cause')['Total_Units_Produced'].count().reset_index()
    pareto_data.columns = ['Root Cause', 'Count']
    pareto_data = pareto_data.sort_values(by='Count', ascending=False)
    pareto_data['Cumulative Percentage'] = pareto_data['Count'].cumsum() / pareto_data['Count'].sum() * 100

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(x=pareto_data['Root Cause'], y=pareto_data['Count'], name="Defect Count", marker_color='#892131'))
    fig_pareto.add_trace(go.Scatter(x=pareto_data['Root Cause'], y=pareto_data['Cumulative Percentage'], name="Cumulative %", yaxis="y2", mode="lines+markers", line_color='#E4A596'))
    
    fig_pareto.update_layout(
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

# Station-wise FPY
with col2:
    st.markdown("### ⚙️ Station-Wise FPY Comparison")
    station_data = df.groupby('Assembly_Station')[['Total_Units_Produced', 'Units_Passed_First_Time']].sum().reset_index()
    station_data['FPY'] = (station_data['Units_Passed_First_Time'] / station_data['Total_Units_Produced']) * 100
    station_data = station_data.sort_values(by='FPY', ascending=True)

    fig_station = px.bar(station_data, x="FPY", y="Assembly_Station", orientation='h', color="FPY", color_continuous_scale=['#892131', '#5B1A24', '#D06060'], title="")
    fig_station.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_station, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)

# FPY Trend over time
with col3:
    st.markdown("### 📈 FPY Trend Over Time")
    df['defect_date'] = pd.to_datetime(df['defect_date'])
    trend_data = df.groupby(df['defect_date'].dt.to_period('W'))[['Total_Units_Produced', 'Units_Passed_First_Time']].sum().reset_index()
    trend_data['defect_date'] = trend_data['defect_date'].dt.to_timestamp()
    trend_data['FPY'] = (trend_data['Units_Passed_First_Time'] / trend_data['Total_Units_Produced']) * 100
    
    fig_trend = px.line(trend_data, x='defect_date', y='FPY', markers=True, color_discrete_sequence=['#892131'])
    fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Date", yaxis_title="First Pass Yield %")
    st.plotly_chart(fig_trend, use_container_width=True)

# Predictive Risk Model
with col4:
    st.markdown("### 🤖 Average Defect Risk Probability by Shift (AI Model)")
    risk_data = df.groupby('Shift')['Defect_Risk_Probability'].mean().reset_index()
    risk_data['Shift'] = "Shift " + risk_data['Shift'].astype(str)
    
    fig_risk = px.pie(risk_data, values='Defect_Risk_Probability', names='Shift', color_discrete_sequence=['#5B1A24', '#892131', '#D06060'])
    fig_risk.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_risk, use_container_width=True)
