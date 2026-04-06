import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AQI Vision", page_icon="🌤️", layout="wide")
st.title("🌤️ AQI Vision: Intelligent Air Quality Prediction")


# The Sidebar Uploader
st.sidebar.header("📁 Step 1: Upload Dataset")
st.sidebar.write("Upload your CSV here to unlock the ML Predictor and Graphs.")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

# We create the Tabs FIRST so they are always visible!
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🌍 LIVE Hyderabad AQI", "🎯 Predict AQI (Requires CSV)", "📊 Data Graphs (Requires CSV)"])

# ==========================================
# --- TAB 1: HOME (Always works) ---
# ==========================================
with tab1:
    st.header("Welcome to AQI Vision")
    st.write("Check the live air quality in Hyderabad instantly, or upload a dataset to train your own prediction model!")
    st.image("https://images.unsplash.com/photo-1534274988757-a28bf1a57c17?w=800")

# ==========================================
# --- TAB 2: LIVE DASHBOARD (Always works, NO CSV NEEDED!) ---
# ==========================================
with tab2:
    st.header("📍 Live Air Quality in Hyderabad")
    st.write("This data is fetched **live** from the internet right now. No dataset required!")

    if st.button("🔄 Check Live Data Now"):
        try:
            # --- Fetch CURRENT readings ---
            current_url = (
                "https://air-quality-api.open-meteo.com/v1/air-quality"
                "?latitude=17.3850&longitude=78.4867"
                "&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,us_aqi"
            )
            current_resp = requests.get(current_url).json()

            live_pm25 = current_resp['current']['pm2_5']
            live_pm10 = current_resp['current']['pm10']
            live_no2  = current_resp['current']['nitrogen_dioxide']
            live_co   = current_resp['current']['carbon_monoxide']
            live_aqi  = current_resp['current']['us_aqi']

            # --- Display metric cards ---
            st.write("### Current Pollutant Levels:")
            live_col1, live_col2, live_col3, live_col4 = st.columns(4)
            live_col1.metric("Live PM2.5", live_pm25)
            live_col2.metric("Live PM10",  live_pm10)
            live_col3.metric("Live NO2",   live_no2)
            live_col4.metric("Live CO",    live_co)

            st.markdown("---")
            st.write("### 🚨 Live Health Status:")
            st.metric("Live AQI Score", live_aqi)

            if live_aqi <= 50:
                st.success("🟢 **Current Status: GOOD** \n\n*Caution:* None! It is a beautiful day, go out and exercise safely.")
                st.balloons()
            elif live_aqi <= 100:
                st.info("🟡 **Current Status: MODERATE** \n\n*Caution:* Air quality is acceptable. Sensitive people should reduce outdoor activity.")
            elif live_aqi <= 150:
                st.warning("🟠 **Current Status: UNHEALTHY FOR SENSITIVE GROUPS** \n\n*Warning:* People with asthma, children, and the elderly should reduce outdoor activities.")
            else:
                st.error("🔴 **Current Status: SEVERE / HAZARDOUS** \n\n*EMERGENCY WARNING:* Everyone should avoid outdoor activities. Wear a mask!")

            st.markdown("---")

            # --- Fetch HOURLY data for the past 3 days for charts ---
            hourly_url = (
                "https://air-quality-api.open-meteo.com/v1/air-quality"
                "?latitude=17.3850&longitude=78.4867"
                "&hourly=pm2_5,pm10,nitrogen_dioxide,us_aqi"
                "&past_days=3"
            )
            hourly_resp = requests.get(hourly_url).json()

            hourly_df = pd.DataFrame({
                "Time":  pd.to_datetime(hourly_resp['hourly']['time']),
                "PM2.5": hourly_resp['hourly']['pm2_5'],
                "PM10":  hourly_resp['hourly']['pm10'],
                "NO2":   hourly_resp['hourly']['nitrogen_dioxide'],
                "AQI":   hourly_resp['hourly']['us_aqi'],
            }).dropna()

            # ---- LINE CHART: PM2.5, PM10, NO2 over time ----
            st.write("### 📈 Pollutant Trends — Last 3 Days (Line Chart)")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=hourly_df["Time"], y=hourly_df["PM2.5"],
                                          mode='lines', name='PM2.5',
                                          line=dict(color='#FF6B6B', width=2)))
            fig_line.add_trace(go.Scatter(x=hourly_df["Time"], y=hourly_df["PM10"],
                                          mode='lines', name='PM10',
                                          line=dict(color='#4ECDC4', width=2)))
            fig_line.add_trace(go.Scatter(x=hourly_df["Time"], y=hourly_df["NO2"],
                                          mode='lines', name='NO2',
                                          line=dict(color='#FFE66D', width=2)))
            fig_line.update_layout(
                xaxis_title="Time",
                yaxis_title="Concentration (µg/m³)",
                legend_title="Pollutant",
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # ---- SCATTER PLOT: PM2.5 vs AQI ----
            st.write("### 🔵 PM2.5 vs AQI — Scatter Plot")
            fig_scatter = px.scatter(
                hourly_df,
                x="PM2.5",
                y="AQI",
                color="AQI",
                color_continuous_scale="RdYlGn_r",
                size="PM10",
                hover_data=["Time", "NO2"],
                labels={"PM2.5": "PM2.5 (µg/m³)", "AQI": "AQI Score"},
                title="How PM2.5 levels relate to AQI (bubble size = PM10)"
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Could not fetch live data. Please check your internet connection. ({e})")

# ==========================================
# --- TAB 3 & 4: THE ML STUFF (Requires the CSV) ---
# ==========================================

# Cache the trained model so it doesn't retrain on every slider interaction
@st.cache_resource
def train_model(file_data):
    data = pd.read_csv(file_data)
    features = ['PM2.5', 'PM10', 'NO2', 'CO']
    target = 'AQI'
    clean_data = data.dropna(subset=features + [target])
    X = clean_data[features]
    y = clean_data[target]
    model = RandomForestRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model, data, clean_data

if uploaded_file is not None:
    st.sidebar.success(f"✅ '{uploaded_file.name}' uploaded successfully!")
    model, data, clean_data = train_model(uploaded_file)

    # Fill Tab 3
    with tab3:
        st.header("Enter Pollution Values Manually")
        col1, col2 = st.columns(2)
        with col1:
            pm25 = st.slider("PM2.5", 0.0, 500.0, 50.0)
            pm10 = st.slider("PM10", 0.0, 500.0, 100.0)
        with col2:
            no2 = st.slider("NO2", 0.0, 200.0, 20.0)
            co = st.slider("CO", 0.0, 50.0, 1.0)

        if st.button("Predict AQI"):
            input_data = [[pm25, pm10, no2, co]]
            prediction = int(model.predict(input_data)[0])

            # FIX 1: st.columns() must always receive a number argument
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted AQI", prediction)
            with res_col2:
                # FIX 2: AQI thresholds now consistent with standard scale & Tab 2
                if prediction <= 50:
                    st.success("🟢 Good — Air is clean and safe.")
                elif prediction <= 100:
                    st.info("🟡 Moderate — Acceptable for most people.")
                elif prediction <= 150:
                    st.warning("🟠 Unhealthy for Sensitive Groups — Children, elderly, and asthma patients should reduce outdoor activity.")
                elif prediction <= 200:
                    st.error("🔴 Unhealthy — Everyone may start to experience health effects.")
                else:
                    st.error("🚨 Severe / Hazardous — Danger! Avoid all outdoor activity. Wear a mask!")

    # Fill Tab 4
    with tab4:
        st.header("📊 Dataset Preview & Visualizations")
        st.dataframe(data.head())
        st.write("### Pollution Trends")
        chart_data = clean_data[['PM2.5', 'PM10']].head(100)
        st.line_chart(chart_data)

# If they haven't uploaded the file yet, show a polite message in Tabs 3 and 4
else:
    with tab3:
        st.warning("🔒 Please upload your CSV file in the sidebar to unlock the AI Predictor.")
    with tab4:
        st.warning("🔒 Please upload your CSV file in the sidebar to view data graphs.")