import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from scipy.stats import zscore


WEATHERAPI_KEY = "4e419fdccbae443fac5161806250105"  

# --- Fetch Live Weather Data ---
def fetch_live_weather(city_name):
    url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={city_name}&days=7&aqi=no"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching live weather data: {e}")
        return None


# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Weather Forecast", layout="wide")
st.title("🌍 Clima AI - Weather Prediction and Climate Insights")



# --- Sidebar Location Picker ---
st.sidebar.header("📍 Location Settings")
city_input = st.sidebar.text_input("Enter City Name / Place", value="New Delhi")

@st.cache_data(show_spinner=False)
def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="weather_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return round(location.latitude, 4), round(location.longitude, 4)
        else:
            return None, None
    except:
        return None, None

latitude, longitude = get_coordinates(city_input)

if latitude is None or longitude is None:
    st.sidebar.error("❌ Could not find city. Please enter a valid city name.")
    st.stop()
else:
    st.sidebar.success(f"📍 Coordinates Found: {latitude}, {longitude}")

# --- Date Range Selection ---
st.sidebar.header("📆 Date Range")
min_date = datetime(2022, 1, 1)
max_date = datetime.today() - timedelta(days=2)  # 2 days before today

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

if start_date >= end_date:
    st.sidebar.error("❌ End date must be after start date")
    st.stop()

# --- Fetch Historical Weather Data ---
@st.cache_data(show_spinner=True)
def fetch_weather_data(latitude, longitude, start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"wind_speed_10m_max,relative_humidity_2m_mean"
        f"&timezone=auto"
    )
    res = requests.get(url).json()
    if 'daily' in res:
        df = pd.DataFrame(res['daily'])
        df.rename(columns={
            'temperature_2m_max': 'temp_max',
            'temperature_2m_min': 'temp_min',
            'precipitation_sum': 'rain',
            'wind_speed_10m_max': 'wind',
            'relative_humidity_2m_mean': 'humidity',
            'time': 'date'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['target'] = df['temp_max'].shift(-1)
        df.dropna(inplace=True)
        return df
    else:
        return pd.DataFrame()

df = fetch_weather_data(latitude, longitude, start_date, end_date)

if df.empty:
    st.error("❌ Weather data not found for this location.")
    st.stop()

# --- Train Model ---
X = df[['temp_max', 'temp_min', 'rain', 'wind', 'humidity']]
y = df['target']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- User Selections ---
st.sidebar.header("🗕️ Weather Prediction Filters")
year = st.sidebar.selectbox("Select Year", sorted(df['date'].dt.year.unique(), reverse=True))
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

filtered_df = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]
if filtered_df.empty:
    st.warning("⚠️ No weather data for the selected month and year.")
    st.stop()

# --- Latest Day Data + Prediction ---
today = filtered_df.iloc[-1]
today_input = today[['temp_max', 'temp_min', 'rain', 'wind', 'humidity']].values.reshape(1, -1)
predicted_temp = model.predict(today_input)[0]

# --- Weather Icons Function ---
def get_weather_icon(temp_max, rain):
    if rain > 0:
        return "🌧️"  # Rainy weather
    elif temp_max > 30:
        return "☀️"  # Hot weather
    elif temp_max < 10:
        return "❄️"  # Cold weather
    else:
        return "☁️"  # Cloudy weather
    
# Function to provide weather recommendations
def get_weather_recommendation(temp_c, humidity, wind_speed, rain_mm):
    recommendations = []

    if temp_c > 35:
        recommendations.append("🌞 Stay hydrated! It's very hot. Avoid going out at noon.")
        recommendations.append("🧴 Wear sunscreen and light clothing.")
    elif temp_c < 10:
        recommendations.append("❄️ It's cold. Wear warm clothes and consider staying indoors.")

    if rain_mm > 1.0:
        recommendations.append("🌧️ Carry an umbrella or raincoat. There's a chance of rain.")
        if rain_mm > 10:
            recommendations.append("⚠️ Heavy rainfall expected. Avoid unnecessary travel.")

    if wind_speed > 30:
        recommendations.append("💨 Strong winds today. Be cautious if you're going outside.")
    
    if humidity > 80:
        recommendations.append("😓 High humidity. It might feel hotter than usual.")
    elif humidity < 20:
        recommendations.append("💧 Low humidity. Use moisturizer and drink plenty of water.")

    if not recommendations:
        recommendations.append("✅ Weather looks good. Ideal for outdoor activities!")

    return recommendations


# --- Tabs Layout ---
tab1, tab2, tab3,tab4= st.tabs(["📊 Overview", "📈 Trends & Insights", "🔧 Custom Forecast","💡 Recommendations" ])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("🌤️ Today's Weather Overview")

    live_weather = fetch_live_weather(city_input)
    if live_weather is None:
        st.warning("⚠️ Unable to fetch live weather data.")
    else:
        current = live_weather['current']
        forecast = live_weather['forecast']['forecastday'][0]['day']

        weather_icon = get_weather_icon(forecast['maxtemp_c'], forecast['totalprecip_mm'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Max Temp (°C)", f"{forecast['maxtemp_c']:.1f}")
        col2.metric("Min Temp (°C)", f"{forecast['mintemp_c']:.1f}")
        col3.metric("Humidity (%)", f"{forecast['avghumidity']:.1f}")

        col4, col5 = st.columns(2)
        col4.metric("Wind Speed (km/h)", f"{forecast['maxwind_kph']:.1f}")
        col5.metric("Rainfall (mm)", f"{forecast['totalprecip_mm']:.1f}")

        

        st.markdown(f"### ⏩ 5-Day Max Temperature Forecast")
        forecast_data = []
        last_features = today[['temp_max', 'temp_min', 'rain', 'wind', 'humidity']].values.reshape(1, -1)
        for i in range(5):
           pred = model.predict(last_features)[0] + np.random.uniform(-2, 2)
           forecast_data.append({
            'Date': (datetime.today() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            'Predicted Max Temp (°C)': round(pred, 2)
        })
        last_features[0][0] = pred
        st.table(pd.DataFrame(forecast_data))


        # Historical Temperature Trend (Interactive Plot)
        st.markdown(f"### 📊 Historical Temperature Trends ({city_input})")
        historical_df = df[df['date'].dt.month == month]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=historical_df['date'], y=historical_df['temp_max'],
                                 mode='lines', name='Max Temp (°C)', line=dict(color='orange')))
        fig1.add_trace(go.Scatter(x=historical_df['date'], y=historical_df['temp_min'],
                                 mode='lines', name='Min Temp (°C)', line=dict(color='skyblue')))

        fig1.update_layout(
            title=f"Temperature Trends in {city_input} - {month}/{year}",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',  # Display data on hover
            template='plotly_dark'
        )
        st.plotly_chart(fig1)


    # --- Evaluation Section ---
    st.subheader("📊 Model Performance Metrics")
    st.write("### Feature Importances")
    st.dataframe(pd.DataFrame({
        'Feature': ['Max Temp', 'Min Temp', 'Rain', 'Wind', 'Humidity'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False), use_container_width=True)

    st.write(f"### Model R² Score: **{model.score(X, y):.4f}**")

    # --- Data Download ---
    st.subheader("📅 Download Cleaned Weather Dataset")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"{city_input.lower().replace(' ', '_')}_weather_data.csv",
        mime='text/csv'
    )

# --- Tab 2: Trends & Insights ---
with tab2:
    st.subheader("📊 Yearly Average Max/Min Temperature")
    df['year'] = df['date'].dt.year
    yearly_avg = df.groupby('year')[['temp_max', 'temp_min']].mean().reset_index()

    fig2, ax2 = plt.subplots()
    ax2.bar(yearly_avg['year'] - 0.2, yearly_avg['temp_max'], width=0.4, label='Avg Max Temp', color='orange')
    ax2.bar(yearly_avg['year'] + 0.2, yearly_avg['temp_min'], width=0.4, label='Avg Min Temp', color='skyblue')
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("Yearly Avg Max/Min Temperature")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("🚨 Anomaly Detection for Selected Month & Year")

    # Filter data for selected month and year
    anomaly_df = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)].copy()
    if anomaly_df.empty:
        st.warning("No data available for anomaly detection in selected month/year.")
    else:
        features = ['temp_max', 'temp_min', 'rain', 'wind', 'humidity']

        # Normalize using z-score
        for feature in features:
            mean = anomaly_df[feature].mean()
            std = anomaly_df[feature].std()
            anomaly_df[f'{feature}_z'] = (anomaly_df[feature] - mean) / std

        # Define anomaly threshold
        threshold = 2  # |z| > 2 considered anomaly
        fig4 = go.Figure()

        # Plot each feature and highlight anomalies
        colors = {
            'temp_max': 'orange',
            'temp_min': 'skyblue',
            'rain': 'green',
            'wind': 'purple',
            'humidity': 'blue'
        }

        for feature in features:
            z_col = f'{feature}_z'
            is_anomaly = anomaly_df[z_col].abs() > threshold

            fig4.add_trace(go.Scatter(
                x=anomaly_df['date'], y=anomaly_df[z_col],
                mode='lines+markers',
                name=f'{feature} (z)',
                line=dict(color=colors[feature]),
                marker=dict(size=6),
                hovertemplate=f"{feature}: %{{y:.2f}}<br>Date: %{{x}}"
            ))

            fig4.add_trace(go.Scatter(
                x=anomaly_df[is_anomaly]['date'],
                y=anomaly_df[is_anomaly][z_col],
                mode='markers',
                name=f'{feature} anomaly',
                marker=dict(size=10, color='red', symbol='x'),
                hovertemplate=f"🚨 {feature} anomaly<br>Date: %{{x}}<br>Z-score: %{{y:.2f}}"
            ))

        fig4.update_layout(
            title=f"Z-Score Based Anomaly Detection - {month}/{year}",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            hovermode='x unified',
            template='plotly_dark',
            legend=dict(orientation='h')
        )

        st.plotly_chart(fig4, use_container_width=True)


    st.subheader("🌧️ Monthly Rainfall Trends")
    df['month'] = df['date'].dt.to_period("M")
    monthly_rain = df.groupby('month')['rain'].mean().reset_index()
    monthly_rain['month'] = monthly_rain['month'].astype(str)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(monthly_rain['month'], monthly_rain['rain'], marker='o', color='green')
    ax3.set_title("Monthly Avg Rainfall Over Time")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Rainfall (mm)")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    st.subheader("🌬️ Yearly Average Wind Speed and Humidity")
    yearly_wind_humidity = df.groupby('year')[['wind', 'humidity']].mean().reset_index()

    fig3, ax3 = plt.subplots()
    ax3.plot(yearly_wind_humidity['year'], yearly_wind_humidity['wind'], marker='o', label='Avg Wind Speed (km/h)', color='green')
    ax3.plot(yearly_wind_humidity['year'], yearly_wind_humidity['humidity'], marker='x', label='Avg Humidity (%)', color='blue')
    
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Wind Speed (km/h) / Humidity (%)")
    ax3.set_title("Yearly Avg Wind Speed and Humidity")
    ax3.legend()
    st.pyplot(fig3)

# --- Tab 3: Custom Forecast ---
with tab3:
    st.subheader("🛠️ Predict Max Temperature from Custom Weather Inputs")

    custom_temp_max = st.number_input("Max Temperature (°C)", -50, 60, value=30)
    custom_temp_min = st.number_input("Min Temperature (°C)", -50, 50, value=20)
    custom_rain = st.number_input("Rainfall (mm)", 0, 500, value=5)
    custom_wind = st.slider("Wind Speed (km/h)", 0, 150, 10)
    custom_humidity = st.slider("Humidity (%)", 0, 100, 50)

    if st.button("🔮 Predict Temperature"):
        custom_input = np.array([[custom_temp_max, custom_temp_min, custom_rain, custom_wind, custom_humidity]])
        custom_prediction = model.predict(custom_input)[0]
        st.success(f"📈 Predicted Max Temperature: **{custom_prediction:.2f} °C**")

# --- Tab 4: Smart Weather Recommendations ---
with tab4:
    st.subheader("💡 Smart Weather-Based Recommendations")

    if live_weather is None:
        st.warning("⚠️ Unable to fetch live weather data.")
    else:
        forecast = live_weather['forecast']['forecastday'][0]['day']
        recommendations = get_weather_recommendation(
            temp_c=forecast['maxtemp_c'],
            humidity=forecast['avghumidity'],
            wind_speed=forecast['maxwind_kph'],
            rain_mm=forecast['totalprecip_mm']
        )

        for recommendation in recommendations:
            st.info(recommendation)


st.caption("© 2025 Global Weather AI Forecasting App | Powered by Open-Meteo + Streamlit")

